import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gp_surrogate.models.utils import get_layer_sizes, apply_layer_dropout_relu


def to_dataset(ind_graphs, y_accuracies=None, x_features=None, batch_size=32, shuffle=True, aux=None):
    y_accuracies = y_accuracies if y_accuracies is not None else [None] * len(ind_graphs)
    x_features = x_features if x_features is not None else [None] * len(ind_graphs)

    if aux is not None:
        aux_in = [a[0] for a in aux]
        aux_out = [a[1] for a in aux]
    else:
        aux_in = [None] * len(ind_graphs)
        aux_out = [None] * len(ind_graphs)

    data = [Data(x=ind[0], edge_index=ind[1], y=y, features=feat, aux_in=ain, aux_out=aout)
            for ind, y, feat, ain, aout in zip(ind_graphs, y_accuracies, x_features, aux_in, aux_out)]
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train_gnn(model: torch.nn.Module, train_loader, n_epochs=5, optimizer=None, criterion=None, verbose=True,
              transform=None, ranking=False, mse_both=False, auxiliary_weight=0.0, device=None):
    model = model.to(device)

    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = criterion if criterion is not None else torch.nn.MSELoss()
    aux_criterion = torch.nn.MSELoss()
    ranking_criterion = torch.nn.MarginRankingLoss()

    epoch_losses = []
    for e in range(n_epochs):
        model.train()
        batch_losses = []

        prev = None
        for data in train_loader:
            data = data if transform is None else transform(data)
            data = data.to(device)
            features = data.features if 'features' in data else None
            aux_in = torch.tensor(data.aux_in) if 'aux_in' in data else None
            aux_out = torch.tensor(data.aux_out) if 'aux_out' in data else None

            out, pred_aux = model(data.x, data.edge_index, data.batch, features=features, aux_in=aux_in)  # Perform a single forward pass.

            # prediction loss
            loss = criterion(out, data.y)

            # aux loss
            if pred_aux is not None:
                aux_loss = aux_criterion(aux_out.flatten(), pred_aux.flatten())
                if verbose:
                    print(f'loss = {loss}, aux_loss = {aux_loss}')
                loss += auxiliary_weight * aux_loss

            # ranking loss
            if ranking and prev is not None and len(prev[0].y) == len(data.y):
                data_prev, feats_prev = prev
                out_prev = model(data_prev.x, data_prev.edge_index, data_prev.batch, features=feats_prev)

                y = torch.where(data.y > data_prev.y, 1, -1)
                loss += ranking_criterion(out, out_prev, y)
                if mse_both:
                    loss += criterion(out_prev, data_prev.y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ranking:
                prev = (data, features)

            batch_losses.append(loss.detach().cpu().numpy())

        e_loss = np.mean(batch_losses)
        e_std = np.std(batch_losses)
        if verbose:
            print(f"Epoch {e} loss: mean {e_loss}, std {e_std}")
            epoch_losses.append(e_loss)

    return epoch_losses


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_linear=2):
        super().__init__()

        sizes = get_layer_sizes(n_linear, hidden_dim, first_size=input_dim)
        self.linears = nn.ModuleList([nn.Linear(indim, outdim) for indim, outdim in sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_linear - 1)])

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            if i != 0:
                x = self.batch_norms[i - 1](x)
                x = F.relu(x)
            x = lin(x)
        return x


def _get_MLP(n_in, n_hidden, n_linear, i):
    return MLP(n_in if i == 0 else n_hidden, n_hidden, n_linear=n_linear)


class GINConcat(torch.nn.Module):
    def __init__(self, n_node_features, n_hidden=64, n_convs=3, n_linear=2, n_mlp_linear=2, dropout=0.1,
                 n_hidden_linear=64, n_features=None, use_auxiliary=False, readout='concat',
                 n_aux_inputs=None, n_aux_outputs=None, aux_hidden=32, aux_sample_size=20):
        super().__init__()
        if readout not in ['concat', 'root', 'mean']:
            raise ValueError(f"Readout must be one of , passed value: {self.readout}")

        # convs are followed by batch norm and ReLU
        self.convs = nn.ModuleList(
            [GINConv(_get_MLP(n_node_features, n_hidden, n_mlp_linear, i)) for i in range(n_convs)]
        )
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_convs)])

        # compute input size of linear layers
        n_features = 0 if n_features is None else n_features
        lin_dim = (n_hidden * n_convs + n_node_features) if readout == 'concat' else n_hidden

        lin_sizes = get_layer_sizes(n_linear, n_hidden_linear, first_size=lin_dim + n_features, last_size=1)
        self.lins = torch.nn.ModuleList([Linear(indim, outdim) for indim, outdim in lin_sizes])

        self.dropout = dropout
        self.readout = readout
        self.use_auxiliary = use_auxiliary
        self.aux_sample_size = aux_sample_size

        if use_auxiliary:
            lin_sizes = get_layer_sizes(n_linear, aux_hidden,
                                        first_size=lin_dim + n_features + n_aux_inputs, last_size=n_aux_outputs)
            self.aux_lins = torch.nn.ModuleList([Linear(indim, outdim) for indim, outdim in lin_sizes])

    def forward(self, x, edge_index, batch, features=None, aux_in=None):
        # flags if root features are GIN's output
        flags = None
        if self.readout == 'root':
            x, flags = x[:, :-1], x[:, -1]

        # concat readout - inputs
        h_out = []
        if self.readout == 'concat':
            h_out.append(global_add_pool(x, batch))

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # concat readout - hidden
            if self.readout == 'concat':
                h_out.append(global_add_pool(x, batch))

        if self.readout == 'concat':
            h = torch.cat(h_out, dim=1)  # concat readout
        elif self.readout == 'root':
            h = x[flags.type(torch.bool)]  # root features readout
        elif self.readout == 'mean':
            h = global_mean_pool(x, batch)  # global avg pooling readout
        else:
            raise ValueError(f"Invalid readout: {self.readout}")

        # add manual features
        if features is not None:
            h = torch.concat([h, features], dim=-1)

        embed = h

        # linear layers
        h = apply_layer_dropout_relu(h, self.lins, self.dropout, self.training)

        # aux inputs
        if self.use_auxiliary and aux_in is not None:
            aux_x = torch.cat([embed.unsqueeze(1).expand(-1, self.aux_sample_size, -1), aux_in], dim=2)
            aux_x = aux_x.reshape(-1, embed.shape[-1] + aux_in.shape[-1])

            aux_x = apply_layer_dropout_relu(aux_x, self.aux_lins, self.dropout, self.training)

            return torch.flatten(h), torch.flatten(aux_x) if aux_x is not None else None

        return torch.flatten(h), None
