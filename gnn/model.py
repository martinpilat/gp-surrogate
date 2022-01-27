import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GIN, global_mean_pool, GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import GCNNorm


def to_dataset(ind_graphs, y_accuracies=None, x_features=None, batch_size=32, shuffle=True):
    y_accuracies = y_accuracies if y_accuracies is not None else [None] * len(ind_graphs)
    x_features = x_features if x_features is not None else [None] * len(ind_graphs)

    data = [Data(x=ind[0], edge_index=ind[1], y=y, features=feat)
            for ind, y, feat in zip(ind_graphs, y_accuracies, x_features)]
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train(model: torch.nn.Module, train_loader, n_epochs=5, optimizer=None, criterion=None, verbose=True,
          transform=None, ranking=False, mse_both=False):
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = criterion if criterion is not None else torch.nn.MSELoss()
    ranking_criterion = torch.nn.MarginRankingLoss()

    epoch_losses = []
    for e in range(n_epochs):
        model.train()
        batch_losses = []

        prev = None
        for data in train_loader:
            data = data if transform is None else transform(data)
            features = data.features if 'features' in data else None

            out = model(data.x, data.edge_index, data.batch, features=features)  # Perform a single forward pass.
            loss = criterion(out, data.y)
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


class GINModel(torch.nn.Module):
    def __init__(self, n_node_features, n_hidden=64, n_convs=3, n_lin=2, dropout=0.1, use_root=False, n_features=None):

        super().__init__()
        self.dropout = dropout

        self.gin = GIN(n_node_features, n_hidden, n_convs, dropout=dropout)
        #self.gin = GCN(n_node_features, n_hidden, n_convs, dropout=dropout)

        if n_features is not None:
            self.concat_lin = Linear(n_hidden + n_features, n_hidden)

        self.lins = torch.nn.ModuleList([Linear(n_hidden, n_hidden) for _ in range(n_lin)])
        self.lin = Linear(n_hidden, 1)

        self.use_root = use_root

    def forward(self, x, edge_index, batch, features=None):
        flags = None
        if self.use_root:
            x, flags = x[:, :-1], x[:, -1]

        x = self.gin(x, edge_index)
        if self.use_root:
            x = x[flags.type(torch.bool)]
        else:
            x = global_mean_pool(x, batch)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if features is not None:
            x = torch.concat([x, features], dim=-1)
            x = self.concat_lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for l in self.lins:
            x = l(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)

        return torch.flatten(x)
