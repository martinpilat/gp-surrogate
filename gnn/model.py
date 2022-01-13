import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GIN, global_mean_pool, GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import GCNNorm


def to_dataset(ind_graphs, y_accuracies=None, batch_size=32, shuffle=True):
    y_accuracies = y_accuracies if y_accuracies is not None else [None] * len(ind_graphs)

    data = [Data(x=ind[0], edge_index=ind[1], y=y) for ind, y in zip(ind_graphs, y_accuracies)]
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train(model: torch.nn.Module, train_loader, n_epochs=5, optimizer=None, criterion=None, verbose=True,
          transform=None):
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = criterion if criterion is not None else torch.nn.MSELoss()

    epoch_losses = []
    for e in range(n_epochs):
        model.train()
        batch_losses = []

        for data in train_loader:
            data = data if transform is None else transform(data)

            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.detach().cpu().numpy())

        e_loss = np.mean(batch_losses)
        e_std = np.std(batch_losses)
        if verbose:
            print(f"Epoch {e} loss: mean {e_loss}, std {e_std}")
            epoch_losses.append(e_loss)

    return epoch_losses


class GINModel(torch.nn.Module):
    def __init__(self, n_node_features, n_hidden=16, n_convs=2, n_lin=0, dropout=0.1, use_root=False):

        super().__init__()
        self.dropout = dropout

        self.gin = GIN(n_node_features, n_hidden, n_convs, dropout=dropout)
        #self.gin = GCN(n_node_features, n_hidden, n_convs, dropout=dropout)
        self.lins = torch.nn.ModuleList([Linear(n_hidden, n_hidden) for _ in range(n_lin)])
        self.lin = Linear(n_hidden, 1)

        self.use_root = use_root

    def forward(self, x, edge_index, batch):
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

        for l in self.lins:
            x = l(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        #x = F.sigmoid(x)

        return torch.flatten(x)

# features co jsou spocitany do toho
# vyndat z neuronky repre co se naučila a dát to jinému modelu
