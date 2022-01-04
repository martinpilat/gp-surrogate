import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GIN, global_mean_pool
from torch_geometric.data import Data, DataLoader


def to_dataset(ind_graphs, y_accuracies, batch_size=32, shuffle=True):
    data = [Data(x=ind[0], edge_index=ind[1], y=y) for ind, y in zip(ind_graphs, y_accuracies)]
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

class GINModel(torch.nn.Module):
    def __init__(self, n_node_features, n_hidden=16, n_convs=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.gin = GIN(n_node_features, n_hidden, n_convs, dropout=dropout)
        self.lin = Linear(n_hidden, 1)

    def forward(self, x, edge_index, batch):
        x = self.gin(x, edge_index)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x
