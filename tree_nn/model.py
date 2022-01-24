import torch
from torch.nn import Linear
import torch.nn.functional as F
import treelstm
import numpy as np

def train(model: torch.nn.Module, train_loader, n_epochs=5, optimizer=None, criterion=None, verbose=False,
          transform=None):
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = criterion if criterion is not None else torch.nn.MSELoss()

    epoch_losses = []
    for e in range(n_epochs):
        model.train()
        batch_losses = []

        for data in train_loader:
            data = data if transform is None else transform(data)
            features = data['features'] if 'features' in data else None
            
            out = model(data['x'], features=features)  # Perform a single forward pass.
            # c1 = torch.combinations(data['y'])
            # c2 = torch.combinations(out)
            # c1 = c1[:, 0] - c1[:, 1]
            # c2 = c2[:, 0] - c2[:, 1]
            loss = criterion(data['y'], out)
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


class TreeLSTMModel(torch.nn.Module):
    def __init__(self, n_node_features, n_hidden=32, n_lin=2, dropout=0.1, use_root=False, n_features=None):

        super().__init__()
        self.dropout = dropout

        self.lstm = treelstm.TreeLSTM(n_node_features, n_hidden)
        self.concat_lin = None
        if n_features is not None:
            self.concat_lin = Linear(n_hidden + n_features, n_hidden)

        self.lins = torch.nn.ModuleList([Linear(n_hidden, n_hidden) for _ in range(n_lin)])
        self.lin = Linear(n_hidden, 1)

        if n_features is not None:
            self.concat_lin = Linear(n_hidden + n_features, n_hidden)

        self.use_root = use_root

    def forward(self, x, features=None):
        batch_info = x['tree_sizes']
        x, c = self.lstm(x['features'],x['node_order'],x['adjacency_list'],x['edge_order'])
        x = treelstm.util.unbatch_tree_tensor(x, batch_info)
        if self.use_root:
            x = torch.vstack([i[0] for i in x])
        else:
            x = torch.vstack([torch.mean(i, axis=0) for i in x])
        
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.concat_lin:
            #features = torch.tanh(features)
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