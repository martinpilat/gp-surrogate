import torch
from torch.nn import Linear
import torch.nn.functional as F
import treelstm
import numpy as np

def train(model: torch.nn.Module, train_loader, n_epochs=5, optimizer=None, criterion=None, verbose=True,
          transform=None, ranking=False, mse_both=False, use_auxiliary=False, auxiliary_weight=0.0):
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
            features = data['features'] if 'features' in data else None
            
            out, aux_out = model(data['x'], features=features, aux_x=data['aux_x'])  # Perform a single forward pass.
            loss = criterion(out, data['y'])
            if ranking and prev is not None and len(prev[0]['y']) == len(data['y']):
                data_prev, feats_prev = prev
                out_prev, _ = model(data_prev['x'], features=feats_prev)

                y = torch.where(data['y'] > data_prev['y'], 1, -1)
                loss += ranking_criterion(out, out_prev, y)
                if mse_both:
                    loss += criterion(out_prev, data_prev['y'])

            if use_auxiliary: 
                aux_loss = aux_criterion(aux_out, data['aux_y'].flatten())
                # print(f'loss = {loss}, aux_loss = {aux_loss}')
                loss += auxiliary_weight*aux_loss

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


class TreeLSTMModel(torch.nn.Module):
    def __init__(self, n_node_features, n_hidden=32, n_lin=2, dropout=0.1, use_root=False, n_features=None, use_auxiliary=False, auxiliary_weight=0.1, n_aux_inputs=None, n_aux_outputs=None):

        super().__init__()
        self.dropout = dropout

        self.lstm = treelstm.TreeLSTM(n_node_features, n_hidden)
        self.concat_lin = None
        self.n_features = n_features
        self.aux_concat_lin = None
        self.aux_lins = None
        self.aux_lin = None
        self.aux_weight = auxiliary_weight
        self.use_auxiliary = use_auxiliary

        self.lins = torch.nn.ModuleList([Linear(n_hidden, n_hidden) for _ in range(n_lin)])
        self.lin = Linear(n_hidden, 1)

        if n_features is not None:
            self.concat_lin = Linear(n_hidden + n_features, n_hidden)

        self.use_root = use_root

        if use_auxiliary:
            self.aux_concat_lin = Linear(n_hidden + (n_features if n_features else 0) + n_aux_inputs, n_hidden) #TODO: separate parameter for aux_hidden
            self.aux_lins = torch.nn.ModuleList([Linear(n_hidden, n_hidden) for _ in range(n_lin)])
            self.aux_lin = Linear(n_hidden, n_aux_outputs)

    def forward(self, x, features=None, aux_x=None):
        batch_info = x['tree_sizes']
        x, c = self.lstm(x['features'],x['node_order'],x['adjacency_list'],x['edge_order'])
        x = treelstm.util.unbatch_tree_tensor(x, batch_info)
        if self.use_root:
            x = torch.vstack([i[0] for i in x])
        else:
            x = torch.vstack([torch.mean(i, axis=0) for i in x])
        
        embed = x

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

        if self.use_auxiliary and aux_x is not None:
            if self.concat_lin: 
                embed = torch.concat([embed, features], dim=-1)              
            aux_x = torch.cat([embed.unsqueeze(1).expand(-1, 20, -1), aux_x], dim=2).reshape(-1, embed.shape[-1]+aux_x.shape[-1])
            aux_x = self.aux_concat_lin(aux_x)
            aux_x = F.relu(aux_x)
            aux_x = F.dropout(aux_x, p=self.dropout, training=self.training)

            for l in self.aux_lins:
                aux_x = l(aux_x)
                aux_x = F.relu(aux_x, inplace=True)
                aux_x = F.dropout(aux_x, p=self.dropout, training=self.training)
            
            aux_x = self.aux_lin(aux_x)

        return torch.flatten(x), torch.flatten(aux_x) if aux_x is not None else None