import torch.nn.functional as F


def get_layer_sizes(n_layers, hidden_size, first_size=None, last_size=None):
    for i in range(n_layers):
        if i == 0 and first_size is not None:
            yield first_size, hidden_size
        elif i == (n_layers - 1) and last_size is not None:
            yield hidden_size, last_size
        else:
            yield hidden_size, hidden_size


def apply_layer_dropout_relu(x, layers, p, training):
    for i, lin in enumerate(layers):
        if i != 0:
            x = F.relu(x)
            x = F.dropout(x, p=p, training=training)
        x = lin(x)
    return x
