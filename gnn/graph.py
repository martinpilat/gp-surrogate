import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse


def gen_feature_vec_template(pset):
    prim_list = pset.primitives[object]
    args = pset.arguments

    possible = [p.name for p in prim_list]
    possible.extend(args)
    return possible


def node_features(node, template):
    vec = []
    found = False
    # function
    for name in template:
        if node.name == name:
            vec.append(1)
            found = True
        else:
            vec.append(0)

    # constant terminal
    if found:
        vec.append(0)
    else:
        vec.append(node.value)

    return vec


def compile_tree(individual, feature_template):
    n_nodes = len(individual)
    x_features = np.zeros((n_nodes, len(feature_template) + 1))
    adjacency = np.zeros((n_nodes, n_nodes))
    stack = []

    for i, node in enumerate(reversed(individual)):
        features = node_features(node, feature_template)
        x_features[i] = features

        if node.arity > 0:
            children = stack[-node.arity:]
            stack = stack[:-node.arity]

            for child in children:
                adjacency[child, i] = 1

        stack.append(i)

    assert len(stack) == 1, "Invalid tree"

    x_features, adjacency = torch.Tensor(x_features), torch.Tensor(adjacency)

    # dense_to_sparse returns (edge_index, edge_attributes)
    return x_features, dense_to_sparse(adjacency)[0]
