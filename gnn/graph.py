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
        vec.append(0)
    else:
        vec.append(1)
        vec.append(node.value)

    return vec


def compile_tree(individual, feature_template, use_root=False, use_global_node=False):
    assert not (use_root and use_global_node), "Only one of use_root and use_global_node can be used."
    n_nodes = len(individual)
    n_nodes = (n_nodes + 1) if use_global_node else n_nodes
    n_feats = len(feature_template) + 2
    n_feats = (n_feats + 1) if use_root or use_global_node else n_feats

    x_features = np.zeros((n_nodes, n_feats))
    adjacency = np.zeros((n_nodes, n_nodes))
    stack = []

    for i, node in enumerate(reversed(individual)):
        features = node_features(node, feature_template)
        if use_root:
            features.append(1 if (i == len(individual) - 1) else 0)

        if use_global_node:
            features.append(0)

        x_features[i] = features
        adjacency[i, -1] = 1
        adjacency[-1, i] = 1

        if node.arity > 0:
            children = stack[-node.arity:]
            stack = stack[:-node.arity]

            for child in children:
                adjacency[child, i] = 1
                adjacency[i, child] = 1

        stack.append(i)
    #for i in range(n_nodes):
    #    adjacency[i, i] = 1
    if use_global_node:
        x_features[-1, -1] = 1

    assert len(stack) == 1, "Invalid tree"

    x_features, adjacency = torch.Tensor(x_features), torch.Tensor(adjacency)

    # dense_to_sparse returns (edge_index, edge_attributes)
    return x_features, dense_to_sparse(adjacency)[0]
