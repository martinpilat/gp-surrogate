import torch
import treelstm

from gp_surrogate.models.gnn.graph import node_features

def ind_to_tree(ind, template):
    if ind[0].arity == 0:
        return {'features': node_features(ind[0], template), 'children': [], 'labels': None}, ind[1:]
    children = []
    r = ind[1:]
    for i in range(ind[0].arity):
        c, r = ind_to_tree(r, template)
        children.append(c)
    return {'features': node_features(ind[0], template), 'children': children, 'labels': None}, r

def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list

def convert_tree_to_tensors(tree, device=torch.device('cpu')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, 'features')
    labels = _gather_node_attributes(tree, 'labels')
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = treelstm.calculate_evaluation_orders(adjacency_list, len(features))

    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        #'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }

def _parse_tree(self, ind):
    if ind[0].arity == 0:
        return {'features': node_features(ind[0], self.feature_template), 
                'children': [], 
                'labels': [0]}, ind[1:]
    children = []
    r = ind[1:]
    for i in range(ind[0].arity):
        c, r = self._parse_tree(r)
        children.append(c)
    return {'features': node_features(ind[0], self.feature_template), 
            'children': children, 
            'labels': [0]}, r

def _create_dataset(self, inds, fitness, first_gen=False):
    trees = [convert_tree_to_tensors(self._parse_tree(i)[0]) for i in inds]
    return trees