import os
import random
import pickle
import pygraphviz as pgv

import numpy as np
import scipy
import scipy.stats
import sklearn.metrics
from deap import creator, base, gp

from gp_surrogate import surrogate
from gp_surrogate.benchmarks import bench_by_name


def plot_best_tree(inds, fitness, save_path):
    best_id = np.argmin(fitness)

    nodes, edges, labels = gp.graph(inds[best_id])
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for ni in nodes:
        n = g.get_node(ni)
        n.attr["label"] = labels[ni]

    g.draw(save_path)
    return inds[best_id], fitness[best_id]


def inds_to_str(inds):
    return [str(ind) for ind in inds]


def get_common_inds(inds1, inds2):
    inds1, inds2 = set(inds_to_str(inds1)), set(inds_to_str(inds2))
    return inds1.intersection(inds2)


def load_dataset(file_list, dir_path=None, data_size=None, unique_only=False):
    if not len(file_list):
        return None

    if dir_path is not None:
        file_list = [os.path.join(dir_path, f) for f in file_list]

    inds = []
    fitness = []
    for file in file_list:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            inds.extend(data[0])
            fitness.extend(data[1])

    fitness = [f[0] for f in fitness]

    if unique_only:
        ind_str = np.array([str(ind) for ind in inds])
        _, idxs = np.unique(ind_str, return_index=True)
        inds = [inds[i] for i in idxs]
        fitness = [fitness[i] for i in idxs]

    if data_size is not None:
        indices = random.sample(range(len(inds)), data_size)
        inds = [inds[i] for i in indices]
        fitness = [fitness[i] for i in indices]

    return inds, fitness


def get_model_class(name):
    if name == 'GNN':
        surrogate_cls = surrogate.NeuralNetSurrogate
    elif name == 'TNN':
        surrogate_cls = surrogate.TreeLSTMSurrogate
    else:
        raise ValueError(f"Invalid surrogate name: {name}, valid: (GNN, TNN).")
    return surrogate_cls


def get_files_by_index(files, index_list):
    res_files = []

    for file in files:
        gen_id = int(file.split('.')[-2])
        if gen_id in index_list:
            res_files.append(file)

    return res_files


def init_bench(data_dir):
    all_files = os.listdir(data_dir)
    bench_name = all_files[0].split('.')[-7]

    bench_description = bench_by_name(bench_name)
    pset = bench_description['pset']

    # create the types for fitness and individuals
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    return all_files, bench_description, pset


def top_third(sorted_ix):
    bad_ix = sorted_ix[-int(2 * len(sorted_ix) / 3):]

    mask = np.ones_like(sorted_ix)
    mask[bad_ix] = 0
    return mask


def get_rank_scores(sort_ids):
    scores = np.zeros_like(sort_ids)
    for pos, index in enumerate(sort_ids):
        scores[index] = pos

    return len(sort_ids) - scores


def _mask_out(preds, y_true, mask=None):
    return (preds, y_true) if mask is None else (preds[mask], y_true[mask])


def eval_metrics(preds, y_true, mask=None):
    """
    mask: list of bools for np array filtering
    """
    res = {}
    cpreds, cy_true = _mask_out(preds, y_true, mask=mask)

    res['spearman'] = scipy.stats.spearmanr(cpreds, cy_true).correlation
    tau, _ = scipy.stats.kendalltau(cpreds, cy_true)
    res['tau'] = tau

    sorted_preds = np.argsort(preds)
    sorted_y_true = np.argsort(y_true)
    if mask is None:
        score_preds, score_y_true = get_rank_scores(sorted_preds), get_rank_scores(sorted_y_true)
        res['ndcg'] = sklearn.metrics.ndcg_score(score_preds.reshape(1, -1), score_y_true.reshape(1, -1))

    top_preds = top_third(sorted_preds)
    top_y_true = top_third(sorted_y_true)
    top_preds, top_y_true = _mask_out(top_preds, top_y_true, mask=mask)

    tp = np.sum(top_preds & top_y_true)
    res['precision_tt'] = tp / np.sum(top_preds)  # TP + FP
    res['recall_tt'] = tp / np.sum(top_y_true)  # all positives
    res['accuracy'] = np.sum(top_preds == top_y_true) / len(top_preds)

    return res
