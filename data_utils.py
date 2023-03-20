import os
import random
import pickle

import numpy as np
import scipy
import scipy.stats
import sklearn.metrics
from deap import creator, base, gp

from gp_surrogate import surrogate
from gp_surrogate.benchmarks import bench_by_name


def load_dataset(file_list, dir_path=None, data_size=None):
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
    bench_name = all_files[0].split('.')[2]

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


def eval_metrics(preds, y_true):
    corr = scipy.stats.spearmanr(preds, y_true).correlation
    tau, _ = scipy.stats.kendalltau(preds, y_true)

    sorted_preds = np.argsort(preds)
    sorted_y_true = np.argsort(y_true)
    ndcg = sklearn.metrics.ndcg_score(sorted_preds.reshape(1, -1), sorted_y_true.reshape(1, -1))

    top_preds = top_third(sorted_preds)
    top_y_true = top_third(sorted_y_true)

    tp = np.sum(top_preds & top_y_true)
    precision = tp / np.sum(top_preds)  # TP + FP
    recall = tp / np.sum(top_y_true)  # all positives
    accuracy = np.sum(top_preds == top_y_true) / len(top_preds)

    return {'spearman': corr, 'tau': tau, 'ndcg': ndcg, 'precision_tt': precision, 'recall_tt': recall,
            'accuracy_tt': accuracy}
