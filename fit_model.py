import argparse
import os
import pickle
import random

import scipy.stats
import torch
from deap import creator, base, gp
import optuna
import functools

from gp_surrogate import benchmarks, surrogate
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


def objective(trial, train_set, val_set, n_features, n_aux_inputs, n_aux_outputs):
    
    readout = trial.suggest_categorical('readout', ['concat', 'root', 'mean'])
    use_global_node = False
    # if readout != 'root':
    #     use_global_node = trial.suggest_categorical('use_global_node', [False, True])
    include_features = trial.suggest_categorical('include_features', [False, True])
    #ranking = trial.suggest_categorical('ranking', [False, True])
    mse_both = trial.suggest_categorical('mse_both', [False, True])
    use_auxiliary = trial.suggest_categorical('use_auxiliary', [False, True])
    dropout = trial.suggest_float('dropout', 0.0, 1.0)
    auxiliary_weight = 0.0
    #sample_size = 0
    if use_auxiliary:
        auxiliary_weight = trial.suggest_float('auxiliary_weight', 0.0, 1.0)
        #sample_size = trial.suggest_int('sample_size', 8, 128, log=True)
    gnn_hidden = trial.suggest_int('gnn_hidden', 8, 64, log=True)
    dense_hidden = trial.suggest_int('dense_hidden', 8, 64, log=True)
    n_convs = trial.suggest_int('n_convs', 1, 10)

    kwargs = {'readout': readout, 
              'use_global_node': False,
              'n_epochs': 20, 
              'shuffle': False, 
              'include_features': include_features,
              'n_features': n_features, 
              'ranking': False, 
              'mse_both': mse_both,
              'use_auxiliary': use_auxiliary, 
              'auxiliary_weight': auxiliary_weight, 
              'n_aux_inputs': n_aux_inputs, 
              'n_aux_outputs': n_aux_outputs,
              'dropout': dropout, 
              'gnn_hidden': gnn_hidden, 
              'dense_hidden': dense_hidden,
              'batch_size': 32, 
              'n_convs': n_convs}

    print(kwargs)    

    preds = None
    try:
        clf = surrogate.NeuralNetSurrogate(pset, **kwargs)

        clf.fit(train_set[0], train_set[1], val_set=val_set)
        preds = clf.predict(val_set[0])
    except Exception as e:
        print(e.with_traceback())
        return -1
    return scipy.stats.spearmanr(preds, val_set[1]).correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP with surrogate model')
    parser.add_argument('--data_dir', '-D', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', '-C', type=str, help='Path to save checkpoint to.')
    parser.add_argument('--train_ids', '-T', type=int, nargs='+', required=True,
                        help='Generation indices for the train set.')
    parser.add_argument('--val_ids', '-V', type=int, nargs='*', default=[], help='Generation indices for the val set.')
    parser.add_argument('--train_size', '-N', type=int, default=None, help='Train set subsample size.')

    args = parser.parse_args()

    assert not os.path.exists(args.checkpoint_path)
    # TODO model kwargs
    model_kwargs = {}

    # TODO choose class
    surrogate_cls = surrogate.NeuralNetSurrogate

    # no overlap possible
    assert len(set(args.train_ids).intersection(set(args.val_ids))) == 0

    all_files = os.listdir(args.data_dir)
    bench_name = all_files[0].split('.')[1]

    pset = bench_by_name(bench_name)['pset']

    # create the types for fitness and individuals
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    train_files = []
    val_files = []

    for file in all_files:
        gen_id = int(file.split('.')[-2])
        if gen_id in args.train_ids:
            train_files.append(file)
        elif gen_id in args.val_ids:
            val_files.append(file)

    train_set = load_dataset(train_files, args.data_dir, data_size=args.train_size)
    val_set = load_dataset(val_files, args.data_dir)

    n_features = train_set[0][0].features.values.shape[1]
    n_aux_inputs = bench_by_name(bench_name)['variables']
    n_aux_outputs = 2 if 'lunar' in bench_name else 1
    opt_obj = functools.partial(objective, train_set=train_set, val_set=val_set,
                                n_features=n_features, n_aux_inputs=n_aux_inputs, n_aux_outputs=n_aux_outputs)
    study = optuna.create_study(direction='maximize')
    study.optimize(opt_obj, n_trials=100)
    print(study.best_params)

    # TODO train model with best_params found by optuna
    clf = surrogate_cls(pset, **model_kwargs)
    clf.fit(train_set[0], train_set[1], val_set=val_set)

    checkpoint = {'kwargs': model_kwargs, 'state_dict': clf.model.state_dict()}
    torch.save(checkpoint, args.checkpoint_path)

    # TODO TEST, move to predict then
    checkpoint = torch.load(args.checkpoint_path)
    model = surrogate_cls(pset, **checkpoint['kwargs'])
    model.load_state_dict(checkpoint['state_dict'])
