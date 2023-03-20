import argparse
import json
import os
import pickle
import random

import scipy.stats
import torch
from deap import creator, base, gp
import optuna
import functools

from gp_surrogate import surrogate
from gp_surrogate.benchmarks import bench_by_name, ot_lunarlander
from data_utils import load_dataset, get_model_class, get_files_by_index, init_bench


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

    try:
        clf = surrogate.NeuralNetSurrogate(pset, **kwargs)

        clf.fit(train_set[0], train_set[1], val_set=val_set)
        preds = clf.predict(val_set[0])
    except Exception as e:
        print(e.with_traceback())
        return -1
    return scipy.stats.spearmanr(preds, val_set[1]).correlation


def run_optuna(train_data, val_data, bench_data):
    n_features = train_data[0][0].features.values.shape[1]
    n_aux_inputs = bench_data['variables']
    n_aux_outputs = 2 if bench_data['output_transform'] == ot_lunarlander else 1
    opt_obj = functools.partial(objective, train_set=train_data, val_set=val_data,
                                n_features=n_features, n_aux_inputs=n_aux_inputs, n_aux_outputs=n_aux_outputs)
    study = optuna.create_study(direction='maximize')
    study.optimize(opt_obj, n_trials=100)
    print(study.best_params)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP with surrogate model')
    parser.add_argument('--surrogate', '-S', type=str, help='Which surrogate to use (GNN, TNN)', default='GNN')
    parser.add_argument('--data_dir', '-D', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', '-C', type=str, help='Path to save checkpoint to.')
    parser.add_argument('--train_ids', '-T', type=int, nargs='+', required=True,
                        help='Generation indices for the train set.')
    parser.add_argument('--val_ids', '-V', type=int, nargs='*', default=[], help='Generation indices for the val set.')
    parser.add_argument('--train_size', '-N', type=int, default=None, help='Train set subsample size.')
    parser.add_argument('--kwargs_json', '-J', type=str, default=None, help='Json path to model kwargs.')
    parser.add_argument('--optuna', '-O', action='store_true', help='If True, run optuna optimization to find a model.')

    args = parser.parse_args()

    # either run one model or optima optimization
    assert args.kwargs_json is None or (not args.optuna)

    # create save dir, check if checkpoint name unique
    assert not os.path.exists(args.checkpoint_path)
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    surrogate_cls = get_model_class(args.surrogate)

    # no overlap possible
    assert len(set(args.train_ids).intersection(set(args.val_ids))) == 0

    # load benchmark and dataset info
    all_files, bench_description, pset = init_bench(args.data_dir)

    train_files = get_files_by_index(all_files, args.train_ids)
    val_files = get_files_by_index(all_files, args.val_ids)

    train_set = load_dataset(train_files, args.data_dir, data_size=args.train_size)
    val_set = load_dataset(val_files, args.data_dir)

    # run search or model training
    # TODO train model with best_params found by optuna
    if args.optuna:
        study = run_optuna(train_set, val_set, bench_description)
        model_kwargs = None  # TODO get from optuna
    else:
        with open(args.kwargs_json, 'r') as f:
            model_kwargs = json.load(f)

    clf = surrogate_cls(pset, **model_kwargs)
    clf.fit(train_set[0], train_set[1], val_set=val_set)

    # save best/trained model
    checkpoint = {'kwargs': model_kwargs, 'state_dict': clf.model.state_dict()}
    torch.save(checkpoint, args.checkpoint_path)

    # TODO TEST, move to predict then
    checkpoint = torch.load(args.checkpoint_path)
    model = surrogate_cls(pset, **checkpoint['kwargs'])
    model.load_state_dict(checkpoint['state_dict'])
