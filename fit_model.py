import argparse
import json
import os
import pickle

import scipy.stats
import torch
import optuna
import functools

from data_utils import load_dataset, get_model_class, get_files_by_index, init_bench, inds_to_str, eval_metrics, \
    get_rescale_val, rescale_fitness, get_n_aux_features


def suggest_params_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
    max_depth = trial.suggest_int('max_depth', 5, 20)

    kwargs = {
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }

    trial.set_user_attr('model_kwargs', kwargs)

    return kwargs


def suggest_params_gnn(trial):
    #readout = trial.suggest_categorical('readout', ['concat', 'root', 'mean'])
    # if readout != 'root':
    #     use_global_node = trial.suggest_categorical('use_global_node', [False, True])
    include_features = trial.suggest_categorical('include_features', [False, True])
    #ranking = trial.suggest_categorical('ranking', [False, True])
    #mse_both = trial.suggest_categorical('mse_both', [False, True])
    use_auxiliary = trial.suggest_categorical('use_auxiliary', [False, True])
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    auxiliary_weight = 0.0
    #sample_size = 0
    if use_auxiliary:
        auxiliary_weight = trial.suggest_float('auxiliary_weight', 0.0, 1.0)
        #sample_size = trial.suggest_int('sample_size', 8, 128, log=True)
    gnn_hidden = trial.suggest_int('gnn_hidden', 8, 64, log=True)
    dense_hidden = trial.suggest_int('dense_hidden', 8, 64, log=True)
    aux_hidden = trial.suggest_int('aux_hidden', 8, 64, log=True)
    n_convs = trial.suggest_int('n_convs', 1, 10)
    n_epochs = trial.suggest_int('n_epochs', 5, 30, step=5)

    kwargs = {'readout': 'concat',
              'use_global_node': False,
              'n_epochs': n_epochs,
              'shuffle': False, 
              'include_features': include_features,
              #'ranking': ranking,
              #'mse_both': mse_both,
              'use_auxiliary': use_auxiliary, 
              'auxiliary_weight': auxiliary_weight,
              'dropout': dropout, 
              'gnn_hidden': gnn_hidden, 
              'dense_hidden': dense_hidden,
              'aux_hidden': aux_hidden,
              'batch_size': 32, 
              'n_convs': n_convs}

    trial.set_user_attr('model_kwargs', kwargs)

    return kwargs
    
def suggest_params_tnn(trial):
    use_root = trial.suggest_categorical('use_root', [False, True])
    include_features = trial.suggest_categorical('include_features', [False, True])
    #ranking = trial.suggest_categorical('ranking', [False, True])
    #mse_both = trial.suggest_categorical('mse_both', [False, True])
    use_auxiliary = trial.suggest_categorical('use_auxiliary', [False, True])
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    auxiliary_weight = 0.0
    #sample_size = 0
    if use_auxiliary:
        auxiliary_weight = trial.suggest_float('auxiliary_weight', 0.0, 1.0)
        #sample_size = trial.suggest_int('sample_size', 8, 128, log=True)
    tnn_hidden = trial.suggest_int('tnn_hidden', 8, 64, log=True)
    dense_hidden = trial.suggest_int('dense_hidden', 8, 64, log=True)
    aux_hidden = trial.suggest_int('aux_hidden', 8, 64, log=True)
    n_epochs = trial.suggest_int('n_epochs', 5, 30, step=5)

    kwargs = {'use_root': use_root, 
              'use_global_node': False,
              'n_epochs': n_epochs,
              'shuffle': False, 
              'include_features': include_features,
              #'ranking': ranking,
              #'mse_both': mse_both,
              'use_auxiliary': use_auxiliary, 
              'auxiliary_weight': auxiliary_weight,
              'aux_hidden': aux_hidden, 
              'dropout': dropout, 
              'tnn_hidden': tnn_hidden,
              'dense_hidden': dense_hidden,
              'batch_size': 32}

    trial.set_user_attr('model_kwargs', kwargs)

    return kwargs

def objective(trial, train_set, val_set, bench_data, surrogate, metric='spearman', device=None):
    surrogate_cls = get_model_class(surrogate)
    dev_args = {'device': device}

    if surrogate == 'GNN':
        model_kwargs = suggest_params_gnn(trial)
    elif surrogate == 'TNN':
        model_kwargs = suggest_params_tnn(trial)
    elif surrogate == 'RF':
        model_kwargs = suggest_params_rf(trial)
        dev_args = {}
    else:
        raise ValueError(f"Invalid surrogate: {surrogate}.")
    
    print(model_kwargs)

    metrics = []
    for ts, vs, bench in zip(train_set, val_set, bench_data):
        try:
            nf, nai, nao = get_n_aux_features(ts, bench)
            clf = surrogate_cls(bench['pset'], n_features=nf, n_aux_inputs=nai, n_aux_outputs=nao, **dev_args,
                                **model_kwargs)

            clf.fit(ts[0], ts[1], val_set=vs)
            preds = clf.predict(vs[0])
        except Exception as e:
            print(e.with_traceback()) #TODO: this is not a valid call to with_traceback, but at least it prints the exception
            return -1
        
        metrics.append(eval_metrics(preds, vs[1])[metric])

    return sum(metrics)/len(metrics)


def run_optuna(train_data, val_data, bench_data, surrogate, study_name=None, trials=5, metric='spearman', device=None):
    opt_obj = functools.partial(objective, train_set=train_data, val_set=val_data, bench_data=bench_data,
                                surrogate=surrogate, metric=metric, device=device)

    if study_name:
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=f'sqlite:///{study_name}.db') 
    else:
        study = optuna.create_study(direction='maximize')

    study.optimize(opt_obj, n_trials=trials)
    print(study.best_params)

    return study


def rescale_and_save_val(train, val, name, save_path):
    save_path = f"{os.path.splitext(save_path)[0]}_{name}_rescale.txt"

    r_val = get_rescale_val(train[1])
    train_fitness = rescale_fitness(train[1], r_val)
    val_fitness = rescale_fitness(val[1], r_val)

    with open(save_path, 'w+') as f:
        f.write(f"{r_val}\n")

    return (train[0], train_fitness), (val[0], val_fitness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP with surrogate model')
    parser.add_argument('--surrogate', '-S', type=str, help='Which surrogate to use (GNN, TNN)', default='GNN')
    parser.add_argument('--data_dir', '-D', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', '-C', type=str, help='Path to save checkpoint to.')
    parser.add_argument('--train_ids', '-T', type=int, nargs='+', required=False,
                        help='Generation indices for the train set.')
    parser.add_argument('--val_ids', '-V', type=int, nargs='*', default=[], help='Generation indices for the val set.')
    parser.add_argument('--train_size', '-N', type=int, default=None, help='Train set subsample size.')
    parser.add_argument('--kwargs_json', '-J', type=str, default=None, help='Json path to model kwargs.')
    parser.add_argument('--optuna', '-O', action='store_true', help='If True, run optuna optimization to find a model.')
    parser.add_argument('--force', '-F', action='store_true', help='If True, overwrite existing checkpoints.')
    parser.add_argument('--unique_train', '-U', action='store_true', help='Use only unique individuals for training')
    parser.add_argument('--study_name', type=str, help='Optuna study name', default=None)
    parser.add_argument('--rescale', '-R', action='store_true', help='If True, replace invalid ind fitness with train '
                                                                     'max + 1.')
    parser.add_argument('--optuna_trials', '-K', type=int, help='Number of trials for optuna', default=20)
    parser.add_argument('--training_spec', '-f', type=str, help='File with training/val specification', default=None)
    parser.add_argument('--metric', '-M', type=str, help='Which metric to optimize', default='spearman')
    parser.add_argument('--device', type=str, help='Device for neural nets.', default=None)

    args = parser.parse_args()

    assert args.train_ids or args.training_spec

    # either run one model or optima optimization
    assert args.kwargs_json is None or (not args.optuna)

    # create save dir, check if checkpoint name unique
    if not args.force:
        assert not os.path.exists(args.checkpoint_path)
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if len(checkpoint_dir) and not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    surrogate_cls = get_model_class(args.surrogate)

    # no overlap possible

    # load benchmark and dataset info
    if args.training_spec:
        val_set = []
        train_set = []
        names = []
        bench_data = []

        with open(args.training_spec, 'r') as f:
            spec = json.load(f)

            for i, s in enumerate(spec):
                all_files, bench_description = init_bench(s['data_dir'])

                if isinstance(s['train_ids'], str):
                    lower, upper = s['train_ids'].split('-')
                    s['train_ids'] = [i for i in range(int(lower), int(upper))]
                
                assert len(set(s['train_ids']).intersection(s['val_ids'])) == 0

                train_files = get_files_by_index(all_files, s['train_ids'])
                val_files = get_files_by_index(all_files, s['val_ids'])

                ts = load_dataset(train_files, s['data_dir'], data_size=s['train_size'], unique_only=s['unique_train'])
                vs = load_dataset(val_files, s['data_dir'])
                n = str(i) if 'name' not in s else s['name']

                if s['rescale']:
                    ts, vs = rescale_and_save_val(ts, vs, n, args.checkpoint_path)
                
                train_set.append(ts)
                val_set.append(vs)
                names.append(n)
                bench_data.append(bench_description)

    else:
        all_files, bench_description = init_bench(args.data_dir)

        assert len(set(args.train_ids).intersection(set(args.val_ids))) == 0

        train_files = get_files_by_index(all_files, args.train_ids)
        val_files = get_files_by_index(all_files, args.val_ids)

        train_set = load_dataset(train_files, args.data_dir, data_size=args.train_size, unique_only=args.unique_train)
        val_set = load_dataset(val_files, args.data_dir)

        if args.rescale:
            train_set, val_set = rescale_and_save_val(train_set, val_set, '0', args.checkpoint_path)

        train_set = [train_set]
        val_set = [val_set]
        names = ['0']
        bench_data = [bench_description]

    # run search or model training
    if args.optuna:
        study = run_optuna(train_set, val_set, bench_data, args.surrogate, study_name=args.study_name,
                           trials=args.optuna_trials, metric=args.metric, device=args.device)
        model_kwargs = study.best_trial.user_attrs['model_kwargs']
    else:
        with open(args.kwargs_json, 'r') as f:
            model_kwargs = json.load(f)

    for ts, vs, n, bd in zip(train_set, val_set, names, bench_data):
        nf, nai, nao = get_n_aux_features(ts, bd)
        clf = surrogate_cls(bd['pset'], n_features=nf, n_aux_inputs=nai, n_aux_outputs=nao, **model_kwargs)
        clf.fit(ts[0], ts[1], val_set=vs)

        preds = clf.predict(vs[0])
        r = scipy.stats.spearmanr(preds, vs[1])
        print(f'Validation Spearman R: {r}')

        # save best/trained model
        checkpoint = {'kwargs': model_kwargs, 'state_dict': clf.save()}
        check_name, ext = os.path.splitext(args.checkpoint_path)
        check_path = f"{check_name}_{n}{ext}"
        torch.save(checkpoint, check_path)

        train_set_name = f"{os.path.splitext(check_path)[0]}_train_ids.pickle"
        with open(train_set_name, 'wb') as f:
            pickle.dump(set(inds_to_str(ts[0])), f)
