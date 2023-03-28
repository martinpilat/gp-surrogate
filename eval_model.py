import argparse
import os.path
import pickle
import warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import torch

from data_utils import get_files_by_index, init_bench, load_dataset, get_model_class, eval_metrics, inds_to_str

sns.set()


def plot_predictions(preds, y_true, save_dir=None, prefix=''):
    sns.histplot(preds - y_true)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, f'{prefix}diff_hist.png'))

    plt.scatter(y_true, preds)

    if save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, f'{prefix}pred_true.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP with surrogate model')
    parser.add_argument('--surrogate', '-S', type=str, help='Which surrogate to use (GNN, TNN)', default='GNN')
    parser.add_argument('--data_dir', '-D', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', '-C', type=str, help='Path of saved model checkpoint.')
    parser.add_argument('--test_ids', '-V', type=int, nargs='*', default=[], help='Gen indices for the test set.')
    parser.add_argument('--train_individuals', '-T', type=str, default=None,
                        help='Individuals that were used for training.')
    parser.add_argument('--rescale', '-R', type=float, help='New value for invalid individuals.', default=None)
    parser.add_argument('--out_dir', '-O', type=str, help='Directory path where to write outputs to.', default=None)

    args = parser.parse_args()

    if args.out_dir is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        else:
            warnings.warn(f"Output directory already exists: {args.out_dir}")

    all_files, bench_description, pset = init_bench(args.data_dir)

    test_files = get_files_by_index(all_files, args.test_ids)
    test_set = load_dataset(test_files, args.data_dir, rescale_val=args.rescale)

    # mask - shared individuals between train and test set
    mask = None
    if args.train_individuals is not None:
        with open(args.train_individuals, 'rb') as f:
            train_inds = pickle.load(f)

        common = train_inds.intersection(set(inds_to_str(test_set[0])))
        print(f"Common networks between train generations and test generations: {len(common)}")
        mask = np.array([(str(ind) not in common) for ind in test_set[0]])

    surrogate_cls = get_model_class(args.surrogate)

    checkpoint = torch.load(args.checkpoint_path)
    clf = surrogate_cls(pset, **checkpoint['kwargs'])
    clf.load_state_dict(checkpoint['state_dict'])

    res = []

    def print_and_update(metrics, name):
        metrics['name'] = name
        res.append(metrics)
        print(metrics)


    preds = clf.predict(test_set[0])
    y_true = np.array(test_set[1])
    metrics = eval_metrics(preds, y_true, val=args.rescale)
    print("Full metrics:")
    print_and_update(metrics, 'all')

    if mask is not None:
        print("\nNew individuals only:")
        metrics = eval_metrics(preds, y_true, mask=mask, val=args.rescale)
        print_and_update(metrics, 'new')

        print("\nSeen individuals only:")
        metrics = eval_metrics(preds, y_true, mask=~mask, val=args.rescale)
        print_and_update(metrics, 'seen')

    if args.out_dir is not None:
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(args.out_dir, 'metrics.csv'))

    plot_predictions(preds, y_true, save_dir=args.out_dir)

    # plot only valid individuals
    lim = 1000 if args.rescale is None else args.rescale
    mask = y_true < lim
    preds = preds[mask]
    y_true = y_true[mask]

    plot_predictions(preds, y_true, save_dir=args.out_dir, prefix='valid_')
