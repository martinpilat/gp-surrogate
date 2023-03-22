import argparse
import pickle

import numpy as np
import torch

from data_utils import get_files_by_index, init_bench, load_dataset, get_model_class, eval_metrics, inds_to_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP with surrogate model')
    parser.add_argument('--surrogate', '-S', type=str, help='Which surrogate to use (GNN, TNN)', default='GNN')
    parser.add_argument('--data_dir', '-D', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', '-C', type=str, help='Path of saved model checkpoint.')
    parser.add_argument('--test_ids', '-V', type=int, nargs='*', default=[], help='Gen indices for the test set.')
    parser.add_argument('--train_individuals', '-T', type=str, default=None,
                        help='Individuals that were used for training.')

    args = parser.parse_args()

    all_files, bench_description, pset = init_bench(args.data_dir)

    test_files = get_files_by_index(all_files, args.test_ids)
    test_set = load_dataset(test_files, args.data_dir)

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

    preds = clf.predict(test_set[0])
    y_true = np.array(test_set[1])
    print("Full metrics:")
    print(eval_metrics(preds, y_true))

    if mask is not None:
        print("\nNew individuals only:")
        print(eval_metrics(preds, y_true, mask=mask))

        print("\nSeen individuals only:")
        print(eval_metrics(preds, y_true, mask=~mask))
