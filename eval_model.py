import argparse
import torch

from data_utils import get_files_by_index, init_bench, load_dataset, get_model_class, eval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GP with surrogate model')
    parser.add_argument('--surrogate', '-S', type=str, help='Which surrogate to use (GNN, TNN)', default='GNN')
    parser.add_argument('--data_dir', '-D', type=str, help='Dataset directory')
    parser.add_argument('--checkpoint_path', '-C', type=str, help='Path of saved model checkpoint.')
    parser.add_argument('--test_ids', '-V', type=int, nargs='*', default=[], help='Gen indices for the test set.')

    args = parser.parse_args()

    all_files, bench_description, pset = init_bench(args.data_dir)

    test_files = get_files_by_index(all_files, args.val_ids)
    test_set = load_dataset(test_files, args.data_dir)

    surrogate_cls = get_model_class(args.surrogate)

    checkpoint = torch.load(args.checkpoint_path)
    clf = surrogate_cls(pset, **checkpoint['kwargs'])
    clf.load_state_dict(checkpoint['state_dict'])

    preds = clf.predict(test_set[0])
    print(eval_metrics(preds, test_set[1]))
