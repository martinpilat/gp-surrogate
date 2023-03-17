import argparse
import os
import pickle
import random

import torch
from deap import creator, base, gp

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
    bench_name = all_files[0].split('.')[2]

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

    clf = surrogate_cls(pset, **model_kwargs)
    clf.fit(train_set[0], train_set[1], val_set=val_set)

    checkpoint = {'kwargs': model_kwargs, 'state_dict': clf.model.state_dict()}
    torch.save(checkpoint, args.checkpoint_path)

    # TODO TEST, move to predict then
    checkpoint = torch.load(args.checkpoint_path)
    model = surrogate_cls(pset, **checkpoint['kwargs'])
    model.load_state_dict(checkpoint['state_dict'])
