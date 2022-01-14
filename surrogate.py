from abc import abstractmethod

import torch
from deap import gp
import itertools
import collections
import pandas as pd
import statistics
import numpy as np
from sklearn import pipeline, ensemble, impute
from torch_geometric.transforms import GCNNorm

from gnn.model import GINModel, to_dataset, train
from gnn.graph import gen_feature_vec_template, compile_tree


def extract_features(ind: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped):
    """ Extracts the features from the individual

    :param ind: the individual
    :param pset: the primitive set used by the individual
    :return: the extracted features as pandas dataframe
    """
    # get the names of primitives, terminals and arguments from the pset
    primitive_names = list(map(lambda x: x.name, itertools.chain.from_iterable(pset.primitives.values())))
    terminal_names = list(map(lambda x: x.name if isinstance(x, gp.Terminal) else x.__name__,
                              itertools.chain.from_iterable(pset.terminals.values())))
    arg_names = list(map(lambda x: x.name, filter(lambda y: isinstance(y, gp.Terminal),
                                                  itertools.chain.from_iterable(pset.terminals.values()))))

    # get the names used in the individual, use names of constants instead of their values
    ind_names = list(map(lambda x: x.__class__.__name__ if isinstance(x, gp.Ephemeral) else x.name, ind))

    # extract the values of the constants from the individual
    consts = filter(lambda x: isinstance(x, gp.Ephemeral), ind)
    const_values = list(map(lambda x: float(x.name), consts))

    # count how many times each name is in the individual
    counts = collections.Counter(ind_names)

    # compute statistics on constants
    const_avg = sum([abs(x) for x in const_values])/len(const_values) if const_values else 0
    const_max = max(const_values) if const_values else -1e6
    const_min = min(const_values) if const_values else +1e6
    const_distinct = len(set(const_values))/len(const_values) if const_values else 1
    const_stats = [len(const_values)/len(ind), const_avg, const_max, const_min, const_distinct]

    # compute statistics on arguments (number of arguments and number of distinct arguments)
    arg_stats = [len([a for a in ind_names if a in arg_names])/len(arg_names),
                 len(set(ind_names) & set(arg_names))/len(arg_names)]

    pfit_min = np.nan
    pfit_max = np.nan
    pfit_avg = np.nan

    if hasattr(ind, 'parfitness'):
        pfit = ind.parfitness
        pfit_min = min(pfit)
        pfit_max = max(pfit)
        pfit_avg = statistics.mean(pfit)

    pfit_stats = [pfit_min, pfit_max, pfit_avg]

    # create the dataframe
    frame = pd.DataFrame([len(ind), ind.height] + const_stats + arg_stats + pfit_stats +
                         [counts[x]/len(ind) for x in primitive_names] +
                         [counts[x]/len(ind) for x in terminal_names]).transpose()
    frame.columns = ['len', 'height', 'const_count', 'const_mean', 'const_max', 'const_min', 'const_distinct',
                     'arg_count', 'arg_distinct', 'pfit_min', 'pfit_max', 'pfit_avg'] + primitive_names + terminal_names
    return frame


class SurrogateBase:
    def __init__(self, pset, n_jobs=1):
        self.pset = pset
        self.n_jobs = n_jobs

    @abstractmethod
    def fit(self, inds, fitness, first_gen=False):
        pass

    @abstractmethod
    def predict(self, inds):
        pass


class FeatureSurrogate(SurrogateBase):
    def __init__(self, pset, n_jobs=1, model=None):
        super().__init__(pset, n_jobs)

        if model is None:
            model = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs, max_depth=14)

        self.pipeline = pipeline.Pipeline([
            ('impute', impute.SimpleImputer(strategy='median')),
            ('model', model)
        ])

    def fit(self, inds, fitness, first_gen=False):
        for ind in inds:
            ind.features = extract_features(ind, self.pset)

        features = [ind.features for ind in inds]
        features_df = pd.concat(features)

        if first_gen:
            features_df.fillna(0, inplace=True)

        self.pipeline.fit(features_df, fitness)
        return self

    def predict(self, inds):
        for ind in inds:
            ind.features = extract_features(ind, self.pset)

        pred_x = [ind.features for ind in inds]
        pred_x = pd.concat(pred_x)
        preds = self.pipeline.predict(pred_x)
        return preds


class NeuralNetSurrogate(SurrogateBase):
    def __init__(self, pset, n_jobs=1, model=None,
                 n_epochs=30, batch_size=32, shuffle=False, optimizer=None, loss=None, verbose=False,
                 use_root=False, use_global_node=False, gcn_transform=False, include_features=False, n_features=None,
                 **kwargs):

        super().__init__(pset, n_jobs)
        self.feature_template = gen_feature_vec_template(pset)

        if model is None:
            n_features = None if not include_features else n_features
            model = GINModel(len(self.feature_template) + 2,
                             use_root=use_root or use_global_node, n_features=n_features, **kwargs)

        self.model = model

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.optimizer = optimizer
        self.criterion = loss
        self.verbose = verbose

        self.use_root = use_root
        self.use_global_node = use_global_node
        self.include_features = include_features
        self.transform = None if not gcn_transform else GCNNorm()

    def _get_features(self, inds, first_gen=False):
        feats = np.vstack([ind.features.to_numpy() for ind in inds])
        fill_val = 0.0 if first_gen else np.nanmedian(feats, axis=0)
        feats = np.nan_to_num(feats, nan=fill_val, copy=False)

        if np.isnan(feats).any():
            feats = np.nan_to_num(feats, nan=0.0, copy=False)

        return [torch.Tensor(f[np.newaxis]) for f in feats]

    def _create_dataset(self, inds, fitness=None, first_gen=False):
        feats = self._get_features(inds, first_gen=first_gen) if self.include_features else None
        inds = [compile_tree(ind, self.feature_template,
                             use_root=self.use_root, use_global_node=self.use_global_node) for ind in inds]

        dataset = to_dataset(inds, y_accuracies=fitness, x_features=feats,
                             batch_size=self.batch_size, shuffle=self.shuffle)
        return dataset

    def fit(self, inds, fitness, first_gen=False):
        dataset = self._create_dataset(inds, fitness=fitness, first_gen=first_gen)

        train(self.model, dataset, n_epochs=self.n_epochs, optimizer=self.optimizer, criterion=self.criterion,
              verbose=self.verbose, transform=self.transform)

        return self

    def predict(self, inds):
        dataset = self._create_dataset(inds)

        res = []
        for batch in dataset:
            batch = self.transform(batch) if self.transform is not None else batch
            features = batch.features if 'features' in batch else None

            pred = self.model(batch.x, batch.edge_index, batch.batch, features=features)
            res.append(pred.detach().cpu().numpy())

        return np.hstack(res)
