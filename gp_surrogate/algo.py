import sys
import sklearn.metrics
from deap import tools
import random
from gp_surrogate import surrogate
import pandas as pd
import math
import numpy as np
import joblib


def add_features(ind, pset):
    """ Extracts the features from the individual and adds them a new field called 'features' created in the individual

    :param ind: the individual to extract features from
    :param pset: the primitive set used by the individual
    :return: the individual with added features
    """
    ind.features = surrogate.extract_features(ind, pset)
    return ind

def ea_surrogate_localsearch(population, toolbox, cxpb, mutpb, max_evals, pset,
                        stats=None, halloffame=None, verbose=__debug__, n_jobs=-1,
                        scale=False, train_fit_lim=1000, surrogate_cls=None, surrogate_kwargs=None):
    """ Performs the surrogate version of the ea

    :param population: the initial population
    :param toolbox: the toolbox to use
    :param cxpb: probability of crossover
    :param mutpb: probability of muatation
    :param max_evals: maximum number of fitness evaluations
    :param pset: the primitive set
    :param stats: the stats object to compute and save stats
    :param halloffame: the hall of fame
    :param verbose: verobosity level (whether to print the log or not)
    :param n_jobs: the number of jobs use to train the surrogate model and to compute the fitness
    :return: the final population and the log of the run
    """
    if surrogate_cls is None:
        surrogate_cls = surrogate.FeatureSurrogate
    if surrogate_kwargs is None:
        surrogate_kwargs = {}

    print(surrogate_cls)

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'tot_evals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            ind.estimate = False
            add_features(ind, pset)

        # add the evaluated individuals into archive
        archive = invalid_ind
        n_evals = len(invalid_ind)
        # update the hall of fame
        if halloffame is not None:
            halloffame.update(population)

        # record the stats
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        gen = 1
        # Begin the generational process
        while n_evals < max_evals:

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            in_evals = 0

            if len(archive) > 1000:
                
                train = archive
                if len(train) > 5000:
                    train = random.sample(archive, 5000)

                features = [ind for ind in train if ind.fitness.values[0] < train_fit_lim]
                targets = [ind.fitness.values[0] for ind in train if ind.fitness.values[0] < train_fit_lim]

                if scale:
                    a = min(targets)
                    b = max(targets)
                    targets = [math.log(1+(t-a)) for t in targets]

                # build the surrogate model (random forest regressor)
                clf = surrogate_cls(pset, n_jobs=n_jobs, **surrogate_kwargs)
                clf.fit(features, targets, first_gen=gen == 1)

                in_par_ix = list(range(len(offspring)))
                random.shuffle(in_par_ix)
                in_par_ix = in_par_ix[:50]
                in_par = [toolbox.clone(offspring[i]) for i in in_par_ix]

                preds = clf.predict(in_par)
                for o, f in zip(in_par, preds):
                    o.fitness.values = (f,)

                for _ in range(50):
                    
                    in_off = toolbox.select(in_par, len(in_par))
                    in_off = varAnd(in_off, toolbox, cxpb, mutpb)
                    preds = clf.predict(in_off)
                    
                    for o, f in zip(in_off, preds):
                       o.fitness.values = (f,)
                       o.estimate = True

                    in_par[:] = tools.selBest(in_off, len(in_par) - 1) + tools.selBest(in_par, 1)

                    # if verbose:
                    #     best = tools.selBest(in_off, 1)
                    #     print(best[0].fitness.values)

                fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in in_par)
                for ind, fit in zip(in_par, fitnesses):
                    ind.fitness.values = fit
                    ind.estimate = False
                    add_features(ind, pset)

                archive = archive+in_par

                in_evals += len(in_par)

                for i, o in zip(in_par_ix, in_par):
                    offspring[i] = o
                
            # prepare the rest of the individuals for evaluation with the real fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # evaluate invalid individuals with the real fitness
            fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.estimate = False
                add_features(ind, pset)

            assert all(not ind.estimate for ind in offspring)

            # add the evaluated individual into the archive
            archive = archive+invalid_ind

            # Update the hall of fame with the generated and evaluated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring

            population[:] = tools.selBest(offspring, len(offspring) - 1) + tools.selBest(population, 1)

            # Append the current generation statistics to the logbook
            evaluated = [ind for ind in population if not ind.estimate]
            record = stats.compile(evaluated) if stats else {}
            if len(invalid_ind) > 0:
                n_evals += len(invalid_ind) + in_evals
                logbook.record(gen=gen, nevals=len(invalid_ind) + in_evals, tot_evals=n_evals, **record)
            if verbose:
                print(logbook.stream)
                if gen%5 == 0:
                    sys.stdout.flush()

            gen += 1

    if verbose:
        sys.stdout.flush()
    
    return population, logbook



def ea_surrogate_simple(population, toolbox, cxpb, mutpb, max_evals, pset,
                        stats=None, halloffame=None, verbose=__debug__, n_jobs=-1,
                        scale=False, train_fit_lim=1000, surrogate_cls=None, surrogate_kwargs=None):
    """ Performs the surrogate version of the ea

    :param population: the initial population
    :param toolbox: the toolbox to use
    :param cxpb: probability of crossover
    :param mutpb: probability of muatation
    :param max_evals: maximum number of fitness evaluations
    :param pset: the primitive set
    :param stats: the stats object to compute and save stats
    :param halloffame: the hall of fame
    :param verbose: verobosity level (whether to print the log or not)
    :param n_jobs: the number of jobs use to train the surrogate model and to compute the fitness
    :return: the final population and the log of the run
    """
    if surrogate_cls is None:
        surrogate_cls = surrogate.FeatureSurrogate
    if surrogate_kwargs is None:
        surrogate_kwargs = {}

    print(surrogate_cls)

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'tot_evals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            ind.estimate = False
            add_features(ind, pset)

        # add the evaluated individuals into archive
        archive = invalid_ind
        n_evals = len(invalid_ind)
        # update the hall of fame
        if halloffame is not None:
            halloffame.update(population)

        # record the stats
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        gen = 1
        # Begin the generational process
        while n_evals < max_evals:

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            selected = toolbox.clone(offspring)

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            if len(archive) > 1000:

                train = archive
                if len(train) > 5000:
                    train = random.sample(archive, 5000)

                features = [ind for ind in train if ind.fitness.values[0] < train_fit_lim]
                targets = [ind.fitness.values[0] for ind in train if ind.fitness.values[0] < train_fit_lim]

                if scale:
                    a = min(targets)
                    b = max(targets)
                    targets = [math.log(1+(t-a)) for t in targets]

                # build the surrogate model (random forest regressor)
                clf = surrogate_cls(pset, n_jobs=n_jobs, **surrogate_kwargs)
                clf.fit(features, targets, first_gen=gen == 1)

                # Evaluate the individuals with an invalid fitness using the surrogate model
                invalid_ind = [add_features(ind, pset) for ind in offspring if not ind.fitness.valid]
                invalid_ix = [ix for ix in range(len(offspring)) if not offspring[ix].fitness.valid]
                pred_x = [ind for ind in invalid_ind]
                preds = clf.predict(pred_x)

                # real_preds = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
                # import scipy.stats
                # print(scipy.stats.spearmanr(preds, real_preds).correlation)

                sorted_ix = np.argsort(preds)
                bad_ix = sorted_ix[-int(2*len(invalid_ind)/3):]

                # replace the bad individuals by their parents, effectively removing them from the population
                for ind, ix in zip(invalid_ind, range(len(invalid_ix))):
                    if ix in bad_ix:
                        offspring[invalid_ix[ix]] = selected[invalid_ix[ix]]

            # prepare the rest of the individuals for evaluation with the real fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # evaluate invalid individuals with the real fitness
            fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.estimate = False
                add_features(ind, pset)

            # add the evaluated individual into the archive
            archive = archive+invalid_ind

            # Update the hall of fame with the generated and evaluated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring

            population[:] = tools.selBest(offspring, len(offspring) - 1) + tools.selBest(population, 1)

            # Append the current generation statistics to the logbook
            evaluated = [ind for ind in population if not ind.estimate]
            record = stats.compile(evaluated) if stats else {}
            if len(invalid_ind) > 0:
                n_evals += len(invalid_ind)
                logbook.record(gen=gen, nevals=len(invalid_ind), tot_evals=n_evals, **record)
            if verbose:
                print(logbook.stream)
                if gen % 5 == 0:
                    sys.stdout.flush()

            gen += 1

    if verbose:
        sys.stdout.flush()

    return population, logbook


def ea_baseline_simple(population, toolbox, cxpb, mutpb, max_evals, stats=None,
                       halloffame=None, verbose=__debug__, n_jobs=1):
    """ Performs the baseline version of the ea

    :param population: the initial population
    :param toolbox: the toolbox to use
    :param cxpb: probability of crossover
    :param mutpb: probability of muatation
    :param ngen: number of generations
    :param stats: the stats object to compute and save stats
    :param halloffame: the hall of fame
    :param verbose: verbosity level (whether to print the log or not)
    :param n_jobs: the number of jobs use to train the surrogate model and to compute the fitness
    :return: the final population and the log of the run
    """

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        n_evals = len(invalid_ind)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        gen = 1
        while n_evals < max_evals:
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            n_evals += len(invalid_ind)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = tools.selBest(offspring, len(population) - 1) + tools.selBest(population, 1)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            
            gen += 1

        return population, logbook


def ea_baseline_model(population, toolbox, cxpb, mutpb, ngen, pset, stats=None,
                       halloffame=None, verbose=__debug__, n_jobs=1, surrogate_cls=None, surrogate_kwargs=None):
    """ Performs the tests of the model

    :param population: the initial population
    :param toolbox: the toolbox to use
    :param cxpb: probability of crossover
    :param mutpb: probability of muatation
    :param ngen: number of generations
    :param stats: the stats object to compute and save stats
    :param halloffame: the hall of fame
    :param verbose: verbosity level (whether to print the log or not)
    :param n_jobs: the number of jobs use to train the surrogate model and to compute the fitness
    :return: the final population and the log of the run
    """
    if surrogate_cls is None:
        surrogate_cls = surrogate.FeatureSurrogate
    if surrogate_kwargs is None:
        surrogate_kwargs = {}

    print(surrogate_cls)

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'spear'] + (stats.fields if stats else [])

        evals = 0

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            add_features(ind, pset)

        archive = invalid_ind
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        evals += len(invalid_ind)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        feat_imp = pd.DataFrame(columns=archive[0].features.columns)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            train = archive
            if len(train) > 5000:
                train = random.sample(archive, 5000)

            features = [ind for ind in train if ind.fitness.values[0] < 1000]
            targets = [ind.fitness.values[0] for ind in train if ind.fitness.values[0] < 1000]

            # build the surrogate model (random forest regressor)

            clf = surrogate_cls(pset, n_jobs=n_jobs, **surrogate_kwargs)
            # clf = pipeline.Pipeline([('impute', preprocessing.Imputer(strategy='median')), ('scale', preprocessing.StandardScaler()), ('svm', svm.SVR())])

            # clf = pipeline.Pipeline([('impute', preprocessing.Imputer(strategy='median')),
            #                          ('model', ensemble.RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=n_jobs))])
            clf.fit(features, targets, first_gen=gen == 1)

            preds = clf.predict(features)
            score = sklearn.metrics.mean_squared_error(preds, targets)
            print(score)

            # columns = archive[0].features.columns
            # importances = clf.named_steps['model'].feature_importances_
            # cur_feat = pd.DataFrame(columns=columns, data=[importances])
            # cur_feat.index = [evals]
            # feat_imp = feat_imp.append(cur_feat)

            # Evaluate the individuals with an invalid fitness using the surrogate model
            invalid_ind = [add_features(ind, pset) for ind in offspring if not ind.fitness.valid]
            pred_x = [ind for ind in invalid_ind]
            preds = clf.predict(pred_x)

            real_preds = parallel(joblib.delayed(toolbox.evaluate)(ind) for ind in invalid_ind)
            import scipy.stats
            spear = scipy.stats.spearmanr(preds, real_preds).correlation

            for ind, fit in zip(invalid_ind, real_preds):
                ind.fitness.values = fit
                add_features(ind, pset)

            archive = archive+invalid_ind
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = tools.selBest(offspring, len(population) - 1) + tools.selBest(population, 1)

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            evals += len(invalid_ind)
            logbook.record(gen=gen, nevals=len(invalid_ind), spear=spear, **record)
            if verbose:
                print(logbook.stream)

        return population, logbook, feat_imp


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    for off in offspring:
        if hasattr(off, 'parfitness'):
            del off.parfitness

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            offspring[i - 1].parfitness = [population[i - 1].fitness.values[0], population[i].fitness.values[0]]
            offspring[i].parfitness = [population[i - 1].fitness.values[0], population[i].fitness.values[0]]

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            if not hasattr(offspring, 'parfitness'):
                offspring[i].parfitness = [population[i].fitness.values[0]]

    return offspring
