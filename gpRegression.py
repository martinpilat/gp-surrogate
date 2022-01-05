from deap import tools, base, creator, gp
import pandas as pd
import operator
import math
import random
import numpy as np
import benchmarks
import algo
import argparse
import json

# decription of the benchmarks
import surrogate

benchmark_description = [
    {'name': 'keijzer-6',
     'variables': 1,
     'pset': benchmarks.get_primitive_set_for_benchmark('keijzer-6', 1)},
    {'name': 'korns-12',
     'variables': 5,
     'pset': benchmarks.get_primitive_set_for_benchmark('korns-12', 5)},
    {'name': 'pagie-1',
     'variables': 2,
     'pset': benchmarks.get_primitive_set_for_benchmark('pagie-1', 2)},
    {'name': 'nguyen-7',
     'variables': 1,
     'pset': benchmarks.get_primitive_set_for_benchmark('nguyen-7', 1)},
    {'name': 'vladislavleva-4',
     'variables': 5,
     'pset': benchmarks.get_primitive_set_for_benchmark('vladislavleva-4', 5)}
]

version_info = json.load(open('version.json', 'r'))
version = version_info['version']

# parse command-line arguments

parser = argparse.ArgumentParser(description='Run GP with surrogate model')
parser.add_argument('--problem_number', '-P', type=int, help='The number of problem to start', default=0)
parser.add_argument('--use_surrogate', '-S', help='Whether to use surrogate', action='store_true')
args = parser.parse_args()

bench_number = args.problem_number
bench_number = 2

# get the primitive set for the selected benchmark
pset = benchmark_description[bench_number]['pset']

# create the types for fitness and individuals
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# create the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# define the fitness function (log10 of the rmse or 1000 if overflow occurs)
def eval_symb_reg(individual, points, values):
        try:
            func = toolbox.compile(expr=individual)
            sqerrors = [(func(*z) - valx)**2 for z, valx in zip(points, values)]
            return math.log10(math.sqrt(math.fsum(sqerrors)) / len(points)),
        except OverflowError:
            return 1000.0,

# register the selection and genetic operators - tournament selection and, one point crossover and sub-tree mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# set height limits for the trees
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def run_baseline(i, x, y):
    """ Executes one run of the baseline algorithm

    :param i: number of the run
    :param x: the values for the training instances
    :param y: the targets for the training instances
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    # set seed to the number of the run
    random.seed(i)
    np.random.seed(i)

    # register fitness function with the right x and y
    toolbox.register("evaluate", eval_symb_reg, points=x, values=y)

    # create population
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    # create the stats object
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # run the baseline algorithm
    pop, log = algo.ea_baseline_simple(pop, toolbox, 0.2, 0.7, 200,
                                       stats=mstats, halloffame=hof, verbose=True, n_jobs=1)

    return pop, log, hof


def run_model_test(i, x, y):
    """ Executes one run of the model tests

    :param i: number of the run
    :param x: the values for the training instances
    :param y: the targets for the training instances
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    # set seed to the number of the run
    random.seed(i)
    np.random.seed(i)

    # register fitness function with the right x and y
    toolbox.register("evaluate", eval_symb_reg, points=x, values=y)

    # create population
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    # create the stats object
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # surr_cls = None
    surr_cls = surrogate.NeuralNetSurrogate
    # run the baseline algorithm
    pop, log, feat_imp = algo.ea_baseline_model(pop, toolbox, 0.2, 0.7, 110,
                                       stats=mstats, halloffame=hof, verbose=True, n_jobs=1, pset=pset,
                                       surrogate_cls=None)#surr_cls)

    return pop, log, hof, feat_imp

def run_surrogate(i, x, y):
    """ Executes one run of the surrogate algorithm

    :param i: number of the run
    :param x: the values for the training instances
    :param y: the targets for the training instances
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    # set seed to the number of the run
    random.seed(i)
    np.random.seed(i)

    # register fitness with the correct x and y
    toolbox.register("evaluate", eval_symb_reg, points=x, values=y)

    # create the initial population
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    # create the stats object
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # run the surrogate algorithm
    pop, log = algo.ea_surrogate_simple(pop, toolbox, 0.2, 0.7, 15000, pset=pset,
                                        stats=mstats, halloffame=hof, verbose=True, n_jobs=1,
                                        surrogate_cls=surrogate.NeuralNetSurrogate)

    return pop, log, hof


def run_all_baseline():
    """ Wrapper to start 25 runs of the baseline algorithm and store the results
    """
    pdlogs = pd.DataFrame()

    # get the name of the benchmark
    b_name = benchmark_description[bench_number]['name']

    # run the 15 runs
    for i in range(25):
        # read data for this run
        data = pd.read_csv('benchmarks/{bname}-train.{num}.csv'.format(bname=b_name, num=i+1), sep=';')
        y = data['y'].values
        data = data.drop('y', axis=1)
        x = data.values

        # start the baseline algorithm
        pop, log, hof = run_baseline(i, x, y)
        # append the min fitness from this run to the log
        pdlog = pd.Series(log.chapters['fitness'].select('min'), index=np.cumsum(log.select('nevals')),
                          name='run_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)

    # store the logs
    pdlogs.to_csv('output/baseline.{bname}.v{version}.csv'.format(bname=b_name, version=version))


def run_all_surrogate():
    """ Wrapper to start 25 runs of the surrogate algorithm and store the results
    """

    pdlogs = pd.DataFrame()

    # get the name of the benchmark
    b_name = benchmark_description[bench_number]['name']

    # make the 15 runs
    for i in range(25):
        # read training data for this run
        data = pd.read_csv('benchmarks/{bname}-train.{num}.csv'.format(bname=b_name, num=i+1), sep=';')
        y = data['y'].values
        data = data.drop('y', axis=1)
        x = data.values

        # start the surrogate algorithm
        pop, log, hof = run_surrogate(i, x, y)
        # concat the log from this run to the logs
        pdlog = pd.Series(log.chapters['fitness'].select('min'), index=np.cumsum(log.select('nevals')),
                          name='run_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)

    # store the logs
    pdlogs.to_csv('output/surrogate.{bname}.v{version}.csv'.format(bname=b_name, version=version))

def run_all_model_tests():
    """ Wrapper to start 25 runs of the model testing (Spearman corellation and feature imporatences)
    """
    pdlogs = pd.DataFrame()

    # get the name of the benchmark
    b_name = benchmark_description[bench_number]['name']

    # run the 15 runs
    for i in range(25):
        # read data for this run
        data = pd.read_csv('benchmarks/{bname}-train.{num}.csv'.format(bname=b_name, num=i+1), sep=';')
        y = data['y'].values
        data = data.drop('y', axis=1)
        x = data.values

        # start the baseline algorithm
        pop, log, hof, feat_imp = run_model_test(i, x, y)
        # append the min fitness from this run to the log
        pdlog = pd.Series(log.select('spear'), index=np.cumsum(log.select('nevals')),
                          name='run_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)
        # feat_imp.to_csv('output/feats_rf_def.{name}.{run}.v{version}.csv'.format(name=b_name, run=i, version=version))

    # store the logs
    pdlogs.to_csv('output/spear.rf_inf.{bname}.v{version}.csv'.format(bname=b_name, version=version))


def main():
    run_all_model_tests()
    #run_all_surrogate()
    #run the benchmark on the selected function
    # if args.use_surrogate:
    #     run_all_surrogate()
    # else:
    #     run_all_baseline()

if __name__ == "__main__":
    main()
