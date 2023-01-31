from distutils import log
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
import gym

# decription of the benchmarks
import surrogate

def ot_cartpole(x):
    return 1 if x > 0 else 0

def ot_mountaincar(x):
    return 0 if x < -1 else 2 if x > 1 else 1

def ot_acrobot(x):
    return -1 if x < -1 else 1 if x > 1 else 0

def ot_pendulum(x):
    return [x]

def ot_mountaincarcont(x):
    return [min(max(x, -1), 1)]

def ot_lunarlander(x):
    return list(x)

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
     'pset': benchmarks.get_primitive_set_for_benchmark('vladislavleva-4', 5)},
    {'name': 'rl_cartpole',
     'env_name': 'CartPole-v1',
     'variables': 4,
     'output_transform': ot_cartpole,
     'pset': benchmarks.get_primitive_set_for_benchmark('pagie-1', 4)},
    {'name': 'rl_mountaincar',
     'env_name': 'MountainCar-v0',
     'variables': 2,
     'output_transform': ot_mountaincar,
     'pset': benchmarks.get_primitive_set_for_benchmark('pagie-1', 2)},
    {'name': 'rl_acrobot',
     'env_name': 'Acrobot-v1',
     'env_kwargs': {},
     'variables': 6,
     'output_transform': ot_acrobot,
     'pset': benchmarks.get_primitive_set_for_benchmark('pagie-1', 6)},
    {'name': 'rl_pendulum',
     'env_name': 'Pendulum-v1',
     'env_kwargs': {},
     'variables': 3,
     'output_transform': ot_pendulum,
     'pset': benchmarks.get_primitive_set_for_benchmark('pagie-1', 3)},
    {'name': 'rl_mountaincarcontinuous',
     'env_name': 'MountainCarContinuous-v0',
     'env_kwargs': {},
     'variables': 2,
     'output_transform': ot_mountaincarcont,
     'pset': benchmarks.get_primitive_set_for_benchmark('pagie-1', 2)},
    {'name': 'rl_lunarlander',
     'env_name': 'LunarLanderContinuous-v2',
     'env_kwargs': {},
     'variables': 8,
     'output_transform': ot_lunarlander,
     'pset': benchmarks.get_primitive_set_for_benchmark('lander', 8)} 
]

version_info = json.load(open('version.json', 'r'))
version = version_info['version']

# parse command-line arguments

parser = argparse.ArgumentParser(description='Run GP with surrogate model')
parser.add_argument('--problem_number', '-P', type=int, help='The number of problem to start', default=0)
parser.add_argument('--use_surrogate', '-S', type=str, help='Which surrogate to use (RF, GNN, TNN)', default=None)
parser.add_argument('--use_ranking', '-R', help='Whether to use ranking loss', action='store_true')
parser.add_argument('--mse_both', '-B', help='Whether to use MSE from both batches with ranking', action='store_true')
parser.add_argument('--n_cpus', '-C', type=int, default=1)
parser.add_argument('--repeats', '-K', type=int, help='How many times to run the algorithm', default=25)
parser.add_argument('--max_evals', '-E', type=int, help='Maximum number of fitness evaluations', default=10000)
parser.add_argument('--use_local_search', '-L', help='Use local search algorithm', action='store_true')
parser.add_argument('--use_auxiliary', '-A', help='Use auxiliary task during training', action='store_true')
parser.add_argument('--auxiliary_weight', '-W', type=float, help='The weight for auxiliary task', default=0.1)
args = parser.parse_args()

print(args)

if args.use_surrogate:
    if args.use_surrogate not in ['RF', 'TNN', 'GNN']:
        print('Surrogate type must be one of RF, TNN, or GNN')
        parser.print_help()

bench_number = args.problem_number
#bench_number = 0

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

# get number of features
sample_ind = toolbox.individual()
n_features = surrogate.extract_features(sample_ind, pset)
n_features = n_features.shape[1]

surrogate_name = args.use_surrogate
if surrogate_name == 'GNN':
    surrogate_cls = surrogate.NeuralNetSurrogate
    surrogate_kwargs = {'use_root': False, 'use_global_node': True, 'gcn_transform': False,
                        'n_epochs': 20, 'shuffle': False, 'include_features': False, 'n_features': n_features,
                        'ranking': args.use_ranking, 'mse_both': args.mse_both,
                        'use_auxiliary': args.use_auxiliary, 'auxiliary_weight': args.auxiliary_weight, 
                        'n_aux_inputs': benchmark_description[bench_number]['variables'],
                        'n_aux_outputs': 1}
if surrogate_name == 'TNN':
    surrogate_cls = surrogate.TreeLSTMSurrogate
    surrogate_kwargs = {'use_root': True, 'use_global_node': False, 'n_epochs': 20, 'shuffle': False,
                        'include_features': False, 'n_features': n_features,
                        'ranking': args.use_ranking, 'mse_both': args.mse_both, 
                        'use_auxiliary': args.use_auxiliary, 'auxiliary_weight': args.auxiliary_weight,
                        'n_aux_inputs': benchmark_description[bench_number]['variables'],
                        'n_aux_outputs': 1}

if surrogate_name == 'RF':
    surrogate_cls = surrogate.FeatureSurrogate
    surrogate_kwargs = {}

if args.use_ranking and surrogate_name in ['GNN', 'TNN']:
    surrogate_name += '-R1'

    if args.mse_both:
        surrogate_name += '-B'

if args.use_local_search:
    surrogate_name += '-LS'

# define the fitness function (log10 of the rmse or 1000 if overflow occurs)
def eval_symb_reg(individual, points, values):
        try:
            func = toolbox.compile(expr=individual)
            outputs = [func(*z) for z in points]
            io_feat = [(i, o) for i, o in zip(points, outputs)]
            sqerrors = [(o - valx)**2 for o, valx in zip(outputs, values)]
            if args.use_auxiliary:
                individual.io = np.array([io[0] for io in io_feat]), np.array([io[1] for io in io_feat])
            return math.log10(math.sqrt(math.fsum(sqerrors)) / len(points)),
        except OverflowError:
            return 1000.0,

def eval_rl(individual, environment, output_transform):
    try:
        func = toolbox.compile(expr=individual)
        R = 0
        io_feat = []
        for s in range(5):
            environment.seed(s)
            obs = environment.reset()
            done = False
            while not done:
                obs = list(obs)
                o = func(*obs)
                action = output_transform(o)
                io_feat.append((obs, o))
                obs, r, done, _ = environment.step(action)
                R += float(r)
        if args.use_auxiliary:
            individual.io = io_feat
        return -R/5,
    except OverflowError:
        return 100000.0,

# register the selection and genetic operators - tournament selection and, one point crossover and sub-tree mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# set height limits for the trees
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def run_baseline(i, bench):
    """ Executes one run of the baseline algorithm

    :param i: number of the run
    :bench: benchmark description
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    # set seed to the number of the run
    random.seed(i)
    np.random.seed(i)

    # initialize fitness for benchmark
    if bench['name'].startswith('rl_'):
        env = gym.make(bench['env_name'], **bench['env_kwargs'])
        toolbox.register("evaluate", eval_rl, environment=env, output_transform=bench['output_transform'])
    else:
        data = pd.read_csv('benchmarks/{bname}-train.{num}.csv'.format(bname=bench['name'], num=i+1), sep=';')
        y = data['y'].values
        data = data.drop('y', axis=1)
        x = data.values
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
    pop, log = algo.ea_baseline_simple(pop, toolbox, 0.2, 0.7, args.max_evals,
                                       stats=mstats, halloffame=hof, verbose=True, n_jobs=1)

    return pop, log, hof


def run_model_test(i, bench):
    """ Executes one run of the model tests

    :param i: number of the run
    :bench: the benchmark description
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    # set seed to the number of the run
    random.seed(i)
    np.random.seed(i)

    # initialize fitness for benchmark
    if bench['name'].startswith('rl_'):
        env = gym.make(bench['env_name'], **bench['env_kwargs'])
        toolbox.register("evaluate", eval_rl, environment=env, output_transform=bench['output_transform'])
    else:
        data = pd.read_csv('benchmarks/{bname}-train.{num}.csv'.format(bname=bench['name'], num=i+1), sep=';')
        y = data['y'].values
        data = data.drop('y', axis=1)
        x = data.values
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
    pop, log, feat_imp = algo.ea_baseline_model(pop, toolbox, 0.2, 0.7, 110,
                                       stats=mstats, halloffame=hof, verbose=True, n_jobs=1, pset=pset,
                                       surrogate_cls=surrogate_cls, surrogate_kwargs=surrogate_kwargs)

    return pop, log, hof, feat_imp

def run_surrogate(i, bench):
    """ Executes one run of the surrogate algorithm

    :param i: number of the run
    :param bench: the benchmark to use
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    # set seed to the number of the run
    random.seed(i)
    np.random.seed(i)
    scale = False
    train_fit_lim = 1000
    # initialize fitness for benchmark
    if bench['name'].startswith('rl_'):
        scale = True
        train_fit_lim = 100000
        env = gym.make(bench['env_name'], **bench['env_kwargs'])
        toolbox.register("evaluate", eval_rl, environment=env, output_transform=bench['output_transform'])
    else:
        data = pd.read_csv('benchmarks/{bname}-train.{num}.csv'.format(bname=bench['name'], num=i+1), sep=';')
        y = data['y'].values
        data = data.drop('y', axis=1)
        x = data.values
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
    if args.use_local_search :
        pop, log = algo.ea_surrogate_localsearch(pop, toolbox, 0.2, 0.7, args.max_evals, pset=pset,
                                        stats=mstats, halloffame=hof, verbose=True, n_jobs=1, scale=scale,
                                        train_fit_lim=train_fit_lim,
                                        surrogate_cls=surrogate_cls, surrogate_kwargs=surrogate_kwargs)
    else: 
        pop, log = algo.ea_surrogate_simple(pop, toolbox, 0.2, 0.7, args.max_evals, pset=pset,
                                            stats=mstats, halloffame=hof, verbose=True, n_jobs=1, scale=scale,
                                            train_fit_lim=train_fit_lim,
                                            surrogate_cls=surrogate_cls, surrogate_kwargs=surrogate_kwargs)

    return pop, log, hof


def run_all(fn, log_prefix, repeats=25):
    """ Wrapper to start 25 runs of the algorithm and store the results
    """
    import multiprocessing
    import functools

    pool = multiprocessing.Pool(args.n_cpus)
    pdlogs = pd.DataFrame()

    # get the name of the benchmark
    b_name = benchmark_description[bench_number]['name']

    runner = functools.partial(fn, bench=benchmark_description[bench_number])

    # run the 25 runs
    logs = map(runner, range(repeats))
    for i, l in enumerate(logs):
        _, log, _ = l
        pdlog = pd.Series(log.chapters['fitness'].select('min'), index=np.cumsum(log.select('nevals')),
                          name='run_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)

    # store the logs
    pdlogs.to_csv(f'output/{log_prefix}.{b_name}.v{version}.csv')


def run_all_model_tests():
    """ Wrapper to start 25 runs of the model testing (Spearman corellation and feature imporatences)
    """
    pdlogs = pd.DataFrame()

    # get the name of the benchmark
    b_name = benchmark_description[bench_number]['name']

    # run the 15 runs
    for i in range(25):
        
        # start the baseline algorithm
        pop, log, hof, feat_imp = run_model_test(i, benchmark_description[bench_number])
        # append the min fitness from this run to the log
        pdlog = pd.Series(log.select('spear'), index=np.cumsum(log.select('nevals')),
                          name='run_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)
        # feat_imp.to_csv('output/feats_rf_def.{name}.{run}.v{version}.csv'.format(name=b_name, run=i, version=version))

    # store the logs
    pdlogs.to_csv('output/spear.rf_inf.{bname}.v{version}.csv'.format(bname=b_name, version=version))


def main():
    
    #run_all_model_tests()
    #run_all_baseline()
    #run_all_surrogate()
    #run the benchmark on the selected function
    if args.use_surrogate:
        prefix = 'surrogate.' if not args.use_local_search else 'surrogate-ls.'
        run_all(fn=run_surrogate, log_prefix=prefix + surrogate_name, repeats=args.repeats)
    else:
        run_all(fn=run_baseline, log_prefix='baseline', repeats=args.repeats)

if __name__ == "__main__":
    main()
