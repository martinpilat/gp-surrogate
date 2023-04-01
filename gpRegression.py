import os.path

from deap import tools, base, creator, gp
import pandas as pd
import operator
import math
import random
import numpy as np
import argparse
import json
import gym

# decription of the benchmarks
from gp_surrogate import surrogate, algo, benchmarks

benchmark_description = benchmarks.benchmark_description

version_info = json.load(open('version.json', 'r'))
version = version_info['version']

# parse command-line arguments

parser = argparse.ArgumentParser(description='Run GP with surrogate model')
parser.add_argument('--problem_number', '-P', type=int, help='The number of problem to start', default=0)
parser.add_argument('--use_surrogate', '-S', type=str, help='Which surrogate to use (RF, GNN, TNN)', default=None)
parser.add_argument('--filename', type=str, help='Filename prefix', default=None)
parser.add_argument('--gnn_readout', '-O', type=str, help='Read to use in GNN (concat, root, mean).', default='concat')
parser.add_argument('--tree_readout', '-T', type=str, help='Read to use in TreeLSTM (root, mean).', default='root')
parser.add_argument('--use_ranking', '-R', help='Whether to use ranking loss', action='store_true')
parser.add_argument('--mse_both', '-B', help='Whether to use MSE from both batches with ranking', action='store_true')
parser.add_argument('--n_cpus', '-C', type=int, default=1)
parser.add_argument('--device', '-D', type=str, help='Model device (cpu, cuda).', default=None)
parser.add_argument('--repeats', '-K', type=int, help='How many times to run the algorithm', default=25)
parser.add_argument('--max_evals', '-E', type=int, help='Maximum number of fitness evaluations', default=10000)
parser.add_argument('--n_train_epochs', '-N', type=int, help='Number of model train epochs', default=20)
parser.add_argument('--use_local_search', '-L', help='Use local search algorithm', action='store_true')
parser.add_argument('--use_auxiliary', '-A', help='Use auxiliary task during training', action='store_true')
parser.add_argument('--use_features', '-F', help='Use features as input during NN training', action='store_true')
parser.add_argument('--use_global_node', '-G', help='Use features as input during NN training', action='store_true')
parser.add_argument('--auxiliary_weight', '-W', type=float, help='The weight for auxiliary task', default=0.1)
parser.add_argument('--dropout', '-U', type=float, help='Dropout p.', default=0.1)
parser.add_argument('--gnn_hidden', '-H', type=int, help='GIN/TreeLSTM hidden size', default=32)
parser.add_argument('--dense_hidden', '-J', type=int, help='Linear hidden size', default=32)
parser.add_argument('--aux_hidden', type=int, help='Hidden layer size for the aux network', default=32)
parser.add_argument('--retrain_every', type=int, help='How often is the surrogate retrained (generations)', default=1)
parser.add_argument('--max_train_size', type=int, help='The maximum size of the training set sampled from the archive', default=5000)
parser.add_argument('--batch_size', type=int, help='Batch size for GNN / TreeLSTM', default=32)
parser.add_argument('--n_convs', type=int, help='Number of GNN conv layers', default=3)
parser.add_argument('--save_training_data', help='Save data for training of surrogates', action='store_true')
parser.add_argument('--save_dir', type=str, help='Save data directory', default=None)
args = parser.parse_args()

print(args)

if args.use_surrogate:
    if args.use_surrogate not in ['RF', 'TNN', 'GNN', 'RAND', 'IDEAL']:
        print('Surrogate type must be one of RF, TNN, GNN, RAND, or IDEAL')
        parser.print_help()

bench_number = args.problem_number
bench_name = benchmark_description[bench_number]['name']
safe_bname = bench_name.replace('-', '_')
#bench_number = 0

# get the primitive set for the selected benchmark
pset = benchmark_description[bench_number]['pset']

# create the types for fitness and individuals
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create(f'Individual_{safe_bname}', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# create the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, eval(f'creator.Individual_{safe_bname}'), toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# get number of features
sample_ind = toolbox.individual()
n_features = surrogate.extract_features(sample_ind, pset)
n_features = n_features.shape[1]


surrogate_name = args.use_surrogate
if surrogate_name == 'GNN':
    surrogate_cls = surrogate.NeuralNetSurrogate
    surrogate_kwargs = {'readout': args.gnn_readout, 'use_global_node': args.use_global_node,
                        'n_epochs': args.n_train_epochs, 'shuffle': False, 'include_features': args.use_features,
                        'n_features': n_features, 'ranking': args.use_ranking, 'mse_both': args.mse_both,
                        'use_auxiliary': args.use_auxiliary, 'auxiliary_weight': args.auxiliary_weight, 
                        'n_aux_inputs': benchmark_description[bench_number]['variables'], 'device': args.device,
                        'n_aux_outputs': 2 if 'lunar' in benchmark_description[bench_number]['name'] else 1,
                        'dropout': args.dropout, 'gnn_hidden': args.gnn_hidden, 'dense_hidden': args.dense_hidden,
                        'batch_size': args.batch_size, 'n_convs': args.n_convs}
if surrogate_name == 'TNN':
    if args.tree_readout == 'root':
        use_root = True
    elif args.tree_readout == 'mean':
        use_root = False
    else:
        raise ValueError(f"Invalid TreeLSTM readout: {args.tree_readout} (allowed: root, mean).")

    surrogate_cls = surrogate.TreeLSTMSurrogate
    surrogate_kwargs = {'use_root': use_root, 'use_global_node': args.use_global_node, 'n_epochs': args.n_train_epochs,
                        'shuffle': False, 'include_features': args.use_features, 'n_features': n_features,
                        'ranking': args.use_ranking, 'mse_both': args.mse_both, 'device': args.device,
                        'use_auxiliary': args.use_auxiliary, 'auxiliary_weight': args.auxiliary_weight,
                        'n_aux_inputs': benchmark_description[bench_number]['variables'],
                        'n_aux_outputs': 2 if 'lunar' in benchmark_description[bench_number]['name'] else 1,
                        'aux_hidden': args.aux_hidden, 'dropout': args.dropout, 'tnn_hidden':args.gnn_hidden, 'dense_hidden':args.dense_hidden,
                        'batch_size': args.batch_size}
if surrogate_name == 'RF':
    surrogate_cls = surrogate.FeatureSurrogate
    surrogate_kwargs = {}

if surrogate_name == 'RAND':
    surrogate_cls = surrogate.RandomSurrogate
    surrogate_kwargs = {}

if args.use_ranking and surrogate_name in ['GNN', 'TNN']:
    surrogate_name += '-R1'

    if args.mse_both:
        surrogate_name += '-B'

if surrogate_name and args.use_local_search:
    surrogate_name += '-LS'

if surrogate_name and args.use_auxiliary:
    surrogate_name += '-AUX'

if surrogate_name == 'IDEAL':
    surrogate_cls = None # surrogate will be set later, once the fitness is defined
    surrogate_kwargs = {}

# define the fitness function (log10 of the rmse or 1000 if overflow occurs)
def eval_symb_reg(individual, points, values):
        try:
            func = toolbox.compile(expr=individual)
            outputs = []
            io_feat = []
            for z in points:
                out = func(*z)
                outputs.append(out)
                io_feat.append((z, out))
            sqerrors = [(o - valx)**2 for o, valx in zip(outputs, values)]
            if args.use_auxiliary:
                individual.io = np.array([io[0] for io in io_feat]), np.array([io[1] for io in io_feat])
            return math.log10(math.sqrt(math.fsum(sqerrors)) / len(points)),
        except OverflowError:
            if args.use_auxiliary and io_feat:
                individual.io = np.array([io[0] for io in io_feat]), np.array([io[1] for io in io_feat])
            elif args.use_auxiliary:
                individual.io = np.array([points[0]]), np.array([0])
            return 1000.0,

def eval_rl(individual, environment, output_transform):
    try:
        func = toolbox.compile(expr=individual)
        R = 0
        io_feat = []
        for s in range(5):
            terminated = False
            truncated = False
            obs, _ = environment.reset(seed=s)
            done = False
            while not (terminated or truncated):
                obs = list(obs)
                o = func(*obs)
                action = output_transform(o)
                io_feat.append((obs, o))
                obs, r, terminated, truncated, _  = environment.step(action)
                R += float(r)
        if args.use_auxiliary:
            individual.io = np.array([io[0] for io in io_feat]), np.array([io[1] for io in io_feat])
        return -R/5,
    except OverflowError:
        if args.use_auxiliary and io_feat:
            individual.io = np.array([io[0] for io in io_feat]), np.array([io[1] for io in io_feat])
        elif args.use_auxiliary:
            individual.io = np.array([obs]), np.array([0])
        return 100000.0,

# register the selection and genetic operators - tournament selection and, one point crossover and sub-tree mutation
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# set height limits for the trees
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def run_baseline(i, bench, out_prefix=""):
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

    save_training_data = args.save_training_data
    if args.save_training_data:
        subdir = f"{args.save_dir}/train_data" if args.save_dir is not None else 'train_data'
        os.makedirs(subdir, exist_ok=True)
        save_training_data = f'{subdir}/{out_prefix}.r{i}.g'

    # run the baseline algorithm
    pop, log = algo.ea_baseline_simple(pop, toolbox, 0.2, 0.7, args.max_evals,
                                       pset=pset, stats=mstats, halloffame=hof, verbose=True, n_jobs=1, 
                                       save_data=save_training_data)

    return pop, log, hof


def run_model_test(i, bench, out_prefix=""):
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

    save_training_data = args.save_training_data
    if args.save_training_data:
        subdir = f"{args.save_dir}/train_data" if args.save_dir is not None else 'train_data'
        os.makedirs(subdir, exist_ok=True)
        save_training_data = f'{subdir}/{out_prefix}.r{i}.g'

    # run the baseline algorithm
    pop, log, feat_imp = algo.ea_baseline_model(pop, toolbox, 0.2, 0.7, 110,
                                                stats=mstats, halloffame=hof, verbose=True, n_jobs=1, pset=pset,
                                                surrogate_cls=surrogate_cls, surrogate_kwargs=surrogate_kwargs, 
                                                save_data=save_training_data)

    return pop, log, hof, feat_imp

def run_surrogate(i, bench, out_prefix=""):
    """ Executes one run of the surrogate algorithm

    :param i: number of the run
    :param bench: the benchmark to use
    :return: population in the last generation, log of the run, and the hall-of-fame,
    """

    global surrogate_cls
    global surrogate_kwargs

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

    if args.use_surrogate == 'IDEAL':
        surrogate_cls = surrogate.IdealSurrogate
        surrogate_kwargs = {'fn': lambda x: toolbox.evaluate(x)[0]}

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

    save_training_data = args.save_training_data
    if args.save_training_data:
        subdir = f"{args.save_dir}/train_data" if args.save_dir is not None else 'train_data'
        os.makedirs(subdir, exist_ok=True)
        save_training_data = f'{subdir}/{out_prefix}.r{i}.g'

    # run the surrogate algorithm
    if args.use_local_search :
        pop, log = algo.ea_surrogate_localsearch(pop, toolbox, 0.2, 0.7, args.max_evals, pset=pset,
                                                 stats=mstats, halloffame=hof, verbose=True, n_jobs=1, scale=scale,
                                                 train_fit_lim=train_fit_lim,
                                                 surrogate_cls=surrogate_cls, surrogate_kwargs=surrogate_kwargs,
                                                 retrain_every=args.retrain_every, max_train_size=args.max_train_size,
                                                 save_data=save_training_data)
    else: 
        pop, log = algo.ea_surrogate_simple(pop, toolbox, 0.2, 0.7, args.max_evals, pset=pset,
                                            stats=mstats, halloffame=hof, verbose=True, n_jobs=1, scale=scale,
                                            train_fit_lim=train_fit_lim,
                                            surrogate_cls=surrogate_cls, surrogate_kwargs=surrogate_kwargs,
                                            retrain_every=args.retrain_every, max_train_size=args.max_train_size,
                                            save_data=save_training_data)

    return pop, log, hof


def run_all(fn, log_prefix, repeats=25):
    """ Wrapper to start 25 runs of the algorithm and store the results
    """
    import multiprocessing
    import functools
    import hashlib
    import json

    pool = multiprocessing.Pool(args.n_cpus)
    pdlogs = pd.DataFrame()

    # get the name of the benchmark
    b_name = benchmark_description[bench_number]['name']

    args_hash = hashlib.sha1(str(args).encode('utf8')).hexdigest()[:10]
    out_prefix = f'{log_prefix}.{b_name}.v{version}.{args_hash}'

    runner = functools.partial(fn, bench=benchmark_description[bench_number], out_prefix=out_prefix)

    # run the 25 runs
    logs = pool.map(runner, range(repeats))
    for i, l in enumerate(logs):
        _, log, _ = l
        pdlog = pd.Series(log.chapters['fitness'].select('min'), index=np.cumsum(log.select('nevals')),
                          name='run_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)
        pdlog = pd.Series(log.select('elapsed_time'), index=np.cumsum(log.select('nevals')),
                          name='time_' + str(i))
        pdlogs = pd.concat([pdlogs, pdlog], axis=1)

    # store the logs
    pdlogs.to_csv(f'output/{out_prefix}.csv')
    with open(f'output/{out_prefix}.json', 'w') as f:
        json.dump(vars(args), f, indent=1)



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
    if not os.path.exists('output/'):
        os.mkdir('output')
    print(os.getcwd())
    #run_all_model_tests()
    #run_all_baseline()
    #run_all_surrogate()
    #run the benchmark on the selected function
    filename = f"{args.filename}_" if args.filename is not None else ""
    if args.use_surrogate:
        prefix = 'surrogate.' if not args.use_local_search else 'surrogate-ls.'
        prefix = filename + prefix
        run_all(fn=run_surrogate, log_prefix=prefix + surrogate_name, repeats=args.repeats)
    else:
        run_all(fn=run_baseline, log_prefix=filename + 'baseline', repeats=args.repeats)

if __name__ == "__main__":
    main()
