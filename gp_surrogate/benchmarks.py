from deap import gp
import operator
import math
import random


def logabs(x):
    return math.log(math.fabs(x)) if x != 0 else 0


def safediv(x, y):
    return x/y if y != 0 else 1


def safesqrt(x):
    return math.sqrt(abs(x))


def inverse(x):
    return 1/x if x != 0 else 1


def sqr(x):
    return x**2


def cube(x):
    return x**3


def v_f1(x, y):
    return abs(x)**y if x != 0 else 1


def v_f2(x, y):
    return x+y


def id(x):
    return x


def cos(x):
    return math.cos(x) if abs(x) < float('inf') else 0


def sin(x):
    return math.sin(x) if abs(x) < float('inf') else 0


def tan(x):
    try:
        return math.tan(x)
    except  ValueError:
        return 0

def concat(*args):
    return tuple(args)

class TupleOut:
    pass

def get_primitive_set_for_benchmark(benchmark_name: str, num_variables: int):
    """ Creates and returns the primitive sets for given benchmark based on its name

    :param benchmark_name: the name of the benchmark
    :param num_variables: number of variables in this benchmark
    :return: the primitive set for the benchmark
    """
    if benchmark_name.startswith('keijzer'):
        pset = gp.PrimitiveSet('MAIN', num_variables)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(inverse, 1)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(safesqrt, 1)
        pset.addEphemeralConstant('keijzer_const', lambda: random.gauss(0, 5))
        return pset

    # in fact, all the numbers in the following are float, the int is used only to ensure that the constants are used
    # only inside the functions
    if benchmark_name.startswith('vladislavleva-4'):
        pset = gp.PrimitiveSetTyped('MAIN', [float]*num_variables, float)
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(safediv, [float, float], float)
        pset.addPrimitive(sqr, [float], float, name='sqr')
        pset.addPrimitive(v_f1, [float, int], float, name='V_F1')
        pset.addPrimitive(v_f2, [float, int], float, name='V_F2')
        pset.addPrimitive(operator.mul, [float, int], float, name='V_F3')
        pset.addEphemeralConstant('vf_const', lambda: random.uniform(-5, 5), int)
        pset.addPrimitive(id, [int], int, name='id')
        return pset

    if benchmark_name.startswith('nguyen') or benchmark_name.startswith('pagie'):
        pset = gp.PrimitiveSet('MAIN', num_variables)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(safediv, 2)
        pset.addPrimitive(math.exp, 1)
        pset.addPrimitive(logabs, 1)
        pset.addPrimitive(cos, 1)
        pset.addPrimitive(sin, 1)
        return pset

    if benchmark_name.startswith('korns'):
        pset = gp.PrimitiveSet('MAIN', num_variables)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(safediv, 2)
        pset.addPrimitive(math.exp, 1)
        pset.addPrimitive(logabs, 1)
        pset.addPrimitive(cos, 1)
        pset.addPrimitive(sin, 1)
        pset.addPrimitive(sqr, 1, name='square')
        pset.addPrimitive(cube, 1, name='cube')
        pset.addPrimitive(inverse, 1, name='inverse')
        pset.addPrimitive(tan, 1)
        pset.addPrimitive(math.tanh, 1)
        pset.addEphemeralConstant('korns_const', lambda: random.uniform(-1e10, 1e10))
        return pset

    if benchmark_name.startswith('lander'):
        pset = gp.PrimitiveSetTyped('MAIN', [float]*num_variables, TupleOut)
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(safediv, [float, float], float)
        pset.addPrimitive(math.exp, [float], float)
        pset.addPrimitive(logabs, [float], float)
        pset.addPrimitive(cos, [float], float)
        pset.addPrimitive(sin, [float], float)
        pset.addPrimitive(concat, [float, float], TupleOut)
        return pset


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

def ot_swimmer(x):
    return [min(max(xx, -1), 1) for xx in list(x)]

def ot_invpendulum(x):
    return [min(max(x, -3), 3)]


def bench_by_name(name):
    for b in benchmark_description:
        if b['name'] == name:
            return b

    raise ValueError(f"Invalid name: {name}")


benchmark_description = [
    {'name': 'keijzer-6',
     'variables': 1,
     'pset': get_primitive_set_for_benchmark('keijzer-6', 1)},
    {'name': 'korns-12',
     'variables': 5,
     'pset': get_primitive_set_for_benchmark('korns-12', 5)},
    {'name': 'pagie-1',
     'variables': 2,
     'pset': get_primitive_set_for_benchmark('pagie-1', 2)},
    {'name': 'nguyen-7',
     'variables': 1,
     'pset': get_primitive_set_for_benchmark('nguyen-7', 1)},
    {'name': 'vladislavleva-4',
     'variables': 5,
     'pset': get_primitive_set_for_benchmark('vladislavleva-4', 5)},
    {'name': 'rl_cartpole',
     'env_name': 'CartPole-v1',
     'env_kwargs': {},
     'variables': 4,
     'output_transform': ot_cartpole,
     'pset': get_primitive_set_for_benchmark('pagie-1', 4)},
    {'name': 'rl_mountaincar',
     'env_name': 'MountainCar-v0',
     'env_kwargs': {},
     'variables': 2,
     'output_transform': ot_mountaincar,
     'pset': get_primitive_set_for_benchmark('pagie-1', 2)},
    {'name': 'rl_acrobot',
     'env_name': 'Acrobot-v1',
     'env_kwargs': {},
     'variables': 6,
     'output_transform': ot_acrobot,
     'pset': get_primitive_set_for_benchmark('pagie-1', 6)},
    {'name': 'rl_pendulum',
     'env_name': 'Pendulum-v1',
     'env_kwargs': {},
     'variables': 3,
     'output_transform': ot_pendulum,
     'pset': get_primitive_set_for_benchmark('pagie-1', 3)},
    {'name': 'rl_mountaincarcontinuous',
     'env_name': 'MountainCarContinuous-v0',
     'env_kwargs': {},
     'variables': 2,
     'output_transform': ot_mountaincarcont,
     'pset': get_primitive_set_for_benchmark('pagie-1', 2)},
    {'name': 'rl_lunarlander',
     'env_name': 'LunarLanderContinuous-v2',
     'env_kwargs': {},
     'variables': 8,
     'output_transform': ot_lunarlander,
     'pset': get_primitive_set_for_benchmark('lander', 8)},
    {'name': 'rl_invpendulum',
     'env_name': "InvertedPendulum-v4",
     'env_kwargs': {},
     'variables': 4,
     'output_transform': ot_invpendulum,
     'pset': get_primitive_set_for_benchmark('pagie-1', 4)},
    {'name': 'rl_invdouble',
     'env_name': "InvertedDoublePendulum-v4",
     'env_kwargs': {},
     'variables': 11,
     'output_transform': ot_mountaincarcont,
     'pset': get_primitive_set_for_benchmark('pagie-1', 11)},
    {'name': 'rl_reacher',
     'env_name': "Reacher-v4",
     'env_kwargs': {},
     'variables': 11,
     'output_transform': ot_swimmer,
     'pset': get_primitive_set_for_benchmark('lander', 11)},
    {'name': 'rl_swimmer',
     'env_name': "Swimmer-v4",
     'env_kwargs': {},
     'variables': 8,
     'output_transform': ot_swimmer,
     'pset': get_primitive_set_for_benchmark('lander', 8)}
]
