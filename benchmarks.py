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
