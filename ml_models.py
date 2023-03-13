import argparse
import pickle

from deap import creator, base, gp

from gp_surrogate import benchmarks

parser = argparse.ArgumentParser(description='Run GP with surrogate model')
parser.add_argument('--problem_number', '-P', type=int, help='The number of problem to start', default=0)

args = parser.parse_args()

pset = benchmarks.benchmark_description[args.problem_number]

# create the types for fitness and individuals
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

with open('train_data/surrogate.RAND-AUX.keijzer-6.v1.6c6d8c8a46.13.pkl', 'rb') as f:
    data = pickle.load(f)

print("individual:" , data[0][0])
print("features:" , data[0][28].features.values)
print("inputs/outputs for aux:", data[0][27].io[0])
print("inputs/outputs for aux:", data[0][30].io[1])
print("fitness:", data[1][0][0])