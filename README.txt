These are the source codes of the surrogate-based genetic programming algorithms as they were used to create the
PPSN 2016 paper.

The main file is gpRegression.py. It contains three functions which execute the baseline, the surrogate algorithm,
or the tests of the models. These are called run_all_baseline, run_all_surrogate, and run_all_models respectivelly.

The feature selection is implemented in surrogate.py.

The algorithm itself is implemented in algo.py, again with three functions for the baseline, surrogate, and for the
version to test the models. These are called ea_baseline_simple, ea_surrogate_simple and ea_baseline_model.

The settings of the benchmarks are in benchmarks.py and the input training files are in the benchmarks directory. These
input files were created using the ECJ library in Java.

The outputs are put into the output directory in the csv format. There is (except for the feature imporances) one
csv file for each 25 runs with the fitness of the best individual after a given number of evaluations. Each run
is in one column. For the feature importances, there is one csv for each run with each column corresponding to one
feature.