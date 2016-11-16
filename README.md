# Feature Extraction for Surrogate Models in Genetic Programming

This projects applies the ideas from surrogate modeling in genetic programming. The main idea, is to build a model, which can predict the fitness of an individual without evaluating it. Surrogate modeling is widely used in evolutionary continuous optimization, however, outside this are, it is used only rarely. Here, we provide a proof-of-concept and show that surrogate modeling can be successfully applied in genetic programming.

The surrogate models in this project are used to pre-filter individuals, i.e. individuals are first evaluated by the surrogate model and only those which seem promising are evaluated with the real fitness function. The other individuals are replaced by their parents and therefore do not need to be evaluated.

## Feature Selection

In this work, we use a random forest to predict the fitness of the individual based on the following features extracted from each individual in the population:

1. Tree features 
    1. number of nodes
    2. depth of tree
2. Features on constants 
	1. maximum value of a constant
	2. minimum value of a constant
	3. mean of the values of the constants
	4. number of constants (divided by tree size)
	5. number of distinct values (divided by the number of constants)
3. Features on arguments
	1. proportion of arguments used
	2. average number of uses of an argument
	2. number of times an argument is used divided by the tree size
4. Features on terminals/primitives
	1. number of times terminal/primitive is used divided by the tree size
5. Features on parents
	1. fitness of the worse parent
	2. fitness of the better parent
	3. mean of parents fitness

## Results

The following image show the fitness of the individual (the logarithm of mean square error between the function represented by the individual and the correct value from the training set). We can see that for three out of the four experiments the number of fitness evaluations needed to obtain a solution of given quality is significantly decreased foe evaluation budget between 2,000 and 6,000 fitness evaluations.

![Results on the symbolic regression benchmarks](https://martinpilat.com/media/research/images/PPSN2016results.png)

## The Source Codes

These are the source codes of the surrogate-based genetic programming algorithms as they were used to create the PPSN 2016 paper.

The main file is `gpRegression.py`. It contains three functions which execute the baseline, the surrogate algorithm, or the tests of the models. These are called `run_all_baseline`, `run_all_surrogate`, and `run_all_models` respectively.

The feature selection is implemented in `surrogate.py`.

The algorithm itself is implemented in `algo.py`, again with three functions for the baseline, surrogate, and for the version to test the models. These are called `ea_baseline_simple`, `ea_surrogate_simple`, and `ea_baseline_model`.

The settings of the benchmarks are in `benchmarks.py` and the input training files are in the benchmarks directory. These input files were created using the ECJ library in Java.

The outputs are put into the output directory in the csv format. There is (except for the feature importances) one `.csv` file for each 25 runs with the fitness of the best individual after a given number of evaluations. Each run is in one column. For the feature importances, there is one `.csv` for each run with each column corresponding to one feature.

## Citation

The full results were published in a PPSN 2016 paper ([available from Springer][springer]).

__Martin Pilát and Roman Neruda__. _Feature Extraction for Surrogate Models in Genetic Programming_. In: Parallel Problem Solving from Nature – PPSN XIV, Proceedings of 14th International Conference, Edinburgh, UK, September 17-21, 2016. LNCS 9921. Springer 2016, pp. 335-344. ISBN 978-3-319-45822-9. ISSN: 0302-9743. DOI: [10.1007/978-3-319-45823-6_31][doi].

### Bibtex

	@Inbook{gp-surrogate,
	author="Pil{\'a}t, Martin
	and Neruda, Roman",
	editor="Handl, Julia
	and Hart, Emma
	and Lewis, Peter R.
	and L{\'o}pez-Ib{\'a}{\~{n}}ez, Manuel
	and Ochoa, Gabriela
	and Paechter, Ben",
	title="Feature Extraction for Surrogate Models in Genetic Programming",
	bookTitle="Parallel Problem Solving from Nature -- PPSN XIV: 14th International Conference, Edinburgh, UK, September 17-21, 2016, Proceedings",
	year="2016",
	publisher="Springer International Publishing",
	address="Cham",
	pages="335--344",
	isbn="978-3-319-45822-9",
	doi="10.1007/978-3-319-45823-6_31",
	url="http://dx.doi.org/10.1007/978-3-319-45823-6_31"
	}

  [springer]: http://link.springer.com/chapter/10.1007/978-3-319-45823-6_31
  [doi]: http://dx.doi.org/10.1007/978-3-319-45823-6_31