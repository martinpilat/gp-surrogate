#!/bin/bash

USER_DIR=/storage/praha1/home/$USER/
cd $USER_DIR

module add conda-modules
module add gcc

conda create --name gpsurrogate python=3.8
conda install numpy pandas scikit-learn
conda activate gpsurrogate

pip install deap
pip install typing_extensions
pip install torch
pip install torch_geometric
pip install torch_sparse
pip install torch_scatter
pip install gym
pip install pytorch-tree-lstm