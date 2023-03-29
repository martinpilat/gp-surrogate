#!/bin/bash

USER_DIR=/storage/plzen1/home/$USER/
cd $USER_DIR

module add conda-modules
module add gcc

conda create --prefix $USER_DIR/.conda/envs/gpsurrogate python=3.10
conda activate gpsurrogate

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install numpy pandas scikit-learn

tmp_dir=$USER_DIR/tmp
pip_args="--cache-dir=$tmp_dir"

TMPDIR=$tmp_dir pip install "$pip_args" --upgrade pip
TMPDIR=$tmp_dir pip install "$pip_args" deap 
TMPDIR=$tmp_dir pip install "$pip_args" typing_extensions
TMPDIR=$tmp_dir pip install "$pip_args" gym
TMPDIR=$tmp_dir pip install "$pip_args" pytorch-tree-lstm
conda install swig
conda install seaborn
pip install optuna
TMPDIR=$tmp_dir pip install "$pip_args" gym[box2d]
TMPDIR=$tmp_dir pip install "$pip_args" gym[mujoco]
