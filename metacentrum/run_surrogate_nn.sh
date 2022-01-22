#!/bin/bash

USER_DIR=/storage/praha1/home/$USER/
GP_DIR=/storage/brno3-cerit/home/$USER/

module add conda-modules
conda activate $USER_DIR/.conda/envs/gpsurrogate

cd $GP_DIR
cd gp-surrogate 
python gpRegression.py -P $PROB_NUM -C 16 -S -N

