#!/bin/bash

USER_DIR=/storage/plzen1/home/$USER/
GP_DIR=/storage/plzen1/home/$USER/code/

module add conda-modules
conda activate $USER_DIR/.conda/envs/gpsurrogate

OUTPTH=`basename "$CFG"`"_$KOPT_$MODEL"

cd $GP_DIR
cd gp-surrogate 
python fit_model.py \
	-S $MODEL \
	-C "$GP_DIR/optuna_out/$OUTPTH/model_$OUTPTH.pt" \
	-f $CFG \
	-K $KOPT \
	-O \
	--study_name "$GP_DIR/optuna_out/db_$OUTPTH" \
	--device $device
