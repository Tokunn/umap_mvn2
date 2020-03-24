#!/usr/bin/env sh

python3 incremental_pca.py ./dataset_all/cable --batch-size=100 --kfold=1 --pngdir=debug_unnorm/wood --prmc=1.0 --seed=55 --useparam=0.999 --uselayer=19 --usereject --vino
