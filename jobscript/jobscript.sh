#!/bin/zsh

#$-l rt_G.small=1
#$-l h_rt=24:00:00
#$-cwd
#$-j y

i={0}
j={1}
k={2}
kk={3}
l={4}
source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130 cudnn/7.4/7.4.2

cd ~/Documents/umap_mnv2

# python3 ./incremental_pca.py /home/aca10370eo/Documents/umap_mnv2/dataset/toothbrush_crossval --batch-size=1
# python3 incremental_pca.py /home/aca10370eo/Documents/umap_mnv2/dataset/toothbrush_crossval --batch-size=100 --kfold=1
# python3 incremental_pca.py /home/aca10370eo/Documents/umap_mnv2/dataset/toothbrush_crossval --batch-size=100 --kfold=1 --pngdir=ee$i --prmc=$i
# python3 incremental_pca.py /home/aca10370eo/Documents/umap_mnv2/dataset/toothbrush_def --batch-size=100 --kfold=1 --pngdir=tb_mbn2layer_$i.$j --prmc=1 --uselayer=$j
# python3 incremental_pca.py $k --batch-size=100 --kfold=1 --pngdir=output/$kk._mbn12_$i.$j --prmc=1 --uselayer=$j
python3 incremental_pca.py $k --batch-size=100 --kfold=1 --pngdir=output/$kk._mbn12_$i.$j.$l --prmc=1 --uselayer=$j --seed=$l
