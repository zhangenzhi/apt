#!/bin/bash
#SBATCH -A bif146
#SBATCH -o apt-sam-b.o%J
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -p batch

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

export PATH="/lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/bin:$PATH"

# set +x
# source /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/envs/gvit

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# exec
srun /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/sam_trans.py \
        --data_dir=../paip/output_images_and_masks \
        --resolution=1024 \
        --fixed_length=1024 \
        --patch_size=8 \
        --pretrain=sam-b \
        --epoch=400 \
        --batch_size=4 \
        --savefile=./sam-b-trans-res1k-f1024-pz8/2