#!/bin/bash
#SBATCH -A bif146
#SBATCH -o apt-qdt.o%J
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
srun /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/unet_miccai.py \
        --data_dir=../miccai_patches/ \
        --resolution=512 \
        --epoch=10 \
        --batch_size=16 \
        --savefile=./unet_miccai-16-16