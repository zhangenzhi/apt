#!/bin/bash
#SBATCH -A bif146
#SBATCH -o sam_miccai.o%J
#SBATCH -t 06:00:00
#SBATCH -N 128
#SBATCH -p batch
#SBATCH --mail-user=zhangsuiyu657@gmail.com
#SBATCH --mail-type=END

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

source export_ddp_envs.sh

# export PATH="/lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/bin:$PATH"

# set +x
# source /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/etc/profile.d/conda.sh
# conda activate /lustre/orion/bif146/world-shared/gvit/dataset/miniconda_frontier/envs/gvit

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# # exec
# srun -N 1 -n 8 --ntasks-per-node 8 /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/sam_miccai.py \
#         --data_dir=../miccai_patches/ \
#         --resolution=512 \
#         --lr=1e-4 \
#         --epoch=10 \
#         --batch_size=4 \
#         --patch_size=8 \
#         --pretrain=sam-b \
#         --savefile=./sam-b_miccai-n1-pz8-bz4


srun -N 128 -n 1024 --ntasks-per-node 8 /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/sam_miccai.py \
        --data_dir=../miccai_patches/ \
        --resolution=2048 \
        --lr=1e-4 \
        --epoch=100 \
        --batch_size=2 \
        --patch_size=32 \
        --pretrain=sam-b \
        --savefile=./sam-b_miccai-n128-r2k-pz16-bz2-c2