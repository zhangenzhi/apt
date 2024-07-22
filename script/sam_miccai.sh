#!/bin/bash
#SBATCH -A bif146
#SBATCH -o sam_miccai.o%J
#SBATCH -t 06:00:00
#SBATCH -N 92
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


srun -N 92 -n 736 --ntasks-per-node 8 /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/sam_miccai.py \
        --data_dir=../miccai_patches/ \
        --resolution=4096 \
        --lr=1e-4 \
        --epoch=1000 \
        --batch_size=1 \
        --patch_size=64 \
        --pretrain=sam-b \
        --savefile=./sam-b_miccai-n32-r4k-pz64-bz1-c2