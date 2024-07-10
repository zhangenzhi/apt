#!/bin/bash
#SBATCH -A bif146
#SBATCH -o unet_miccai.o%J
#SBATCH -t 02:00:00
#SBATCH -N 2
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

# exec
srun -N 2 -n 16 --ntasks-per-node 8 /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/unet_miccai.py \
        --data_dir=../miccai_patches/ \
        --resolution=512 \
        --epoch=10 \
        --batch_size=8 \
        --savefile=./unet_miccai-16-2