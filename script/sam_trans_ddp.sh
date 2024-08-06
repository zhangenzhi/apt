#!/bin/bash
#SBATCH -A bif146
#SBATCH -o sam_trans_ddp.o%J
#SBATCH -t 02:00:00
#SBATCH -N 32
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
srun -N 32 -n 256 --ntasks-per-node 8 /lustre/orion/bif146/world-shared/gvit/env/miniconda3/envs/gvit/bin/python ./train/sam_trans_ddp.py \
        --data_dir=../paip/output_images_and_masks \
        --resolution=8192 \
        --fixed_length=8281 \
        --patch_size=8 \
        --pretrain=sam-b \
        --epoch=1000 \
        --batch_size=1 \
        --savefile=./sam-b-trans-res8k-f8k-pz8-n32-2
# 8281