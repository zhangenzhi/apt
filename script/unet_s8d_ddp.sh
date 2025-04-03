#!/bin/bash
#SBATCH -A lrn075
#SBATCH -o unet_s8d_ddp.o%J
#SBATCH -t 02:00:00
#SBATCH -N 8
#SBATCH -p batch

#SBATCH --mail-user=zhangsuiyu657@gmail.com
#SBATCH --mail-type=END

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

source export_ddp_envs.sh

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# exec
srun -N 8 -n 64 --ntasks-per-node 8 python ./train/unet_s8d_ddp.py \
        --data_dir=/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600 \
        --epoch=100 \
        --batch_size=1 \
        --savefile=./unet-s8d-n8
# 8281