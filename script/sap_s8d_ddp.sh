#!/bin/bash
#SBATCH -A lrn075
#SBATCH -o sap_s8d_ddp.o%J
#SBATCH -t 02:00:00
#SBATCH -N 32
#SBATCH -p batch
#SBATCH --mail-user=zhangsuiyu657@gmail.com
#SBATCH --mail-type=END

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

source export_ddp_envs.sh

module load PrgEnv-gnu/8.5.0
module load gcc/12.2.0
module load rocm/6.2.0

# exec
srun -N 32 -n 256 --ntasks-per-node 8 python ./train/sap_s8d_ddp.py \
        --data_dir=/lustre/orion/nro108/world-shared/enzhi/Riken_XCT_Simulated_Data/8192x8192_2d_Simulations/Noise_0.05_Blur_2_sparsity_2_NumAng_3600 \
        --epoch=1000 \
        --resolution=8192 \
        --fixed_length=10201 \
        --patch_size=8 \
        --pretrain=sam-b \
        --epoch=100 \
        --batch_size=1 \
        --savefile=./sap_s8d_n32
# 8281