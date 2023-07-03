#!/bin/bash
#SBATCH -J test
#SBATCH -p cas_v100nv_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time=18:00:00
#SBATCH --comment tensorflow
#SBATCH --gres=gpu:1

srun python train_ae.py --gpu 0 --epochs 60 --batch_size 32 --lr 1e-3 --save

exit 0