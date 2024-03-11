#!/bin/bash
#SBATCH -J dm
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH -o /public/home/jd_yangfeng/lgr/FedDM4/CBAFed-main/output/1fmnist阈值不限制local全局扩散20倍扰动117.txt
#SBATCH -e /public/home/jd_yangfeng/lgr/FedDM4/CBAFed-main/error/1fmnist阈值不限制local全局扩散20倍扰动117.txt
#SBATCH -p gpu1
cd  $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS
CUDA_VISIBLE_DEVICES=$NP python train_main.py --dataset='fmnist' --gen_path="/public/home/jd_yangfeng/lgr/FedDM/fed-ddim-main/exp/image_samples/fmnist"
