#!/bin/bash
#SBATCH -J dm
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --gres=gpu:1
#SBATCH -o /public/home/jd_yangfeng/lgr/FedDM/fed-ddim-main/output/SVHNddim生成图片2output123.txt
#SBATCH -e /public/home/jd_yangfeng/lgr/FedDM/fed-ddim-main/error/SVHNddim生成图片2output123.txt
#SBATCH -p gpu
cd  $SLURM_SUBMIT_DIR
NP=$SLURM_JOB_GPUS
CUDA_VISIBLE_DEVICES=$NP python main.py
