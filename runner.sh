#!/bin/sh

#SBATCH --time=5-0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --gres=gpu:1,vmem:24g
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH -o /cs/usr/yair.shemer/AudioLab/encodec/slurm_logs/%j.out

/cs/labs/adiyoss/yair.shemer/venv/encodec/bin/python /cs/usr/yair.shemer/AudioLab/encodec/train.py exp_name="with_rvq" batch_size=80 dset=valentini