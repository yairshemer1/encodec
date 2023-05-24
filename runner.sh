#!/bin/sh

#SBATCH --time=5-0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --gres=gpu:1,vmem:24g
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH -o /cs/usr/yair.shemer/AudioLab/encodec/slurm_logs/%j.out

/cs/labs/adiyoss/yair.shemer/venv/encodec/bin/python /cs/usr/yair.shemer/AudioLab/encodec/train.py restart=False exp_name="dset_medium_lr_d  isc_3e-5_lr_gen_3e-4" wandb=True batch_size=80 dset=valentini