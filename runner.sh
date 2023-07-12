#!/bin/sh

#SBATCH --time=5-0
#SBATCH --killable
#SBATCH --requeue
#SBATCH --gres=gpu:1,vmem:24g
#SBATCH --mem=32g
#SBATCH -c4
#SBATCH -o /cs/usr/yair.shemer/AudioLab/encodec/slurm_logs/%j.out

/cs/labs/adiyoss/yair.shemer/venv/encodec/bin/python /cs/usr/yair.shemer/AudioLab/encodec/train.py exp_name="rvq_from_scratch_galil_data_reader" batch_size=32 dset=valentini
