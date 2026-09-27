#!/bin/bash

#SBATCH -J build_datset
#SBATCH -o err_ds.out
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -t 48:00:00

ml python
mpirun --use-hwthread-cpus --bind-to hwthread -np 15 python build_dataset.py --max-steps 501
