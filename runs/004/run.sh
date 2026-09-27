#!/bin/bash

#SBATCH -J pid004
#SBATCH -o err.out
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00

ml python
mpirun --use-hwthread-cpus --bind-to hwthread -np 20 \
    python batch_convergence.py \
    --path /share/scratch8/bespanol/MR/St8-89/ \
    --batch-sizes 1000,2500,5000,7500,10000 \
    --n-repeats 50 --n-lags 5 --max-steps 251 \
    --n-load 50000 --working-batch-size 5000 \
    --output batch_convergence.pdf
