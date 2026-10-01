#!/bin/bash

#SBATCH -J pid005
#SBATCH -o err_dataset.out
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00

# Réplica exacta del build_dataset de runs/002 (mismo código, yaml y parámetros)
ml python
mpirun --use-hwthread-cpus --bind-to hwthread -np 20 \
    python build_dataset.py --max-steps 251 --n-lags 5 \
    --checkpoint-dir ckpt --output dataset.pkl \
    --batches-per-shard 20 --n-shards-per-sim 5
python compare.py dataset.pkl ../002/dataset.pkl --ckpt ckpt --ref-ckpt ../002/ckpt
