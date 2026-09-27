#!/bin/bash

#SBATCH -J pid002
#SBATCH -o err.out
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00

ml python
mpirun --use-hwthread-cpus --bind-to hwthread -np 20 \
    python build_dataset.py --max-steps 251 --n-lags 5 \
    --checkpoint-dir ckpt --output dataset.pkl \
    --batches-per-shard 20 --n-shards-per-sim 5
python eda.py
python train_classifier.py
