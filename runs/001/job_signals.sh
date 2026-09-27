#!/bin/bash

#SBATCH -J build_datset
#SBATCH -o err_signals.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00

ml python
python -u inspect_signals.py --n-lags 100
