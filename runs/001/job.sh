#!/bin/bash

#SBATCH -J build_datset
#SBATCH -o err.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00

ml python
python -u build_dataset.py
