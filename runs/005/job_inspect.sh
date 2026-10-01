#!/bin/bash

#SBATCH -J pid005i
#SBATCH -o err_inspect.out
#SBATCH -N 1
#SBATCH -n 11
#SBATCH -t 48:00:00

# Señales vs lag en tiempo físico para las 11 simulaciones de 002 (un proceso por simulación)
ml python
mpirun --use-hwthread-cpus --bind-to hwthread -np 11 python inspect_signals.py --yaml simuls.yaml \
    --n-particles 1000 --n-lags 50 --mark-lags 1,5,13,45,125 \
    --output inspect_signals.pdf
