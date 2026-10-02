#!/bin/bash

#SBATCH -J pid006i
#SBATCH -o err_inspect.out
#SBATCH -N 1
#SBATCH -n 21
#SBATCH -t 48:00:00

# Señales vs lag en tiempos de giro (T = L/U) para las 21 simulaciones de 006 (un proceso por simulación)
ml python
mpirun --use-hwthread-cpus --bind-to hwthread -np 21 python -u inspect_signals_adim.py \
    --yaml simuls.yaml --n-particles 2000 --n-lags 50 --t-max 20 \
    --mark-lags 0.0668,0.3339,0.868,3.0047,8.3464 --window 16.69 \
    --output inspect_signals_adim.pdf
