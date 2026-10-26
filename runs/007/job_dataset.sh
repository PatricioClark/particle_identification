#!/bin/bash
#SBATCH -J pid007
#SBATCH -o err_dataset.out
#SBATCH -p normal
#SBATCH -w a1
#SBATCH -N 1
#SBATCH -n 21
#SBATCH -t 12:00:00

# Dataset de 007: igual que 006 (21 simulaciones, tiempos de giro, features normalizadas) pero
# con los lags en el t* exacto (--interp-lags): en 512 pasan de 1,6,15,53,148 a
# 1.18,5.91,15.36,53.16,147.67 pasos, y las estadísticas se interpolan entre salidas.
# Se manda con --dependency=afterany:<job de GHOST> para que arranque cuando se libere a1.
ml python
# --mca pml ob1: en sakura, MPI con UCX se cuelga o falla dentro de SLURM (MPI_ERR_OTHER)
mpirun --use-hwthread-cpus --bind-to hwthread --mca pml ob1 -np 21 \
    python -W ignore build_dataset.py --yaml simuls.yaml \
    --lags-tstar 0.0668,0.3339,0.868,3.0047,8.3464 --window-tstar 16.693 \
    --flow-params flow_params_007.json --interp-lags \
    --checkpoint-dir ckpt --output dataset.pkl \
    --batches-per-shard 100 --n-shards-per-sim 1
