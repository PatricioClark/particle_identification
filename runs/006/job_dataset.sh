#!/bin/bash

#SBATCH -J pid006
#SBATCH -o err_dataset.out
#SBATCH -N 1
#SBATCH -n 21
#SBATCH -t 48:00:00

# Dataset de 006: las 21 simulaciones (13 TG768 + 8 RND512), lags y ventana en tiempos de
# giro (T = L/U, uno por flujo, ver flow_params_006.json) y features normalizadas.
# Un shard por simulación con 100 muestras: mismas muestras que 002, 5 veces menos lectura.
ml python
# --mca pml ob1: en sakura, MPI con UCX se cuelga o falla dentro de SLURM (MPI_ERR_OTHER)
mpirun --use-hwthread-cpus --bind-to hwthread --mca pml ob1 -np 21 \
    python -W ignore build_dataset.py --yaml simuls.yaml \
    --lags-tstar 0.0668,0.3339,0.868,3.0047,8.3464 --window-tstar 16.693 \
    --flow-params flow_params_006.json \
    --checkpoint-dir ckpt --output dataset.pkl \
    --batches-per-shard 100 --n-shards-per-sim 1
