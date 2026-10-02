#!/bin/bash

#SBATCH -J pid006v
#SBATCH -o err_verif002.out
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00

# Verificación (paso B3): el build_dataset nuevo, en modo tiempos de giro y SIN normalizar,
# con las simulaciones y parámetros de 002, tiene que dar el mismo dataset que 002/005.
ml python
# --mca pml ob1: en sakura, MPI con UCX se cuelga o falla dentro de SLURM (MPI_ERR_OTHER)
mpirun --use-hwthread-cpus --bind-to hwthread --mca pml ob1 -np 20 \
    python -W ignore build_dataset.py --yaml simuls_002.yaml \
    --lags-tstar 0.0668,0.3339,0.868,3.0047,8.3464 --window-tstar 16.693 --no-normalize --flow-params flow_params_002.json \
    --checkpoint-dir ckpt_verif002 --output dataset_verif002.pkl \
    --batches-per-shard 20 --n-shards-per-sim 5
python compare.py dataset_verif002.pkl ../005/dataset.pkl \
    --ckpt ckpt_verif002 --ref-ckpt ../005/ckpt --strip-label-suffix _TG768
