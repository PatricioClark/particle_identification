#!/bin/bash
#SBATCH -J pid008
#SBATCH -o err_tramos.out
#SBATCH -p normal
#SBATCH -w a1
#SBATCH -N 1
#SBATCH -n 21
#SBATCH -t 12:00:00

# Run 008: dataset con muestras por tramo (tramos_dataset.py) para las 21 simulaciones de
# entrenamiento y las 9 especies de la corrida nueva, y entrenamiento/evaluación con validación
# separada por tramos (tramos_train.py). Resultado: train_tramos.log
ml python
export OMPI_MCA_pml=ob1   # en este cluster, MPI con UCX se cuelga o falla dentro de SLURM
mpirun --use-hwthread-cpus --bind-to hwthread -np 21 python -W ignore tramos_dataset.py \
    --yaml simuls.yaml --out-dir ckpt
mpirun --use-hwthread-cpus --bind-to hwthread -np 9 python -W ignore tramos_dataset.py \
    --yaml simuls_nueva.yaml --out-dir ckpt_nueva
python -W ignore tramos_train.py --ckpt ckpt --ckpt-nueva ckpt_nueva > train_tramos.log 2>&1
