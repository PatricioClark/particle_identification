#!/bin/bash
#SBATCH -J pid007ev
#SBATCH -o err_eval_features.out
#SBATCH -p normal
#SBATCH -w a1
#SBATCH -N 1
#SBATCH -n 9
#SBATCH -t 06:00:00

# Muestras de la corrida nueva RND512 (otra semilla + ONLD St~3.2), una especie por proceso,
# descartando las primeras 50 salidas. Dos versiones: escalas propias de la corrida y escalas
# de entrenamiento de RND512. Se manda con --dependency=afterany:<job de GHOST>.
ml python
for SC in own train; do
    mpirun --use-hwthread-cpus --bind-to hwthread --mca pml ob1 -np 9 \
        python -W ignore eval_new_run.py features --scales $SC --skip 50 \
        --flow-params flow_params_007.json --out-dir new_run/$SC
done
