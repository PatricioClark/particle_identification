#!/bin/bash
#SBATCH -J pid006null
#SBATCH -p normal
#SBATCH -w g2
#SBATCH -o null/err_null.%a.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH --array=0-3%2

# Prueba de control (null_test.py): dos grupos disjuntos de 50 000 partículas (A, B) para
# HPP St0.1 y ONLD St0.1 de RND512. Una tarea por grupo; de a 2 tareas a la vez en g2 para no juntar más de dos cargas
# (~7 GB de pico cada una) en el mismo nodo.
SIMS=(outs_HPP_0-1 outs_HPP_0-1 outs_NLD_0-1 outs_NLD_0-1)
POOLS=(A B A B)
S=${SIMS[$SLURM_ARRAY_TASK_ID]}; P=${POOLS[$SLURM_ARRAY_TASK_ID]}
ml python
python -u -W ignore null_test.py features --sim $S --pool $P --out null/${S#outs_}_$P.npz
