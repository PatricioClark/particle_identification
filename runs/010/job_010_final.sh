#!/bin/bash
#SBATCH -J pid010f
#SBATCH -o err_010_final.out
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 9
#SBATCH -x a1,a2
#SBATCH -t 08:00:00

# Run 010 con todos los datos de las corridas pareadas: salidas hasta 3926 (estado 158, t = 117.75; después
# el flujo se volvió inestable), ~28 T útiles. Partículas distintas para entrenar y probar. Bloques de 6 y 4 T,
# con y sin features de aceleración, dejando afuera un bloque o también sus vecinos; St real / etiqueta por bloque.
# Resultado: train_pares_final.log
ml python
export OMPI_MCA_pml=ob1   # en este cluster, MPI con UCX se cuelga o falla dentro de SLURM
D=/share/data4/msvidal/pid010
mpirun --use-hwthread-cpus --bind-to hwthread -np 9 python -W ignore pares_dataset.py \
    --yaml simuls_pares.yaml --out-dir $D/moments_3926 --last-file 3926 --disjoint
python -u -W ignore train_pares.py --moments $D/moments_3926 > train_pares_final.log 2>&1
