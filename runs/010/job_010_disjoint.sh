#!/bin/bash
#SBATCH -J pid010d
#SBATCH -o err_010_disjoint.out
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -x a1,a2
#SBATCH -t 08:00:00

# Run 010 con partículas distintas para entrenar y probar (dos grupos de 20000 por especie): mismo análisis
# que job_010.sh, sobre las mismas salidas (hasta 3701), para ver si compartir partículas inflaba el acierto.
# Resultado: train_pares_disjoint.log (comparar con train_pares.log)
# (Su log salió de una versión anterior de train_pares.py, sin las variantes de vecinos, aceleración y St real.)
ml python
export OMPI_MCA_pml=ob1   # en este cluster, MPI con UCX se cuelga o falla dentro de SLURM
D=/share/data4/msvidal/pid010
mpirun --use-hwthread-cpus --bind-to hwthread -np 8 python -W ignore pares_dataset.py \
    --yaml simuls_pares.yaml --out-dir $D/moments_disjoint --last-file 3701 --disjoint
python -u -W ignore train_pares.py --moments $D/moments_disjoint > train_pares_disjoint.log 2>&1
