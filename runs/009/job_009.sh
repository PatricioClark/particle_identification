#!/bin/bash
#SBATCH -J pid009
#SBATCH -o err_009.out
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 21
#SBATCH -t 12:00:00

# Run 009: estadísticas por partícula y por ventana (16.7, 6 y 4 T) para las 21 simulaciones y la
# corrida nueva, y comparación de esquemas de entrenamiento (muestras de una ventana o mezcladas,
# propuesta de Patricio) probando siempre con bloques de tiempo no vistos. Resultado: train_schemes.log
ml python
export OMPI_MCA_pml=ob1   # en este cluster, MPI con UCX se cuelga o falla dentro de SLURM
D=/share/data4/msvidal/pid009
mpirun --use-hwthread-cpus --bind-to hwthread -np 21 python -W ignore moments_dataset.py \
    --yaml simuls.yaml --out-dir $D/moments
mpirun --use-hwthread-cpus --bind-to hwthread -np 9 python -W ignore moments_dataset.py \
    --yaml simuls_nueva.yaml --out-dir $D/moments_nueva
python -u -W ignore train_schemes.py --moments $D/moments --moments-nueva $D/moments_nueva > train_schemes.log 2>&1
