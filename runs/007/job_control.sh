#!/bin/bash
#SBATCH -J pid007ct
#SBATCH -o err_control.out
#SBATCH -p normal
#SBATCH -w a2
#SBATCH -N 1
#SBATCH -n 9
#SBATCH -t 06:00:00

# Control de la prueba de semilla (eval_new_run.py, escalas propias, modelo de 007):
#  - corrida ORIGINAL de bespanol, tramos de 313 salidas (lo mismo que se usó de la corrida nueva):
#      baja energía  (skip 2610, t = 78–88, <v²> = 0.72, como la corrida nueva)
#      alta energía  (skip 2240, t = 67–77, <v²> = 1.33)
#      intermedio    (skip 1000, t = 30–40)
#  - corrida NUEVA descartando 0 y 100 salidas al principio (la evaluación base descartó 50).
ml python
ORIG=/share/data4/bespanol/HIT512_multipart/bin
run() {  # $1 = out dir, resto = opciones de eval_new_run.py features
    out=$1; shift
    mpirun --use-hwthread-cpus --bind-to hwthread --mca pml ob1 -np 9 \
        python -W ignore eval_new_run.py features --scales own --flow-params flow_params_007.json \
        --out-dir $out "$@"
    python -W ignore eval_new_run.py predict --model model.pkl --out-dir $out > predict_$(basename $out).log 2>&1
}
run control/orig_baja     --run-dir $ORIG --skip 2610 --n-out 313
run control/orig_alta     --run-dir $ORIG --skip 2240 --n-out 313
run control/orig_media    --run-dir $ORIG --skip 1000 --n-out 313
run control/nueva_skip0   --skip 0
run control/nueva_skip100 --skip 100
