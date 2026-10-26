#!/bin/bash
#SBATCH -J pid007tp
#SBATCH -o err_train_predict.out
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00

# Clasificador de 007 (lags interpolados) y su evaluación en la corrida nueva.
# Se manda con --dependency=afterok:<dataset>:<muestras de la corrida nueva>.
ml python
python -W ignore train_classifier.py --dataset dataset.pkl --output model.pkl > train.log 2>&1
for SC in own train; do
    python -W ignore eval_new_run.py predict --model model.pkl --out-dir new_run/$SC > predict_$SC.log 2>&1
done
