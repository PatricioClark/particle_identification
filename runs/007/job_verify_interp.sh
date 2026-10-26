#!/bin/bash
#SBATCH -J pid007vi
#SBATCH -p normal
#SBATCH -w g2
#SBATCH -o err_verify_interp.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00

# Verificación de la interpolación de lags (verify_interp.py): igual a 006 con lags enteros,
# y prueba de submuestreo con tres simulaciones de TG768.
ml python
python -u -W ignore verify_interp.py --yaml simuls.yaml --n-particles 2000 --n-steps 400
