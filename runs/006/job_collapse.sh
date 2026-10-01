#!/bin/bash

#SBATCH -J pid006c
#SBATCH -o err_collapse.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00

# Prueba de colapso: trazadores TG768 vs casi-trazadores RND512 (St = 0.1), con cada tiempo característico
# (LAG_RND512 no se usa: sus posiciones están mal, ver notes/bitacora.md)
ml python
python -u collapse_test.py --sims "TG768 LAG,RND512 HPP 0.1,RND512 ONLD 0.1" \
    --n-particles 5000 --max-steps 900 --n-lags 60 --output collapse_test.pdf
