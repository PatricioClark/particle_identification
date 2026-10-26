# Pruebas para `particle_id`

## Estructura

- Raíz, `inspect/`, `docs/`, `simuls.yaml`, `CLAUDE.md`: código y docs compartidos (igual que `main`).
- `runs/`: corridas. Cada carpeta tiene una copia de los scripts tal como se usaron en esa corrida, más su `run.sh`/`job.sh`.
- `notes/`: notas (resumen, reuniones, simulaciones).

## Corridas (`runs/`)

- 001: primera prueba, con todas las particulas y sin lags iniciales
- 002: agrego lags iniciales, saco las BB. También bajo max-steps y n-lags
- 003: simul TG nueva con mulitpart
- 004: corro `inspect/batch_converge.py` para ver si tomar 5000 partículas está bien
- 005: réplica de 002 (mismo código, yaml y parámetros) + `compare.py` para verificar que el dataset sea idéntico; `inspect_signals` e `inspect_energetics` para entender la elección de lags
- 006: todas las simulaciones de `simuls.yaml` (768 + 512, sin ONLD 136). Parte 1: `lags_adim.py` pasa los lags a tiempos de giro y verifica que en 768 se recuperan los de 002
- 007: lags en el t* exacto, interpolando entre salidas (`features.py`, `--interp-lags`); evaluación de la corrida nueva de RND512 con otra semilla (`eval_new_run.py`) y control con tramos de la corrida original
- 008: dataset con muestras por tramo (todas las partículas en un mismo tramo de 16.7 T, escalas del tramo) y validación separando por tramos (`tramos_dataset.py`, `tramos_train.py`)


## GHOST

`GHOST/` es un clon aparte de [pmininni/GHOST](https://github.com/pmininni/GHOST), branch `new_paradigm`. Está en el `.gitignore` de este repo; se actualiza con `git pull` adentro de `GHOST/`.

- `src/main.fpp` es un symlink a `/home/clark/repos/multipart/main.fpp` (versión multipartícula de Patricio, en su propio repo).
- Compilación (CMake, ver `GHOST/build/CMakeCache.txt`): `SOLVER=GHOST`, `PRECISION=SINGLE`, `FFTP=fftp-3`, gfortran 15.2.0 y FFTW 3.3.11 de `/opt/ohpc`.
- Ignores locales (no se suben) en `GHOST/.git/info/exclude`: `build/` y salidas de `bin/`. `src/main.fpp` está marcado con `skip-worktree`; para volver a verlo: `git -C GHOST update-index --no-skip-worktree src/main.fpp`.
