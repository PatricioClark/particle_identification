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


## GHOST

`GHOST/` es un clon aparte de [pmininni/GHOST](https://github.com/pmininni/GHOST), branch `new_paradigm`. Está en el `.gitignore` de este repo; se actualiza con `git pull` adentro de `GHOST/`.

- `src/main.fpp` es un symlink a `/home/clark/repos/multipart/main.fpp` (versión multipartícula de Patricio, en su propio repo).
- Compilación (CMake, ver `GHOST/build/CMakeCache.txt`): `SOLVER=GHOST`, `PRECISION=SINGLE`, `FFTP=fftp-3`, gfortran 15.2.0 y FFTW 3.3.11 de `/opt/ohpc`.
- Ignores locales (no se suben) en `GHOST/.git/info/exclude`: `build/` y salidas de `bin/`. `src/main.fpp` está marcado con `skip-worktree`; para volver a verlo: `git -C GHOST update-index --no-skip-worktree src/main.fpp`.
