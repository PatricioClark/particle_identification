# Simulaciones: full_simuls.yaml vs carpetas MR / NLD / ONLD

Las carpetas `MR`, `NLD` y `ONLD` en la raíz del proyecto son symlinks que apuntan
directamente a los paths base usados en `full_simuls.yaml`:

- `MR`   -> `/share/scratch8/bespanol/MR`
- `NLD`  -> `/share/scratch8/bespanol/NLD`
- `ONLD` -> `/share/scratch12/bespanol/ONLD/`

Las simulaciones listadas en `full_simuls.yaml` son un **subconjunto** de lo que hay
en esas carpetas. Cada una tiene además una subcarpeta `src` (no es una simulación)
y, en algunos casos, corridas extra que no están incluidas en el yaml.

## MR

En el yaml: `St0-76`, `St3-2`, `St8-89`

En la carpeta: `St0-76`, `St3-2`, `St8-89`, **`St2`** (no está en el yaml), `src`

## NLD

En el yaml: `St8-89`, `St312`

En la carpeta: `St8-89`, `St312`, **`St136`** (no está en el yaml), `src`

## ONLD

En el yaml: `St3-2`, `St8-89`, `St136`, `St35-6`, `St312`

En la carpeta: `St3-2`, `St8-89`, `St136_wrong_taup`, `St35-6`, `St312`, `src`

**Ojo:** el yaml apunta a `/share/scratch12/bespanol/ONLD/St136/`, pero esa carpeta
**no existe** en disco. Lo único que hay es `St136_wrong_taup` (nombre que sugiere
un `taup` calculado mal). O sea, ese path del yaml está roto/desactualizado.

## Conclusión

`full_simuls.yaml` no incluye todas las corridas presentes en disco (faltan
`MR/St2` y `NLD/St136`), y además tiene un path roto: el `ONLD/St136` que
referencia no existe, solo existe `ONLD/St136_wrong_taup`.

## Archivo de parámetros (parameter.inp) por corrida

Cada corrida individual (cada `path:` del yaml) tiene su propio `parameter.inp`
directamente adentro, con la configuración del solver GHOST para esa simulación
(viscosidad `nu`, paso temporal `dt`, cada cuántos pasos se guardan partículas
`tstep`/`lgmult`, directorio de output `idir`/`odir`, etc). El patrón es:

```
<path del yaml>/parameter.inp
```

| Simulación (yaml) | path del `parameter.inp` |
|---|---|
| LAG St0 | `/share/scratch8/bespanol/LAG/St0/parameter.inp` |
| MR St0.76 | `/share/scratch8/bespanol/MR/St0-76/parameter.inp` |
| MR St3.2 | `/share/scratch8/bespanol/MR/St3-2/parameter.inp` |
| MR St8.89 | `/share/scratch8/bespanol/MR/St8-89/parameter.inp` |
| NLD St8.89 | `/share/scratch8/bespanol/NLD/St8-89/parameter.inp` |
| NLD St312 | `/share/scratch8/bespanol/NLD/St312/parameter.inp` |
| ONLD St3.2 | `/share/scratch12/bespanol/ONLD/St3-2/parameter.inp` |
| ONLD St8.89 | `/share/scratch12/bespanol/ONLD/St8-89/parameter.inp` |
| ONLD St136 | ⚠️ no existe; solo `/share/scratch12/bespanol/ONLD/St136_wrong_taup/parameter.inp` |
| ONLD St35.6 | `/share/scratch12/bespanol/ONLD/St35-6/parameter.inp` |
| ONLD St312 | `/share/scratch12/bespanol/ONLD/St312/parameter.inp` |
| BB St8.89 | `/share/scratch8/bespanol/BB/St8-89/parameter.inp` |
| BB St35.6 | `/share/scratch8/bespanol/BB/St35-6/parameter.inp` |
| FAX St8.89 | `/share/scratch12/bespanol/FAX/St8-89/parameter.inp` |

Como `MR`, `NLD` y `ONLD` son symlinks dentro de este proyecto, también se puede
acceder con la ruta más corta, ej: `MR/St8-89/parameter.inp` en vez del path
completo en `/share/...`.

### Estructura del parameter.inp (ejemplo BB/St8-89)

- `&status` -> `idir`/`odir = "outs"`: los outputs se guardan en la subcarpeta
  `outs/` dentro de la carpeta de la corrida.
- `&velocity` -> `nu`: viscosidad cinemática.
- `&parameter` -> `dt`: paso temporal de integración; `tstep`: cada cuántos pasos
  se escribe el output binario de los campos.
- `&plagpart` -> `lgmult`: multiplicador de output de partículas (debe dividir
  `tstep` exactamente). Las partículas se guardan cada `tstep / lgmult` pasos,
  es decir cada `(tstep / lgmult) * dt` en tiempo de simulación.
- `&pinerpart` -> `tau`: tiempo de Stokes de la partícula.

## Archivos de output en outs/ (partículas)

Dentro de `outs/` cada campo por partícula se guarda con un prefijo distinto,
seguido de una etiqueta numérica (índice de output) y `.lag`, ej.
`xlg.00000021.lag`. Prefijos encontrados (fuente: `GHOST/src/particles/inerpart_solver.f90`):

- `xlg` -> **posiciones** de las partículas (sstate_pos_)
- `vlg` -> velocidad del fluido interpolada en la posición de la partícula (sstate_lag_)
- `vip` -> velocidad de la **partícula** inercial (sstate_vel_) — **no son posiciones**
- `wip`, `ft`, `g0`, `gn` -> campos adicionales, probablemente ligados al término
  de historia de Basset-Boussinesq (`dohistory=1`, `t_win` en `&pinerpart`) en
  las corridas BB; no confirmado en este árbol de `GHOST/src`.

### Relación entre la etiqueta del archivo y el tiempo de simulación

En `GHOST/src/particles/pstatus_mod.f90`:

```fortran
IF ( mod(tstep,lgmult).NE.0 ) STOP  ! lgmult debe dividir tstep exactamente
pstep = tstep/lgmult
```

En `GHOST/src/main.fpp` (loop principal), cada `pstep` pasos de integración
se incrementa en 1 el índice de output de partículas (`pind`, la etiqueta del
archivo) y se escribe el estado:

```fortran
time = (t-1)*dt
IF (timep.eq.pstep) THEN
   timep = 0
   pind = pind + 1
   ! escribe xlg, vlg, vip, etc. con esta etiqueta
ENDIF
```

Por lo tanto:

```
tiempo(etiqueta) ≈ (etiqueta - 1) * pstep * dt        con   pstep = tstep / lgmult
```

Para `BB/St8-89` (`tstep=400`, `lgmult=20`) da `pstep = 20`, de ahí el "×20".
**Ese 20 no es una constante fija**: depende de `tstep` y `lgmult` de cada
`parameter.inp` individual, hay que recalcularlo por corrida.
