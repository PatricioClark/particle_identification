# Adimensionalización de tiempos (768 vs 512)

Objetivo: para poder sumar las simulaciones RND512 a las TG768, los lags de las features tienen que representar la misma escala física en ambas. Hoy (run 002) los lags están en **pasos de archivo** `.lag`, lo que solo funciona porque todas las TG768 tienen el mismo `dt`, `lgmult` y flujo.

## Qué se hizo en 002

- `--max-steps 251 --n-lags 5` → lags `[1, 5, 13, 45, 125]` pasos (`make_lags` en `features.py`: log-espaciados en `[1, n_steps//2]`).
- `inspect/inspect_signals.py`: grafica VACF, MSD, S2, S4 vs lag en tiempo físico, con línea en el lag máximo del pipeline. Sirve para elegir cuántos lags y hasta dónde.
- `inspect/inspect_energetics.py`: calcula U, L, T = L/U, η, τη desde `parameter.inp`, `balance.txt` y `kspectrum.*.txt`.

## Parámetros de cada dataset

| | TG768 | RND512 |
|---|---|---|
| `dt` × `lgmult` en `parameter.inp` | 8e-4 × 20 | 1.5e-3 × 25 |
| Δt entre archivos `.lag` (leído de los archivos) | 0.016 | 0.030 |
| Partículas | 1 000 000 | 500 000 |
| Caja | 2π (default, sin `boxparams`) | 2π (`Lx = 1` = 1×2π) |
| Forzado | Taylor-Green, k = 1 | random, k = 1 |
| ν | 4.5e-4 | 7e-4 |

Ojo: en RND512 el Δt real entre `.lag` (0.030) no coincide con `dt × lgmult` del `parameter.inp` de `bin/` (0.0375). Usar siempre los tiempos guardados en los archivos, como hace `dataset.py`.

Lags de 002 en tiempo físico (TG768): `0.016, 0.08, 0.21, 0.72, 2.0`. Ventana de 251 pasos = 4.0.

## `balance.txt` (resuelto)

Según GHOST (`src/pseudo/pseudospec3D_hd.f90`, subrutina `hdcheck`), las columnas son:

1. tiempo
2. `<|v|²> = <vx² + vy² + vz²>` (sin factor ½)
3. `<|ω|²>` (enstrofía, **no** es ε)
4. tasa de inyección de energía

Entonces **ε = ν<ω²>**. Verificado con los datos, en estado estacionario ν<ω²> ≈ columna 4:

| | ν<ω²> | inyección |
|---|---|---|
| TG768 | 0.234 | 0.236 |
| RND512 | 0.053 | 0.051 |

**Bug en `inspect_energetics.py`:** usa `eps = 2·nu·<ω²>` (el doble). Afecta η y τη, no el tiempo de giro. Avisar a Patricio.

## Qué U usar (resuelto)

- GHOST da el módulo 3D: `U = sqrt(<v²>)`.
- Los papers del grupo usan velocidad por componente (Angriman 2020: rms de vx; Español 2025: u′).
- Para adimensionalizar da igual: `sqrt(<v²>)` y `sqrt(<v²>/3)` difieren en √3 constante, que se cancela si se usa la misma definición en ambos datasets.

## Qué tiempo de giro usar (a definir con Patricio)

Dos opciones:

- **2π/U** (L = tamaño de caja). Es lo de mis notas de reunión y lo que usa Angriman 2020 (Tabla I, DNS: L = 2π, f₀ = 1/2π).
- **L/U con L del espectro**, `L = ∫E(k)/k dk / ∫E(k) dk`. Es lo que calcula `inspect_energetics.py`.

Valores (promedio sobre la segunda mitad de `balance.txt`, últimos 5 espectros):

| | TG768 | RND512 | 768 / 512 |
|---|---|---|---|
| U = sqrt(<v²>) | 1.352 | 1.000 | |
| 2π/U | 4.65 | 6.29 | **0.74** |
| L_espectro | 0.328 | 0.571 | |
| L_espectro/U | 0.243 | 0.571 | **0.43** |

Las dos opciones dan una relación muy distinta entre datasets (el tiempo de giro de 512 es 1.35× o 2.3× el de 768), así que la elección cambia bastante los lags en pasos para las 512. Con 2π la longitud es la misma en ambas y solo cambia U; la L espectral refleja que los flujos son distintos.

**Pregunta para Patricio:** ¿2π/U (como Angriman 2020) o L/U espectral (como `inspect_energetics`)?

## Plan

1. Replicar 002 y verificar que el dataset sea idéntico.
2. Agregar al pipeline lags en unidades del tiempo de giro (t\* = t / T_L), convirtiendo a pasos con el Δt y T_L de cada simulación.
3. Verificar que con TG768 el esquema nuevo reproduzca los lags de 002.
4. Sumar las RND512.

## Referencias

- `docs/Angriman et al. - 2020 ...pdf`: Tabla I, definición de U, L, f₀ para la DNS TG768.
- `docs/Español et al. - 2025 ...pdf`: sección 2, setup de las TG768 (caja (2πL₀)³, forzado TG, 10⁶ partículas).
- GHOST: `src/pseudo/pseudospec3D_hd.f90` (`hdcheck`, `energy`), `src/pseudo/pseudospec3D_mod.f90` (`boxparams`: `Lx` en unidades de 2π).
