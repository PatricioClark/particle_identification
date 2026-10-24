# Bitácora

Registro de lo que fui haciendo, explicado para alguien que no está en tema.

**El proyecto en una frase:** tenemos simulaciones de partículas moviéndose en un fluido turbulento. Cada simulación usa un "modelo" distinto de cómo se mueve la partícula (una ecuación física distinta) y un tamaño/peso de partícula distinto (el *número de Stokes*). Queremos entrenar un clasificador que, mirando solo las trayectorias de las partículas, adivine qué modelo y qué Stokes las generó. La idea a futuro es aplicarlo a datos de laboratorio, donde no se sabe qué modelo describe mejor a las partículas reales.

**Cómo funciona el pipeline:**

1. **Armar el dataset** (`build_dataset.py`): de cada simulación se toman grupos de 5000 partículas y se calculan 24 números que resumen cómo se mueven (las *features*). Cada grupo es una muestra.
2. **Entrenar el clasificador** (`train_classifier.py`): aprende a reconocer el modelo a partir de esas 24 features.

Las corridas están en `runs/` (cada carpeta, 001, 002, …, es un experimento). Patricio hizo de 001 a 004; yo arranqué en 005.

---

## 2026-09-27 — ¿Qué simulaciones usó realmente el experimento 002?

**Contexto:** 002 es el experimento de Patricio del que parto. Armó un dataset y entrenó un clasificador que acierta el **97%** de las veces.

**Lo que encontré:** la lista de simulaciones de 002 tiene 12, pero **el experimento usó solo 11**. Faltó la de **ONLD con Stokes 136** porque la carpeta con sus datos no existe en el disco: solo hay una llamada `St136_wrong_taup` ("tau_p equivocado"), o sea, una versión que se corrió con un parámetro mal puesto. El programa saltea en silencio las simulaciones que no encuentra, así que nadie se enteró.

**Cómo lo comprobé:**

- El registro de la corrida dice `Accessible simulations: 11` (no 12).
- No hay ningún archivo de resultados para esa simulación.
- El clasificador reporta 11 clases, sin ONLD 136.

**Qué significa:** el 97% de acierto es sobre un problema de **11 clases**. El resultado es válido, solo que esa simulación nunca participó. Tampoco están las simulaciones BB: Patricio las sacó a propósito.

**Las 11 simulaciones de 002:**

| Modelo | Qué es | Números de Stokes |
|---|---|---|
| LAG | trazadores (partículas sin peso que siguen al fluido) | 0 |
| MR | Maxey-Riley (arrastre + término de aceleración) | 0.76, 3.2, 8.89 |
| NLD | arrastre no lineal + término de aceleración | 8.89, 312 |
| ONLD | solo arrastre no lineal (sin término de aceleración) | 3.2, 8.89, 35.6, 312 |
| FAX | Maxey-Riley + corrección de Faxén | 8.89 |

**Pendiente:** preguntar a Patricio si existe una versión corregida de ONLD 136. La entrada rota está en todas las listas de simulaciones del repo.

---

## 2026-09-27 — ¿Cómo eligió Patricio los "lags"? (experimento 005, parte 1)

**Contexto:** varias features miden *cuánto cambia la partícula después de un cierto tiempo*. Ese tiempo de espera se llama **lag**. En 002 se usaron 5 lags: 0.016, 0.08, 0.21, 0.72 y 2.0 (en unidades de tiempo de la simulación). Quería entender por qué esos.

**Qué hice:** corrí `inspect_signals.py`, que grafica cuatro medidas del movimiento de las partículas en función del lag, para las 11 simulaciones. Marqué en rojo los 5 lags de 002. Gráfico: `runs/005/inspect_signals.pdf`.

**Qué se ve:**

- **Los modelos se separan sobre todo con lags cortos** (de 0.016 a 0.2). Con lags largos (más de 2–3) todas las curvas se juntan: las partículas ya "se olvidaron" de cómo se movían y no hay información para distinguirlas.
- **Los 5 lags de 002 caen justo en la zona útil.** La elección tiene sentido.
- **Hay dos grupos.** Las ONLD se distinguen muy fácil del resto. En cambio, LAG, MR, NLD y FAX dan curvas casi encimadas. Coincide con el clasificador de 002: acierta 100% en las ONLD y se equivoca algo con LAG, MR y NLD. **El desafío real es separar ese segundo grupo.**

**Lo que el gráfico no responde:** *cuántos* lags conviene usar. Para eso habría que entrenar el clasificador con distintas cantidades y comparar.

**Antecedente:** Patricio hizo un estudio parecido en 001, con 5 simulaciones. El gráfico no se guardó, pero después de eso pasó de lags muy cortos (hasta 0.26) a lags hasta 2.0, lo que es consistente con lo que vemos.

---

## 2026-09-28 — Réplica del dataset de 002 (experimento 005, parte 2)

**Objetivo:** antes de cambiar nada, verificar que puedo reproducir exactamente el dataset de Patricio. Si no, no podría comparar mis resultados con los suyos.

**Qué hice:** corrí `build_dataset.py` con el mismo código, la misma lista de simulaciones y los mismos parámetros que 002. Después comparé los dos datasets número por número con `runs/005/compare.py`.

**Resultado: los datasets son iguales.**

- Mismas simulaciones, mismas clases, mismas 1100 muestras, mismas partículas elegidas.
- De los 26 400 números del dataset, 319 difieren, pero **solo en el último dígito** (diferencias de 1 en 10¹⁶). Eso es error de redondeo de la computadora: esta vez corrió en otra máquina del cluster y probablemente con otra versión de las librerías. Si algo hubiera cambiado de verdad (otras partículas, otros tiempos), las diferencias serían millones de veces más grandes.

**Conclusión:** tengo un punto de partida confirmado. Puedo empezar a modificar cosas sabiendo que, si los resultados cambian, es por lo que yo cambié.

---

## 2026-09-30 — Medir los tiempos en "tiempos de giro" (experimento 006, parte 1)

**Contexto:** quiero sumar las simulaciones de resolución 512. El problema es que son de otro flujo: cada paso de tiempo dura distinto (0.030 en vez de 0.016) y el flujo se mueve a otra velocidad. Entonces "esperar 125 pasos" no significa lo mismo en las 768 que en las 512. La solución es medir el tiempo en una unidad propia de cada flujo: el **tiempo de giro** T, lo que tarda un remolino grande en dar una vuelta (T = 2π/U, donde U es la velocidad típica del fluido). A eso se le dice **adimensionalizar**.

**La prueba que sugirió Patricio:** como todas las 768 son del mismo flujo, si paso los lags de 002 a "tiempos de giro" y después los vuelvo a pasar a pasos, tengo que recuperar los mismos lags. Si no, la cuenta está mal hecha.

**Qué hice:** el script `runs/006/lags_adim.py` calcula T para cada simulación a partir de su `balance.txt`, convierte los lags y muestra el resultado. Corre en segundos, sin mandar un job.

**Resultado: la prueba sale bien.**

- El tiempo de giro de las 768 es casi igual en todas: **T ≈ 4.61** (varía ~1% entre simulaciones).
- Los lags de 002 en tiempos de giro son: 0.0035, 0.017, 0.045, 0.156 y 0.43. O sea, el lag más largo de 002 es menos de medio tiempo de giro.
- Al volver a pasarlos a pasos, **las 13 simulaciones 768 recuperan `[1, 5, 13, 45, 125]`**. En tres de ellas el último da 124 o 126, por esa pequeña variación de T.

**Qué pasa con las 512 (vista previa):** su tiempo de giro es mayor (**T ≈ 6.37**), pero cada paso también dura más. En conjunto, los mismos lags en tiempos de giro dan **menos pasos**:

| | lags en pasos | ventana |
|---|---|---|
| 768 | 1, 5, 13, 45, 125 | 251 |
| 512 | 1, 4, 10, 33, 92 | 185 |

**Ojo:** estos números de las 512 dependen de definir el tiempo de giro como 2π/U. Hay otra definición posible (L/U, con L calculada del espectro) que da valores bastante distintos. Hay que confirmar con Patricio cuál usar (ver `notes/adimensionalizacion.md`).

**Además:** en la lista de simulaciones de 006 dejé comentada ONLD 136, porque sus datos no existen. Quedan 22 simulaciones: 13 de 768 (incluidas las dos BB) y 9 de 512.

---

## 2026-09-30 — ¿Cuál es la forma correcta de medir el tiempo de giro? (experimento 006, parte 2)

**Contexto:** hay varias formas de calcular el tiempo de giro, y para comparar 768 con 512 dan resultados distintos. Para decidir con datos, armé una **prueba de colapso**: si una forma de medir el tiempo es la correcta, partículas que se comportan igual en los dos flujos tienen que dar curvas superpuestas al graficarlas en esas unidades. Las candidatas:

- **2π/U**: tamaño de la caja dividido por la velocidad del fluido.
- **L/U**: tamaño típico de los remolinos (calculado del espectro) dividido por la velocidad.
- **τ_L**: el tiempo en que la velocidad de los trazadores "se olvida" de su valor inicial (lo que usa Angriman 2020).
- **τ_η**: el tiempo de Kolmogorov, la escala de los remolinos más chicos.

**Problema encontrado en los datos:** la idea era comparar los trazadores (LAG) de los dos flujos, porque son las partículas más simples: solo siguen al fluido. Pero **los trazadores de la simulación 512 tienen las posiciones mal guardadas**: avanzan unas **81 veces menos** de lo que indica su velocidad. Lo comprobé comparando cuánto se mueve cada partícula entre un archivo y el siguiente con la velocidad que tiene guardada:

| Simulación | ¿Posición y velocidad coinciden? |
|---|---|
| Las 13 simulaciones de 768 (LAG, MR, NLD, ONLD, BB, FAX) | sí |
| Partículas inerciales 768 (corrida 003) | sí |
| Las 8 de 512 que no son trazadores (4 HPP y 4 ONLD) | sí |
| **Trazadores 512** | **no: se mueven 81.4 veces menos** |

Lo revisé en todas las simulaciones que existen: **el problema está solamente en los trazadores de 512.**

La velocidad guardada está bien (coincide con la del fluido); lo que falla son las posiciones. El factor 81.4 es casi exactamente 512/2π, así que parece un error de unidades al guardar o integrar las posiciones de los trazadores en esa simulación. **Consecuencia:** esos trazadores casi no se mueven, así que sus estadísticas no representan a una partícula que viaja con el fluido. Los saqué de la lista de 006 hasta aclararlo con Patricio.

**Qué hice entonces:** la prueba compara los trazadores de 768 con las partículas **más livianas de 512** (HPP y ONLD con Stokes 0.1), que siguen al fluido casi como trazadores. Script: `runs/006/collapse_test.py`, job `job_collapse.sh` (SLURM 9429).

**Resultado: la peor forma es 2π/U; las mejores son L/U y τ_η.** Gráfico: `runs/006/collapse_test.pdf` (una fila por forma de medir el tiempo; si la forma es buena, las curvas de los dos flujos quedan encimadas).

Cuánto se separan las curvas de los dos flujos con cada forma (0 = curvas idénticas; cuanto más grande, más separadas):

| Forma de medir el tiempo | VACF | S2 | S4 | MSD | ¿Se superponen? |
|---|---|---|---|---|---|
| 2π/U | 0.101 | 0.292 | 0.588 | 0.064 | **no** |
| **L/U** | **0.009** | 0.057 | 0.176 | **0.008** | **sí, muy bien** |
| τ_L | 0.034 | 0.049 | 0.109 | 0.017 | bastante (solo se pudo calcular para una de las dos partículas) |
| **τ_η** | 0.019 | **0.038** | 0.128 | 0.010 | **sí, muy bien** |

**Qué significa:**

- **2π/U no sirve para comparar 768 con 512.** Supone que los remolinos grandes miden lo mismo en los dos flujos (el tamaño de la caja), pero el espectro muestra que en 512 son más grandes.
- **L/U y τ_η funcionan casi igual de bien.** Las dos tienen en cuenta cómo es realmente cada flujo. Según ellas, el tiempo de giro de 512 es unas 2.5–2.8 veces el de 768 (con 2π/U daba solo 1.4).
- **Ninguna superpone del todo S4 en los lags más cortos.** Probablemente porque las partículas de Stokes 0.1 tienen un poquito de inercia que suaviza los cambios bruscos de velocidad, y S4 es la medida más sensible a eso.

**Mi elección: L/U**, pendiente de confirmar con Patricio. Es la que mejor superpone la VACF y la MSD, es la definición del tiempo de giro que figura en el resumen (ec. 4 y 5) y es lo que calcula `inspect_energetics.py`.

**Cómo cambian los lags de las 512** (estimación con los tiempos de esta prueba; las 768 quedan igual que en 002 con cualquier forma):

| Forma | lags 512 en pasos | ventana 512 |
|---|---|---|
| 2π/U (descartada) | 1, 4, 10, 33, 92 | 185 |
| **L/U** | **1, 7, 17, 60, 165** | **~332** |
| τ_η | 1, 7, 19, 67, 186 | ~373 |

**Limitaciones de la prueba:** es un solo par de flujos; no son trazadores puros (por el problema de los trazadores 512); y cada forma compara un rango de tiempos distinto (por ejemplo, con L/U no entran los lags más cortos).

**¿Y `inspect_energetics.py`?** Patricio me dijo que con ese script se puede calcular el tiempo característico, y es así: calcula exactamente **T = L/U**, la forma que ganó. Lo corrí en los dos flujos:

| | `inspect_energetics` | prueba de colapso |
|---|---|---|
| 768 | T = 0.243 | T = 0.235 |
| 512 | T = 0.617 | T = 0.583 |

Las pequeñas diferencias vienen de que `inspect_energetics` da el valor **de un solo instante** (el último), mientras que la prueba usa un **promedio** sobre todo el tiempo que duran las partículas. En 768 casi no importa porque el flujo es muy estable, pero en 512 la velocidad del fluido fluctúa bastante (entre 0.75 y 1.28), así que el valor de un instante puede cambiar ~25%. **Para elegir los lags uso el promedio.** `inspect_energetics` sigue sirviendo para mirar si el flujo está estable (gráficos de energía, espectro y velocidad en el tiempo).

Dos detalles del script: falla si el último archivo de espectro está vacío (pasa en los trazadores 768), y el tiempo de Kolmogorov que muestra está mal por el factor 2 de ε (pregunta 4 para Patricio).

**Ahora quedan 21 simulaciones en 006:** 13 de 768 y 8 de 512.

---

## 2026-10-01 — Lags para todas las simulaciones, con el tiempo de giro L/U (experimento 006, parte 3)

**Qué hice:** con la forma de medir el tiempo que ganó la prueba (L/U, promediada en el tiempo), pasé los lags de 002 a "tiempos de giro" y de ahí a pasos para cada una de las 21 simulaciones. Script: `runs/006/lags_adim.py` (ahora usa L/U por defecto). Resultado completo: `runs/006/lags_LU.txt`.

**Verificación:** las 13 simulaciones de 768 recuperan los lags de 002 (`1, 5, 13, 45, 125`), con diferencias de 1 o 2 pasos en el último lag por pequeñas variaciones de L entre simulaciones. La cuenta está bien.

**Los lags para cada flujo:**

| | tiempo de giro T = L/U | lags en pasos | ventana |
|---|---|---|---|
| 768 | 0.240 | 1, 5, 13, 45, 125 | 251 |
| 512 | 0.531 | 1, 6, 15, 53, 148 | 296 |

En tiempos de giro, los lags son 0.067, 0.33, 0.87, 3.0 y 8.3, y la ventana dura 16.7 tiempos de giro. En los dos flujos los lags representan **el mismo tiempo físico relativo**, aunque en pasos sean distintos.

**Un detalle:** en 512 el tamaño de los remolinos (L) cambia ~10% a lo largo de la simulación (igual que la velocidad del fluido). Por eso uso promedios sobre todo el tiempo que duran las partículas y no el valor de un instante. Una estimación anterior con solo los últimos espectros daba lags algo más largos (1, 7, 17, 60, 165).

**Falta:** comprobar con `inspect_signals` que estos lags también caen en la zona útil para las simulaciones que no estaban en 002 (BB, HPP y las ONLD de 512).

---

## 2026-10-01 — ¿Los lags sirven para todas las simulaciones? (experimento 006, parte 4)

**Contexto:** los lags de 002 se eligieron mirando solo las 11 simulaciones de ese experimento. En 006 hay simulaciones nuevas (las BB, las HPP y las ONLD de 512), así que había que comprobar que esos mismos lags (ya pasados a tiempos de giro) también caen donde se separan sus curvas.

**Qué hice:** el mismo estudio de `inspect_signals`, pero con las 21 simulaciones y todo medido en unidades de cada flujo: el tiempo en tiempos de giro (t/T, con T = L/U) y las amplitudes divididas por la velocidad (S2/U², S4/U⁴) o el tamaño de los remolinos (MSD/L²). Así las curvas de 768 y 512 se pueden poner en el mismo gráfico. Script: `runs/006/inspect_signals_adim.py`, job SLURM 9434. Gráfico: `runs/006/inspect_signals_adim.pdf` (línea llena = 768, discontinua = 512, rojo = los 5 lags, negro = la ventana).

**Resultado: los 5 lags también sirven para las simulaciones nuevas. No hace falta cambiarlos.**

- **Los lags caen en la zona útil en los dos flujos.** En S2 y S4 las curvas se separan sobre todo entre 0.07 y 3 tiempos de giro, y ahí caen 4 de los 5 lags (0.067, 0.33, 0.87 y 3.0). El último (8.3) queda donde las partículas ya casi "se olvidaron" de su velocidad, y la ventana (16.7) en la zona donde todas las curvas se juntan. Es la misma lógica que en 002.
- **Los dos flujos quedan en la misma escala.** Las curvas de 512 se ubican ordenadas entre las de 768 según su número de Stokes: las partículas más livianas de 512 (Stokes 0.1) quedan cerca de los trazadores de 768, y las ONLD pesadas de 512 (Stokes 20 y 100) mezcladas con las ONLD pesadas de 768. **La adimensionalización con L/U funciona para todas las especies**, no solo para las de la prueba de colapso.

**Dos cosas a tener en cuenta para el clasificador:**

1. **En 512, HPP y ONLD con el mismo Stokes (0.1 y 1) dan curvas prácticamente iguales.** Tiene sentido físico: con partículas tan livianas, el arrastre no lineal (ONLD) casi no se diferencia del lineal (HPP). Es probable que el clasificador las confunda, y no por un error.
2. **Las BB quedan encimadas con las MR**, como ya pasaba con NLD y FAX en 002. Se suman al grupo difícil de separar.

**Un límite de los datos:** el lag más corto (0.067 tiempos de giro) es exactamente un paso entre archivos en 768. No se pueden usar lags más cortos porque no hay datos más seguidos.

**Pendiente menor:** en el gráfico cuesta distinguir los colores (HPP y ONLD se confunden). Para mostrárselo a Patricio convendría separarlo en dos columnas (768 y 512) con colores más distintos. No cambia las conclusiones.

---

## 2026-10-01 — Modificar el pipeline para usar tiempos de giro (experimento 006, parte 5)

**Contexto:** el programa que arma el dataset (`build_dataset.py`) usaba **una sola lista de lags en pasos** para todas las simulaciones. Con dos flujos distintos eso ya no sirve: cada flujo necesita sus propios lags en pasos para que representen el mismo tiempo físico. Además, algunas features tienen unidades (por ejemplo, dependen de la velocidad del fluido), y sin normalizarlas el clasificador podría distinguir los flujos en lugar de los modelos de partícula.

**Qué cambié** (en `runs/006`, sin tocar el código de la raíz):

- **Lags y ventana en tiempos de giro.** Ahora se le pasan los lags y la ventana en tiempos de giro, y el programa los convierte a pasos **para cada flujo** (768 y 512), usando un tiempo de giro por flujo (el promedio de sus simulaciones). Así las 768 dan exactamente los lags de 002 (`1, 5, 13, 45, 125`, ventana 251) y las 512 dan `1, 6, 15, 53, 148` (ventana 296).
- **Features normalizadas.** Las que tienen unidades se dividen por las escalas de cada flujo (por ejemplo, S2 por U²). Se puede desactivar.
- **Los parámetros de cada flujo se calculan una vez y se guardan en un archivo** (`flow_params_002.json` para la verificación), porque leer los espectros es lento. El job solo lee ese archivo.

**Chequeo rápido:** una feature que tiene que dar ~1/3 después de normalizar (la varianza de la velocidad dividida por U²) dio 0.343 en 768 y 0.342 en 512. La normalización funciona.

**Verificación en curso:** corro el programa nuevo con las simulaciones de 002, sus mismos parámetros y **sin normalizar**. Tiene que dar el mismo dataset que 005; si da igual, el cambio no rompió nada. Job `runs/006/job_verif002.sh` (SLURM 9572).

**Resultado: idéntico.** El dataset nuevo es **exactamente igual** al de 005, número por número (esta vez ni siquiera hay diferencias de redondeo): mismas 11 clases, mismas 1100 muestras, mismas partículas y los 55 checkpoints iguales. El único cambio son los nombres de las clases, que ahora incluyen el flujo (`MR_St3.2_TG768` en vez de `MR_St3.2`). **El pipeline nuevo funciona: con lags en tiempos de giro reproduce exactamente lo que hacía el original.**

**Tardó bastante (~1 hora y media o más)** porque el disco de las simulaciones 768 estaba lento (~9 MB/s por proceso). Cada simulación se divide en 5 partes y **cada parte vuelve a leer la trayectoria completa**. Para el dataset de 006 conviene usar una sola parte por simulación con más muestras (`--n-shards-per-sim 1 --batches-per-shard 100`): da la misma cantidad de muestras y lee 5 veces menos datos.

### Problema técnico del cluster: MPI falla en el nodo `sakura`

Los dos primeros intentos de la verificación fallaron apenas arrancaron, con el error `MPI_ERR_OTHER`. Lo que pasaba:

- El programa usa 20 procesos que tienen que **comunicarse entre sí** para coordinarse (eso es MPI).
- SLURM mandó esos jobs a **`sakura`**, y en `sakura`, **dentro de un job de SLURM**, la forma de comunicación que MPI usa por defecto (llamada UCX) **se cuelga o falla**. Desde la terminal, fuera de SLURM, funciona bien; por eso no aparecía en las pruebas rápidas.
- Los jobs anteriores no fallaron porque **nunca habían caído en `sakura`** (corrieron en `g1`, `g2`, `a1`, `a2`).

**Solución:** agregarle a `mpirun` la opción **`--mca pml ob1`**, que usa otra forma de comunicación. Lo probé con un job de 20 procesos en `sakura` y funciona. **Conviene usarla en todos los jobs con MPI**, así no importa a qué nodo los mande SLURM:

```bash
mpirun --use-hwthread-cpus --bind-to hwthread --mca pml ob1 -np 20 python ...
```

**Pendiente menor:** `predict.py` todavía no funciona con datasets de lags por flujo (reconstruye los lags a partir de una sola ventana). No hace falta por ahora.

---

## 2026-10-02 — Dataset de 006 armado (experimento 006, parte 6)

**Qué hice:** armé el dataset con las 21 simulaciones (13 de 768 y 8 de 512), con los lags en tiempos de giro y las features normalizadas. Una parte por simulación con 100 muestras cada una. Job `runs/006/job_dataset.sh` (SLURM 9625, nodo `a2`). Archivo: `runs/006/dataset.pkl` (no está en GitHub, como los demás datasets).

**Resultado:** **2100 muestras** (100 por simulación), **21 clases**, 24 features, sin valores faltantes. Lags usados: 768 → `1, 5, 13, 45, 125` (ventana 251); 512 → `1, 6, 15, 53, 148` (ventana 296).

**Chequeo de la normalización:** la varianza de la velocidad dividida por U² da ~1/3 para las partículas livianas de los dos flujos (0.33 trazadores 768, 0.34 HPP y ONLD de Stokes 0.1 en 512) y menos para las muy pesadas (0.22 en ONLD de Stokes 100), que es lo esperable: una partícula pesada suaviza las fluctuaciones de velocidad del fluido.

**Tardó ~2 horas.** Las 768 terminaron en ~1 hora; las 512 tardaron más porque sus trayectorias son más largas (3034 pasos, ~36 GB de lectura cada una).

---

## 2026-10-02 — Clasificador de 006: con y sin BB (experimento 006, parte 7)

**Qué hice:** entrené el clasificador (el mismo de 002: Random Forest, validación cruzada de 5 partes) sobre el dataset de 006 en dos versiones: **con las 21 clases** y **sin las BB** (19 clases; las saqué del dataset con `runs/006/filter_classes.py`, sin recalcular nada). Se corre directo en `sakura`, en un par de minutos. Resultados: `runs/006/train_conBB.log`, `runs/006/train_sinBB.log` y las matrices de confusión `runs/006/model_conBB.confusion.png` y `runs/006/model_sinBB.confusion.png`.

**Resultado: acierta el 98% de las veces en las dos versiones.**

| | clases | muestras | errores | accuracy |
|---|---|---|---|---|
| 002 (Patricio, solo 768) | 11 | 1100 | ~3% | 97% |
| 006 con BB | 21 | 2100 | 46 (2.2%) | **98%** |
| 006 sin BB | 19 | 1900 | 40 (2.1%) | **98%** |

Es decir: aun con el doble de clases y dos flujos distintos, el clasificador anda tan bien como en 002.

**Qué se confunde (versión con BB):**

| Veces | Clase real → predicha como |
|---|---|
| 7 + 5 | NLD 8.89 ↔ MR 3.2 (768) |
| 7 + 4 | HPP 0.1 ↔ ONLD 0.1 (512) |
| 6 + 6 | trazadores (LAG) ↔ BB 8.89 (768) |
| 2 + 2 | trazadores ↔ MR 0.76 (768) |
| resto | 1 o 2 casos sueltos |

- **Son las confusiones que esperábamos** por los gráficos de `inspect_signals`: partículas livianas que se mueven casi como el fluido (trazadores, MR 0.76, BB 8.89) y modelos con curvas casi encimadas (NLD 8.89 y MR 3.2; HPP y ONLD de Stokes 0.1, que físicamente casi no se diferencian).
- **Ninguna confusión entre flujos:** nunca se predice una clase de 768 como de 512 ni al revés.
- **Las BB casi no cambian el resultado.** BB 35.6 se reconoce perfecto; BB 8.89 se confunde con los trazadores. Sin las BB, los trazadores pasan a confundirse un poco más con MR 0.76. Las BB no "rompen" el clasificador, así que **no hay un motivo de rendimiento para sacarlas**.
- **Las features más útiles** son las de lags cortos (S2 y S4 en los primeros lags) y la varianza de la aceleración, igual que en 002 y coherente con los gráficos.

**Ojo con una interpretación:** que no haya confusiones entre flujos **no prueba** que el clasificador sea independiente del flujo. Las clases de 768 y 512 son distintas (otros modelos u otros Stokes), así que no tendría por qué confundirlas. Para medir si realmente "no depende del flujo" (la idea final del proyecto, para aplicarlo a datos de laboratorio) habría que entrenar con un flujo y probar con el otro, usando clases que estén en los dos. Hoy eso no se puede hacer bien: los trazadores de 512 tienen el problema de las posiciones y las ONLD tienen Stokes distintos en cada flujo.

---

## 2026-10-22 — Corridas nuevas de GHOST en 512 (pedido de Patricio)

**La preocupación de Patricio:** el clasificador anda **demasiado** bien (98%). El miedo es que esté reconociendo **la simulación** y no **el modelo de partícula**. Pasa porque cada clase es una sola simulación y la validación cruzada siempre prueba con muestras de simulaciones que ya vio al entrenar. Un clasificador que reconociera "la corrida" por cualquier detalle sacaría el mismo 98%.

**Lo que pidió:** dos simulaciones nuevas del flujo 512 (forzado aleatorio):

1. **El mismo ejemplo con otra semilla.** Si el clasificador reconoce las clases en un flujo que nunca vio, está mirando la física y no la simulación.
2. **El mismo modelo físico en otro flujo:** ONLD con Stokes ≈ 3.2 en 512, para comparar con el ONLD 3.2 de 768. Es el primer par con el mismo modelo y el mismo Stokes en los dos flujos.

**Cómo lo armé:** las dos cosas van en **una sola corrida**, en `/share/data4/msvidal/RND512_nuevas/` (fuera del repo, por el tamaño).

- Arranca del último estado guardado de la corrida de bespanol (t = 90.75), **con otra semilla del forzado** (2207 en lugar de 1105). El flujo arranca turbulento y se va separando del original.
- Usa el mismo programa de bespanol, que corre 9 especies de partículas a la vez. Ocho son las mismas de su corrida (HPP 0.1, 0.6, 1, 4 y ONLD 0.1, 1, 20, 100) y siguen desde donde él las dejó.
- En el lugar de los trazadores (que tenían el problema de las posiciones) puse la **ONLD de Stokes 3.2** (`tau = 0.3766`), que arranca de las posiciones y velocidades de HPP 4 (mismo tau).
- Para el Stokes usé el τ_η real de 512 (0.118, ver la pregunta 7).

**¿Por qué cambiar la semilla del forzado y no la de las partículas?** Con la misma semilla el flujo sería el mismo, y si el clasificador reconoce el flujo la prueba no lo detectaría. Además, la variación de partículas dentro de un flujo ya la mide la validación cruzada (y la prueba de control de abajo). Y reubicar partículas al azar obliga a esperar que las pesadas se acomoden al flujo, lo que para ONLD 100 lleva más que toda la corrida.

**Cuidados para cuando la analicemos:**

- Al principio el flujo nuevo es casi igual al de bespanol. Hay que **descartar las primeras ~50 salidas** (lo mismo cubre el ajuste de la ONLD 3.2, que arrancó de un estado que no es el suyo).
- Va a tener ~360 salidas: alcanza para la ventana de 296, pero casi sin margen para moverla. Bespanol juntó 3034 salidas en ~12 jobs (un mes).
- La energía del flujo nuevo está algo baja (<v²> ≈ 0.72 contra 0.97 de la corrida original), dentro de las variaciones normales de su corrida. Conviene **normalizar con sus propias escalas** (U, L, T de su `balance.txt` y sus espectros).
- El clasificador de 006 no puede contestar la pregunta 2 tal como está, porque sus etiquetas incluyen el flujo ("ONLD_St3.2_TG768"). Para eso habría que entrenar con etiquetas de solo modelo + Stokes. Y aun con el mismo Stokes no es "la misma partícula" en todo sentido: τ_η/T vale 0.18 en 768 y 0.22 en 512.

**Estado:** prueba de 25 minutos (job 11685) bien: las posiciones avanzan de acuerdo con la velocidad en las 9 especies, sin valores raros. Corrida definitiva: **job 11687**, 64 procesos en `a1` (no había lugar para los 94 de bespanol), ~8 minutos por salida (bespanol: ~7 con 94). Arrancó el 22/10 a las 22:05 y termina el 24/10 cerca de las 21:30. Para ver cómo va: `/share/data4/msvidal/RND512_nuevas/estado.sh`.

---

## 2026-10-23 — ¿El clasificador reconoce la simulación o el modelo? (chequeos con 006)

Mientras corre GHOST, revisé el problema con los datos que ya teníamos. Scripts: `runs/006/check_flow_vs_model.py` y `runs/006/null_test.py`. Resultados: `runs/006/check_flow_vs_model.log` y `runs/006/null/null_test.log`.

**1. Las 8 clases de 512 comparten el mismo flujo.** Ahí no hay "simulación" que reconocer: si las separa, es por las partículas. Entrenando solo con ellas acierta el **98.6%**. Sacando las dos features más sensibles a la energía del flujo (`vel_variance` y `acc_variance`) acierta **98.4%**, o sea que no depende de ellas.

**2. Separa pares que casi no difieren.** HPP y ONLD de Stokes 0.1 son casi iguales (a ese Stokes el arrastre no lineal casi no actúa) y aun así los separa en un 95%. La diferencia está en la aceleración (2%) y en S4 a lags cortos (7–12%), en el sentido que espera la física: el arrastre no lineal frena un poco más y la partícula sigue más de cerca al fluido. Lo detecta porque cada muestra promedia 5000 partículas y tiene muy poco ruido.

**3. Prueba de control: ¿reconoce el grupo de partículas?** Las 100 muestras de cada clase salen de un mismo grupo de 50 000 partículas, y comparten sus fluctuaciones. Para descartar que eso infle el resultado, armé para HPP 0.1 y ONLD 0.1 dos grupos de 50 000 partículas que no se pisan (A y B), con el mismo procedimiento que 006 (job `runs/006/job_null.sh`, nodo `g2`, ~20 minutos).

| prueba | acierto |
|---|---|
| misma especie, grupo A contra grupo B (azar = 50%) | **50%** (HPP), **57%** (ONLD) |
| HPP contra ONLD, solo grupos A (como en 006) | 95% |
| entrenar con los grupos A, probar con los B (partículas nunca vistas) | **95.5%** |

**Conclusión:** dos grupos de la misma especie **no** se distinguen, y con partículas nuevas acierta igual. El 98% **no** viene de que las muestras compartan partículas: dentro de un mismo flujo, el clasificador está viendo diferencias físicas reales, aunque sean chicas.

**Lo que esto no contesta:** en 768 cada modelo es una corrida distinta (cada una con su `balance.txt` y su T). Ahí el clasificador **sí** podría estar reconociendo detalles de cada corrida. Eso lo responde la corrida de semilla nueva.

---

## Dónde estamos y próximos pasos

**Hecho:**

- ✅ Replicamos 002 exactamente (005).
- ✅ Entendimos cómo se eligieron los lags de 002 (`inspect_signals`).
- ✅ Verificamos la adimensionalización con las 768: se recuperan los lags de 002.
- ✅ Elegimos el tiempo de giro con datos: **L/U** (prueba de colapso).
- ✅ Calculamos los lags para las 21 simulaciones: 768 → `1, 5, 13, 45, 125`; 512 → `1, 6, 15, 53, 148`.
- ✅ Comprobamos que esos lags sirven también para las simulaciones nuevas (BB, HPP, ONLD de 512).
- ✅ Modificamos el pipeline: lags y ventana por flujo en tiempos de giro, y features normalizadas.
- ✅ Verificamos el pipeline nuevo: con las simulaciones de 002 da exactamente el mismo dataset.
- ✅ Armamos el dataset de 006: 21 simulaciones, 2100 muestras, features normalizadas.
- ✅ Entrenamos el clasificador de 006: 98% de acierto, con y sin BB.
- ✅ Le contamos los resultados a Patricio. Pidió dos simulaciones nuevas de 512 (otra semilla y ONLD Stokes 3.2): **corriendo** (job 11687, termina el 24/10 ~21:30).
- ✅ Dentro de un mismo flujo, el 98% es física y no filtración entre muestras (prueba de control).

**Lo que falta, en orden:**

1. **Armar un script de predicción** que tome una simulación nueva, calcule sus propias escalas (U, L, T), use los lags de su flujo en tiempos de giro y normalice las features (`predict.py` todavía no hace nada de esto).
2. **Cuando termine la corrida:** descartar las primeras ~50 salidas, armar las muestras y:
   - **prueba de semilla:** ¿las 8 especies caen en su clase de 512?
   - **ONLD 3.2 en 512:** ¿dónde cae con el clasificador de 006? Y entrenar un clasificador con etiquetas sin flujo (modelo + Stokes) para ver si la reconoce como "ONLD 3.2".
3. **Contarle a Patricio** los resultados y confirmar lo de la semilla (pregunta 8).
4. Más adelante: **cuántos lags usar** (repetir con más o menos lags y comparar, sobre todo en las clases que se confunden).

**En paralelo:** mandarle a Patricio las preguntas de abajo. Las más importantes son la 1 (los trazadores 512), la 3 (confirmar L/U) y la 5 (si incluimos las BB).

---

## Preguntas para Patricio

1. **Trazadores de la simulación 512 (`RND512 LAG`).** Sus posiciones (`xlg`) avanzan ~81.4 veces menos que lo que indica su velocidad (`vlg`); el factor es casi 512/2π. Las otras especies de esa simulación (HPP, ONLD) están bien. ¿Es un error conocido? ¿Hay una versión corregida? Mientras tanto no los uso.
2. **ONLD con Stokes 136.** Figura en todas las listas de simulaciones, pero su carpeta (`ONLD/St136/`) no existe; solo hay `St136_wrong_taup/`. ¿Existe una versión corregida o la sacamos de la lista?
3. **Definición del tiempo de giro para comparar 768 con 512.** Propongo **L/U, con L calculada del espectro** (la definición del resumen, ec. 4 y 5). Hice una prueba de colapso (`runs/006/collapse_test.pdf`): graficando los trazadores de 768 y las partículas más livianas de 512 (Stokes 0.1) en unidades de cada tiempo candidato, **2π/U no las superpone** (el tiempo de giro de 512 sale solo 1.4 veces el de 768), mientras que **L/U y τ_η (Kolmogorov) las superponen muy bien** (2.5 y 2.8 veces). L/U es la mejor en VACF y MSD; τ_η, en S2. ¿Estás de acuerdo con L/U? Limitaciones: un solo par de flujos y no son trazadores puros (por el problema de la pregunta 1).
4. **Fórmula de ε.** `inspect_energetics.py` y el resumen (ec. 3) usan ε = 2ν⟨ω²⟩. Pero si ⟨ω²⟩ es la 3ª columna de `balance.txt`, eso da el doble del ε real: con los datos, ν⟨ω²⟩ coincide con la tasa de inyección de energía (4ª columna). No afecta el tiempo de giro, pero sí η, τ_η y el Stokes calculado. ¿Lo corregimos?
5. **Simulaciones BB.** En 002 las sacaste ("saco las BB"). ¿Por qué? ¿Las incluyo en 006?
6. **Paso de tiempo de las 512.** El tiempo entre archivos `.lag` es 0.030, pero `dt × lgmult` del `parameter.inp` de `bin/` da 0.0375. Supongo que las partículas usan otro `lgmult` (en `part_parameters/`). No afecta al pipeline, que usa los tiempos de los archivos, pero quiero confirmarlo.
7. **Stokes de las etiquetas de 512.** Los `tau` de los archivos de partículas de 512 corresponden a un τ_η = 0.09416, pero el τ_η del flujo (con ε = ν⟨ω²⟩) da ≈ 0.118. Entonces el Stokes real es 0.8 veces la etiqueta: "HPP 4" es en realidad Stokes ≈ 3.2, y "ONLD 100" ≈ 80. ¿De dónde sale 0.09416? Para la ONLD 3.2 nueva usé 0.118.
8. **Semilla de la corrida nueva.** Cambiamos la semilla **del forzado** (flujo nuevo) y no la de las posiciones de las partículas (mismo flujo), porque lo que se quiere probar es si el clasificador reconoce el flujo. ¿Era eso lo que tenías en mente?
