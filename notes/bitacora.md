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

## Próximos pasos

1. **Repetir `inspect_signals` con las 21 simulaciones en unidades de L/U**, para comprobar que los lags elegidos (768: 1, 5, 13, 45, 125; 512: 1, 6, 15, 53, 148) también caen en la zona útil para las simulaciones nuevas (BB, HPP, ONLD de 512).
2. **Confirmar con Patricio el tiempo de giro L/U** (resultado de la prueba de colapso).
3. **Modificar el pipeline** para que los lags y la ventana se den en tiempos de giro y se conviertan a pasos en cada simulación. También normalizar las features que tienen unidades (S2, S4, MSD…), para que no dependan de la velocidad de cada flujo.
4. Armar el dataset combinado (006) y entrenar.

---

## Preguntas para Patricio

1. **Trazadores de la simulación 512 (`RND512 LAG`).** Sus posiciones (`xlg`) avanzan ~81.4 veces menos que lo que indica su velocidad (`vlg`); el factor es casi 512/2π. Las otras especies de esa simulación (HPP, ONLD) están bien. ¿Es un error conocido? ¿Hay una versión corregida? Mientras tanto no los uso.
2. **ONLD con Stokes 136.** Figura en todas las listas de simulaciones, pero su carpeta (`ONLD/St136/`) no existe; solo hay `St136_wrong_taup/`. ¿Existe una versión corregida o la sacamos de la lista?
3. **Definición del tiempo de giro para comparar 768 con 512.** Propongo **L/U, con L calculada del espectro** (la definición del resumen, ec. 4 y 5). Hice una prueba de colapso (`runs/006/collapse_test.pdf`): graficando los trazadores de 768 y las partículas más livianas de 512 (Stokes 0.1) en unidades de cada tiempo candidato, **2π/U no las superpone** (el tiempo de giro de 512 sale solo 1.4 veces el de 768), mientras que **L/U y τ_η (Kolmogorov) las superponen muy bien** (2.5 y 2.8 veces). L/U es la mejor en VACF y MSD; τ_η, en S2. ¿Estás de acuerdo con L/U? Limitaciones: un solo par de flujos y no son trazadores puros (por el problema de la pregunta 1).
4. **Fórmula de ε.** `inspect_energetics.py` y el resumen (ec. 3) usan ε = 2ν⟨ω²⟩. Pero si ⟨ω²⟩ es la 3ª columna de `balance.txt`, eso da el doble del ε real: con los datos, ν⟨ω²⟩ coincide con la tasa de inyección de energía (4ª columna). No afecta el tiempo de giro, pero sí η, τ_η y el Stokes calculado. ¿Lo corregimos?
5. **Simulaciones BB.** En 002 las sacaste ("saco las BB"). ¿Por qué? ¿Las incluyo en 006?
6. **Paso de tiempo de las 512.** El tiempo entre archivos `.lag` es 0.030, pero `dt × lgmult` del `parameter.inp` de `bin/` da 0.0375. Supongo que las partículas usan otro `lgmult` (en `part_parameters/`). No afecta al pipeline, que usa los tiempos de los archivos, pero quiero confirmarlo.
