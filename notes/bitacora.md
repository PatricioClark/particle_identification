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

## Próximos pasos

1. **Definir con Patricio cómo "adimensionalizar" los tiempos.** Las simulaciones nuevas (resolución 512) tienen otro flujo y otro paso de tiempo, así que un mismo lag no significa lo mismo físicamente que en las 768. Hay que medir los tiempos en unidades de un tiempo característico de cada flujo. Detalle en `notes/adimensionalizacion.md`.
2. Repetir el estudio de lags con las simulaciones 768 y 512 juntas, ya adimensionalizadas.
3. Armar el dataset combinado.
