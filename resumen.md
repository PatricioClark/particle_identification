# Turbulencia y modelos de partículas inerciales

## 1. Escalas de la turbulencia

Consideramos un flujo turbulento homogéneo e isótropo, solución de las ecuaciones de Navier-Stokes incompresibles

$$\nabla \cdot \mathbf{u} = 0, \qquad \frac{D\mathbf{u}}{Dt} \equiv \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u}, \tag{1}$$

donde $\nu$ es la viscosidad cinemática y $\rho$ la densidad del fluido. La turbulencia se caracteriza por una amplia separación de escalas: la energía se inyecta a la escala grande y es transferida, sin disipación apreciable, a través de un rango inercial de escalas cada vez más pequeñas (la cascada de Richardson-Kolmogorov) hasta llegar a las escalas donde la viscosidad finalmente disipa la energía en calor.

### 1.1. Escala integral y tiempo de giro grande

La intensidad de las fluctuaciones de velocidad se mide con la energía cinética media por unidad de masa

$$E = \frac{1}{2}\langle v^2 \rangle \tag{2}$$

y con la velocidad cuadrática media total $U = \langle v^2 \rangle^{1/2} = \sqrt{2E}$ (bajo isotropía, $U = \sqrt{3}\, u_\mathrm{rms}$ si se prefiere la velocidad cuadrática media por componente $u_\mathrm{rms} = \langle u_1^2 \rangle^{1/2}$, usada más abajo para $\mathrm{Re}_\lambda$). La tasa media de disipación de energía cinética por unidad de masa se obtiene a partir de la enstrofía:

$$\varepsilon = 2\nu\langle \omega^2 \rangle. \tag{3}$$

La escala integral $L$, la escala espacial más grande donde reside la mayor parte de la energía (a la cual ésta se inyecta), se calcula a partir del espectro de energía como el cociente de momentos espectrales

$$L = \frac{\int (E(k)/k)\, dk}{\int E(k)\, dk}. \tag{4}$$

Alternativamente, dado que $L$ es por construcción la escala más grande del sistema, puede aproximarse directamente por el tamaño de la caja de simulación, sin necesidad de calcular el espectro.

El tiempo característico de las estructuras de mayor escala (el *tiempo de giro grande*, o *large-eddy turnover time*) es

$$\tau_L = \frac{L}{U}. \tag{5}$$

La intensidad de la turbulencia se caracteriza por el número de Reynolds

$$\mathrm{Re} = \frac{UL}{\nu}. \tag{6}$$

### 1.2. Rango inercial y microescala de Taylor

En el rango inercial, $\eta \ll \ell \ll L$, las diferencias de velocidad entre puntos separados una distancia $\ell$ siguen el escaleo de Kolmogorov (K41):

$$u_\ell \sim (\varepsilon\ell)^{1/3}, \qquad \tau_\ell = \frac{\ell}{u_\ell} \sim \varepsilon^{-1/3}\ell^{2/3}. \tag{7}$$

La intensidad de la turbulencia suele cuantificarse con el número de Reynolds basado en la microescala de Taylor $\lambda$,

$$\lambda = \left(\frac{15\,\nu}{\varepsilon}\right)^{1/2} u_\mathrm{rms}, \qquad \mathrm{Re}_\lambda = \frac{u_\mathrm{rms}\,\lambda}{\nu}. \tag{8}$$

### 1.3. Escalas de Kolmogorov

En el extremo disipativo del rango inercial, la viscosidad se vuelve dominante y detiene la cascada. Las escalas de Kolmogorov se construyen dimensionalmente a partir de $\nu$ y $\varepsilon$, los únicos parámetros relevantes a esa escala:

$$\eta = \left(\frac{\nu^3}{\varepsilon}\right)^{1/4}, \qquad \tau_\eta = \left(\frac{\nu}{\varepsilon}\right)^{1/2}, \qquad u_\eta = (\nu\varepsilon)^{1/4}, \qquad k_\eta = \frac{1}{\eta}. \tag{9}$$

$\eta$ es la escala espacial por debajo de la cual la disipación viscosa domina, $\tau_\eta$ es el tiempo de giro asociado a esa escala, $u_\eta$ la velocidad característica correspondiente, y $k_\eta$ el número de onda de Kolmogorov (nótese que $\eta = u_\eta \tau_\eta$ y que $\tau_\eta$ coincide con la extrapolación de $\tau_\ell$ del rango inercial hasta $\ell = \eta$).

### 1.4. Cascada de energía y espectro

En el rango inercial no actúan ni el forzado (que inyecta energía sólo en las escalas grandes) ni la viscosidad (que sólo disipa por debajo de $\eta$): las escalas intermedias se limitan a transferir energía hacia escalas más pequeñas. En estado estacionario el flujo de energía a través de las escalas es entonces constante e igual a la tasa de inyección, que a su vez iguala a la tasa de disipación $\varepsilon$. Esta es la hipótesis central de la fenomenología de Kolmogorov: en el rango inercial $\varepsilon$ es el *único* parámetro relevante, junto con la escala considerada.

Por análisis dimensional, el espectro de energía $E(k)$ (definido de modo que $\int E(k)\, dk = E$) sólo puede depender de $\varepsilon$ y del número de onda $k$, lo que fija

$$E(k) = C_K\, \varepsilon^{2/3} k^{-5/3}, \qquad k_L \sim 1/L \ll k \ll k_\eta = 1/\eta, \tag{10}$$

con $C_K \approx 1{,}6$ una constante. La ley $k^{-5/3}$ es equivalente al escaleo $u_\ell \sim (\varepsilon\ell)^{1/3}$ del rango inercial. El ancho del rango inercial queda controlado por el número de Reynolds: combinando $\varepsilon \sim U^3/L$ con la definición de $\eta$ se obtiene $L/\eta \sim \mathrm{Re}^{3/4}$, de modo que sólo a Re alto existe una separación de escalas apreciable entre inyección y disipación. La figura 1 resume esquemáticamente los tres rangos.

**Figura 1:** Esquema del espectro de energía de la turbulencia homogénea e isótropa en escala logarítmica. La energía se inyecta en las escalas grandes ($k \lesssim k_L \sim 1/L$), se transfiere sin disipación apreciable a través del rango inercial siguiendo la ley $k^{-5/3}$, y se disipa por viscosidad más allá del número de onda de Kolmogorov $k_\eta = 1/\eta$. La extensión del rango inercial crece como $\mathrm{Re}^{3/4}$.

### 1.5. Número de Stokes

El tiempo de Kolmogorov $\tau_\eta$ es la escala temporal de referencia para caracterizar la inercia de una partícula suspendida en el flujo: el número de Stokes se define como

$$\mathrm{St} = \frac{\tau_p}{\tau_\eta}, \tag{11}$$

donde $\tau_p$ es el tiempo de respuesta de la partícula (Sección 2). $\mathrm{St} \to 0$ corresponde a una partícula que sigue el flujo como un trazador; St finito da lugar a desacoplamiento inercial, muestreo preferencial y clustering.

---

## 2. Modelos de partículas

Cada partícula inercial se caracteriza por su tiempo de respuesta o tiempo de Stokes $\tau_p = a^2/(3\nu)$ (con $a$ el radio de la partícula), que mide cuán rápido relaja su velocidad $\mathbf{v}(t)$ hacia la velocidad del fluido $\mathbf{u}(\mathbf{x}_p, t)$ visto en su posición. Los distintos modelos de arrastre difieren en qué términos de la ecuación de Maxey-Riley completa se retienen. A continuación se presentan, en orden creciente de complejidad, los modelos usados en este proyecto (etiquetas de `simuls.yaml` entre paréntesis).

En todas las simulaciones de este proyecto las partículas son neutralmente flotantes: su densidad es igual a la del fluido ($\gamma = \rho_f/\rho_p = 1$), de modo que el término de flotación se anula y los coeficientes de masa añadida / gradiente de presión se reducen a la unidad. Por eso $\gamma$ no aparece explícitamente en las ecuaciones que siguen. Lo único que se varía entre simulaciones es el radio $a$ de la partícula —y por lo tanto $\tau_p$ y St.

### 2.1. Trazador fluido (`LAG`)

En el límite $\mathrm{St} \to 0$ la partícula no tiene inercia y sigue exactamente el flujo:

$$\dot{\mathbf{x}}_p = \mathbf{v}(t) = \mathbf{u}(\mathbf{x}_p, t). \tag{12}$$

### 2.2. Partícula pesada ingenua / arrastre de Stokes lineal

El modelo más simple para una partícula pesada retiene únicamente el arrastre de Stokes lineal (Angriman et al. 2020, ec. 10):

$$\frac{d\mathbf{x}_p}{dt} = \mathbf{v}(t), \qquad \frac{d\mathbf{v}}{dt} = \frac{1}{\tau_p}\left[\mathbf{u}(\mathbf{x}_p, t) - \mathbf{v}\right]. \tag{13}$$

Es válido para partículas pequeñas y pesadas ($\rho_p \gg \rho_f$), donde el arrastre domina sobre el gradiente de presión, la masa añadida y la historia.

### 2.3. Maxey-Riley (`MR`)

Cuando la densidad de la partícula es comparable a la del fluido, deben retenerse además el término de gradiente de presión / masa añadida. Sin el término de historia (Basset-Boussinesq), la ecuación de Maxey-Riley-Gatignol para una partícula neutralmente flotante se reduce a (Español et al. 2025, ec. 2.6):

$$\frac{d\mathbf{v}}{dt} = \frac{1}{\tau_p}\left[\mathbf{u}(\mathbf{x}, t) - \mathbf{v}(t)\right] + \frac{D\mathbf{u}(\mathbf{x}, t)}{Dt}. \tag{14}$$

Esta forma —arrastre lineal más el término de aceleración del fluido $D\mathbf{u}/Dt$— es la base sobre la que se construyen los modelos de arrastre no lineal a continuación.

### 2.4. Arrastre no lineal (`NLD`, `ONLD`)

Para números de Reynolds de partícula $\mathrm{Re}_p$ moderados ($1 \lesssim \mathrm{Re}_p \lesssim 40$), el arrastre de Stokes lineal deja de ser válido y se reemplaza por una corrección tipo Schiller-Naumann. Only nonlinear drag (`ONLD`; Español et al. 2025, ec. 2.7):

$$\frac{d\mathbf{v}}{dt} = \frac{1 + 0{,}15\,\mathrm{Re}_p^{0{,}687}}{\tau_p}\left[\mathbf{u}(\mathbf{x}, t) - \mathbf{v}(t)\right], \qquad \mathrm{Re}_p = \left(\frac{18\,\tau_p}{\nu}\right)^{1/2}|\mathbf{u}(\mathbf{x}, t) - \mathbf{v}(t)|. \tag{15}$$

`NLD` agrega a esta misma corrección de arrastre no lineal el término de aceleración del fluido $D\mathbf{u}/Dt$ de la ecuación (14), es decir, reemplaza el arrastre lineal de esa ecuación por el arrastre no lineal de arriba. En el segundo set de simulaciones utilizado en este proyecto `NLD` y `ONLD` coinciden (sin término de aceleración); en el primer set, `NLD` incluía ese término extra.

### 2.5. Término de historia de Basset-Boussinesq (`BB`)

La forma más completa de la ecuación de Maxey-Riley agrega, a los términos anteriores, la fuerza de historia: la respuesta viscosa retardada del fluido a los cambios pasados de la velocidad relativa $\mathbf{v} - \mathbf{u}$. Siguiendo la formulación de fuerza hidrodinámica de Maxey & Riley (1983) (ver también Bec et al. 2024, ec. 3),

$$\frac{4}{3}\pi a^3 \rho_p \frac{d\mathbf{v}}{dt} = \frac{4}{3}\pi a^3 \rho_f \frac{D\mathbf{u}}{Dt} - 6\pi\nu\rho_f a\left[\mathbf{v} - \mathbf{u}(\mathbf{x}_p, t)\right] - \frac{4}{3}\pi a^3(\rho_p - \rho_f)\,\mathbf{g}$$

$$- \frac{2}{3}\pi a^3 \rho_f \frac{d}{dt}\left[\mathbf{v} - \mathbf{u}(\mathbf{x}_p, t)\right] - 6a^2\rho_f\sqrt{\pi\nu}\int_{-\infty}^{t} \frac{1}{\sqrt{t-s}}\frac{d}{ds}\left[\mathbf{v} - \mathbf{u}(\mathbf{x}_p, s)\right]ds. \tag{16}$$

El último término, la integral de Basset-Boussinesq, depende de toda la historia pasada de la aceleración relativa partícula-fluido, pesada por un núcleo $\propto (t-s)^{-1/2}$: es una fuerza de memoria, no markoviana, que distingue a `BB` de `MR` (donde este término se descarta, como se verifica explícitamente en Español et al. 2025 que produce resultados similares al omitirlo para partículas neutralmente flotantes).

### 2.6. Corrección de Faxén (`FAX`)

Cuando el tamaño de la partícula $a$ no es despreciable frente a la escala de curvatura del flujo no perturbado, deben agregarse correcciones de Faxén: términos proporcionales al laplaciano de la velocidad del fluido, evaluados en el centro de la partícula, que corrigen tanto el arrastre de Stokes como el término de gradiente de presión. Para el arrastre en régimen de Stokes estacionario (Maxey & Riley 1983, ec. 7),

$$\mathbf{F} = 6\pi a\mu\left\{\mathbf{u}(\mathbf{x}_p, t) - \mathbf{v}(t)\right\} + \mu\pi a^3\left(\nabla^2\mathbf{u}\right)_{\mathbf{x}_p}, \tag{17}$$

donde el segundo término es la corrección de Faxén al orden más bajo. Maxey & Riley (1983) derivan además la corrección de Faxén análoga para el flujo de Stokes no estacionario, que debe añadirse consistentemente al término de masa añadida / gradiente de presión. Estas correcciones son de orden $(a/\eta)^2$ y por lo tanto solo son relevantes cuando la partícula deja de ser puntual respecto de la escala de Kolmogorov; `FAX` agrega estos términos de curvatura a la ecuación de `MR`.

### 2.7. Resumen

| Etiqueta | Términos retenidos respecto de Maxey-Riley completo |
|----------|-----------------------------------------------------|
| `LAG`  | ninguno: $\mathbf{v} = \mathbf{u}$ (trazador, $\mathrm{St} = 0$) |
| `MR`   | arrastre de Stokes + masa añadida/gradiente de presión |
| `NLD`  | arrastre no lineal + término de aceleración $D\mathbf{u}/Dt$ |
| `ONLD` | arrastre no lineal solamente |
| `BB`   | $\texttt{MR}$ + integral de historia de Basset-Boussinesq |
| `FAX`  | $\texttt{MR}$ + corrección de Faxén (curvatura, $\nabla^2\mathbf{u}$) |