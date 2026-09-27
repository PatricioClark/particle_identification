31/8  
Modelos de particulas en fluidos

* partimos del problema de stokes y dsps hay modelos mas complicados  
* particulas neutrally voyant (igual de pesadas que el agua) pero que tienen un tamaño finito  
* es dificil contrastar esos modelos vs lo que pasa en el laboratorio  
* experimento hoy: podemos seguir muchas partículas en un fluido (sus trayectorias), pero no podes medir el campo de velocidades del fluido al mismo tiempo.   
* quiero comparar: cual es el modelo de particula en el fluido que mejor se ajusta al experimento: esto es dificil porque no tenemos el campo de velocidades, que es un termino muy importante de la ecuacion.  
* Idea: hagamos mucha estadistica: porque los modelos dan distintas predicciones: cual es el modelo que mas se acerca a la estadistica de la realidad.  
* simulaciones: simulamos un flujo parecido al del experimento. Queremos ver cuanto se separan los modelos a nivel de simulaciones. Si el modelo de clasificador esta tan bien entrenado que no depende del flujo detras tan fuertemente, podemos pasarle datos experimentales y ver de que modelo de particula en flujo queda mas cerca. O descartar terminos de la ecuacion. El objetivo es ver cual es el modelo que mejor describe la realidad de una particula en un fluido atraves de analisis estadistico con herramientas de machine learning.   
* yaml con simulaciones  
* features:  
  * desviacion estandar de los desplazamientos de las particulas   
  * lags en funcion de la autocorrelacion: x(t)\*x(t+tau): se tiene que descorrelacionar  
* 1er prueba: overfitting  
  * las series temporales todas arrancaban a t=0, el algoritmo era bueno identificando a cada t=0 de las distintas simulaciones: identificaba el campo mas que las particulas.   
  * data leakage: todos los datos al mismo tiempo: muy poco descorrelacionado las simulaciones de las venian : se aprendia la distribucion de las particulas a t=0  
  * eso cambio: arranca las series temporales de las particulas en distintos tiempos: para que sea mas dificil darse cuenta de que simulacion viene.   
* las clases del problema de clasificador: el modelo de particula \+ el nro de stokes (que tan grande es la particula)  
* que es la señal que viene de la partícula en si y la señal del flujo

Archivos

* la posición de la particula a tiempo dado: si quiero reconstruir la trayectoria de la partícula 100 tengo que leer muchos archivos que tienen todas las partículas en el medio  
* corro 1 millon de particulas y armo batches de 100,000 particulas para entrenar y dsps evaluar a distintos tiempos: que sea aleatorio el momento de la simulacion en el que tomo los datos  
* 1ras pruebas con simulaciones todos con los mismos parametros fisicos  
* hay tiempos caracteristicos: el lag en la funcion de autocorrelacion. Comparar con el tiempo caracteristico de la simulacion: **adimensionalizado por el tiempo caracteristico de cada simulacion** : to do list:  escribir todo en funcion del turn over time 

Escalas que importan a nivel particulas son las escalas mas chicas: escalas de kolmogorov

* escalas grandes donde inyectamos energias   
* nu: cuanta energia le estamos inyectando al sistema  
* epsilon: es la energia que se inyecta  
* fuera del equilibrio: hay un forzado y una disipacion : hay un flujo de energia desde las escalas grandes (donde se inyecta energia) a la escala donde la viscosidad empieza a disipar  
* la escala donde la viscosidad empieza a disipar es eta  
* tiempo caracteristico asociado a esa escala es el tiempo de kolmogorov

nro de stokes: tiempo de respuesta de la particula (relacionado a que tan grande es la particula) vs el tiempo de kolmogorov

* stokes →0 la particula se updatea instantáneamente

Escala mas grandes : **Escala integral y tiempo de giro grande**

* adimensionalizar: Escala integral y tiempo de giro grande  
  * El tiempo caracterıstico de las estructuras de mayor escala (el tiempo de giro grande, o large-eddy turnover time) es τL \= L/ U . **Tengo que adimensionalizar por eso mis simulaciones: 2pi/velocidad media rms (tiempo mas lento del sistema)**  
  * la caja es periodica y el tamaño de la caja puede ser 1 o 2pi, ver como esta en las simulaciones   
  * todo en la caja tiene promedio=0, lo que estamos viendo es un campo de fluctuaciones, la velocidad rms es el cuadrado de la suma, no hay que restarle un valor medio  
  * codigo con el que se arman las simulaciones de los diferentes modelos de particulas [https://github.com/pmininni/GHOST/tree/new\_paradigm](https://github.com/pmininni/GHOST/tree/new_paradigm) (new\_paradigm : esta branch)

Archivo de output

* archivo de parametros: donde se van a guardar, la viscosidad, el paso temporal, cada cuanto se escriben las particulas (el output)  
  * el paso temporal es paso temporal\*indice\*20: 20 es lgmult en la simulacion, multiplo de 20\.   
* vip.lag : posiciones de todas las particulas en  
* **balance.txt** : energia en funcion del tiempo : tercera columna ? es el epsilon: . Vamos a usarlo para sacar el tiempo caracteristico y adimensionalizar por el tiempo caracteristico de la simulacion. Necesito sacar e**l u\_rms**  
  * **balance.txt :** segunda columna es el u cuadrado ya promediado en todo el dominio, ultima columna (tercera) es el epsilon: la inyeccion de energia.   
  * **balance.txt**: para chequear como fue hecho la simulacion.   
  *   
* spectrum: modulo de la transformada de fourier del campo de velocidades: no nos vamos a meter  
* otros archivos con los campos de velocidades: x,y,z: campo de velocidades en cada punto de la grilla: eso tampoco lo vamos a usar  
* yaml de simulaciones: tienen el path donde estan las simulaciones/archivos. En el repo hay otro yaml diferente. Hay un archivo armado que sabe leer los path de los archivos.   
  * yaml de sakura: **full\_simuls.yaml**: tiene solo las de 768  
    * que forzado y que resolucion se uso, el modelo de particula que se uso y el nro de stokes. Las particulas tienen un nombre?  
    * forcing: TG : taylor green: tipo de forzado que es parecido a los experimentos: senos y cosenos, poner varios vórtices.   
    * ya analizadas por pato: 768 (512 no).   
      * para las de 768 no adimensializo porque todas tenias la misma dimension. Cuando agreguemos las de 512 ahi si hay que adimensionalizar (512 simulacion levemente distinta a 768). Hay que ver que estemos comparando escalas temporales similares. El tiempo t=2 tenemos que poder compararlo igual en ambas simulaciones entre 768 y 512 : tienen resolucion distinta y fisicamente son ligeramente distintas.   
  * [simuls\_yaml](https://github.com/PatricioClark/particle_identification/blob/main/simuls.yaml) : tiene 768 \+ 512

* [run.sh](http://run.sh) de la home es el mismo que el experimento 4: se fijo convergencia de batch de particulas en este experimento (que tan grande tiene que ser el batch de las particulas). Los archivos .sh es para correr en el cluster: son del idioma mpi:  
  * es para correr en paralelo np 20 : 20 procesos en paralelo  
* **002/[job.sh](http://job.sh) : experimento 2 : este es del cual tenemos que partir para seguir**   
  * ml python  
  * mpirun \--use-hwthread-cpus \--bind-to hwthread \-np 20 \\  
  *     python build\_dataset.py \--max-steps 251 \--n-lags 5 \\  
  *     \--checkpoint-dir ckpt \--output dataset.pkl \\  
  *     \--batches-per-shard 20 \--n-shards-per-sim 5  
  * python eda.py  
  * python train\_classifier.py  
  * el build\_dataset.py corre en 20 procesos y el el eda y train\_classifier.py en uno solo (sin paralelo)  
  * el dataset se guarda en en **dataset.pkl,** guarda checkpoints, y standard output guarda que hizo.   
  * aca con inspect eligio los lags y que tiempos se usaron. Usar esto como ejemplo, no esta adimensializado, pero ver que si adimensionalizamos nos queden estos mismos tiempos  
  * aca lags temporales NO adimensionalizados  
* **arrancar con build\_Dataset**  
  * **que se usaron en cada caso/simulacion: 768+512**

[**Inspect**](https://github.com/PatricioClark/particle_identification/tree/main/inspect)

* analisis mas fisico del flujo detras  
* grafica la trayectoria de una particula  
* tiempo caracteristico calculado vs un grafico  
* inspect\_energetics: te da los tiempos  
* inspect\_signals: marca los lags: es lo que determina lags de cuanto tiempo usamos y cuantos lags consideramos

**Comandos del cluster \- SLURM**

* sbatch [run.sh](http://run.sh)  
  * con 20 procesos el build\_dataset puede tardar 24 horas  
  * vos podes correr como maximo 48hs en el cluster: sino se corta  
* squeue: como esta la cola  
* scancel \<job-id\>: cancelar  un job  
* **readme end sakura??**

