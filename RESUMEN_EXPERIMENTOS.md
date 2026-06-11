# Resumen de experimentos — Práctica 3 SID (MARL)

> Documento de diseño para el equipo. Explica **qué** experimentos hacemos y,
> sobre todo, **por qué** cada decisión está tomada así. Para detalle de
> ejecución (instalación, comandos, salidas) ver el `README.md`.

---

## 1. Objetivo y contexto

Comparamos dos algoritmos de aprendizaje por refuerzo multiagente sobre POGEMA,
siempre con **2 agentes**:

- **IQL** (Independent Q-Learning): cada agente aprende su Q-table individual
  `[estados x acciones]` e **ignora** la existencia del otro agente. Es el
  mismo Q-Learning tabular de la Practica 2 (FrozenLake) trasladado al entorno
  multiagente.
- **JAL-GT** (Joint-Action Learning with Game Theory): mantiene una Q-table
  sobre **acciones conjuntas** `[agentes x estados x acciones_conjuntas]` y deriva
  la politica con un *concepto de solucion* (Nash, Pareto, Welfare o Minimax).

La pregunta central es como afecta la **no-estacionariedad** a IQL: como ignora
al otro agente, desde su punto de vista el entorno cambia mientras aprende
(porque la politica del otro agente evoluciona), lo que viola la asuncion de
estacionariedad que Q-Learning necesita para converger.

---

## 2. Parametros fijos y su justificacion

| Parametro | Valor | Por que |
|---|---|---|
| N agentes | 2 | Restriccion del enunciado |
| `obs_to_state` | la del baseline | Restriccion del enunciado |
| `num_states` | 1024 | Consecuencia directa de `obs_to_state`: codifica obstaculos (16) x otros agentes (16) x objetivo (4). Es independiente del tamano del mapa porque la observacion es de radio 1 (solo el vecindario local) |
| `alpha` (learning rate) | 0.01 | **Decision debatida.** Los profesores lo fijan a 0.01 en el `config.py` (el 0.5 del constructor de JALGT nunca llega a usarse). Es un valor conservador y deliberado: la Q-table conjunta de JAL-GT tiene un target inestable (depende de la politica conjunta, que cambia en cada paso), y un alpha bajo amortigua esas oscilaciones. **No lo tocamos** porque el enunciado no pide estudiar alpha y cambiarlo introduciria una variable no controlada |
| `episodes_per_epoch` | 10 | Configuracion baseline. Da resolucion suficiente en las curvas sin generar ruido |
| `maps` | 10 | Con 10 mapas y 10 episodios/epoch, cada mapa se ve ~1 vez por epoch: equilibrio entre diversidad y frecuencia de visita |
| Seeds | 0..9 (10 seeds) | Minimo razonable para estimar media y varianza inter-seed sin depender de una semilla afortunada |

---

## 3. Densidades de obstaculos: 0.1 y 0.25

La densidad es la **probabilidad de que cada celda sea un obstaculo** (muestreo
binomial celda a celda). No quisimos elegir valores a ojo, asi que los
**verificamos empiricamente con busqueda de caminos (BFS)** sobre cientos de
mapas, midiendo dos cosas: que fraccion de mapas son resolubles, y cuanto fuerzan
los obstaculos a desviarse del camino recto (ratio de rodeo).

**Densidad baja = 0.1.** ~98% de mapas resolubles, pocos obstaculos. El conflicto
viene sobre todo de las interacciones entre agentes, no del entorno. Aisla la
coordinacion pura.

**Densidad media = 0.25.** Fue una decision debatida. El primer instinto fue
0.2, pero el analisis BFS mostro que el contraste de rodeo entre 0.1 y 0.2 era
casi imperceptible en mapas pequenos (1.06 vs 1.08), lo que arriesgaba que el
analisis no apreciara diferencias. 0.25 sube el rodeo de forma clara justo en los
tamanos donde la coordinacion importa (1.13 en 6x6, 1.21 en 10x10) y mantiene
>=84% de resolubilidad. **Descartamos 0.3** porque triplicaba los mapas
completamente muertos (del 2-5% al 7-9%), y ese ruido caeria sobre la comparativa
algoritmica principal, donde no lo queremos.

> Este analisis de resolubilidad por BFS es en si mismo material para la rubrica:
> justifica la eleccion de parametros con datos, no con numeros arbitrarios.

---

## 4. Grid de hiperparametros: epsilon y gamma

El enunciado pide estudiar `epsilon` y `gamma`. Lo hacemos con un **grid conjunto
3x3** (no secuencial), porque los dos parametros estan acoplados: el epsilon
optimo puede depender del gamma. Un grid permite ver la interaccion real y se
visualiza como heatmap.

**`gamma` in {0.90, 0.95, 0.99}.** Replica el rango que ya validamos en la
Practica 2 sobre el mismo paradigma Q-Learning, donde gamma alto fue claramente
mejor (de 62.5% a 71.6% al subir de 0.9 a 0.99). Eliminamos `gamma=0.8` porque
descuenta demasiado para nuestros mapas grandes: a 10 pasos solo queda el 11% de
la senal de la recompensa terminal, lo que impide aprender trayectorias largas.
Mantenemos `0.95` porque es el valor del baseline y sirve de ancla.

**`epsilon_max` in {0.5, 0.75, 1.0}.** Saltos uniformes de 0.25 en la zona
"sana". Debatimos incluir `0.3`, que en la Practica 2 producia un fenomeno
interesante (bimodalidad: o convergia o fracasaba segun la seed). Lo descartamos
del grid porque ya sabiamos que daria un resultado pobre y cruzarlo con los 3
valores de gamma desperdiciaria 3 runs confirmando lo predecible. La razon de
mantener todos los valores >=0.5 es que ninguno es un descarte teorico seguro.

El experimento del grid **fija el par optimo** `(epsilon*, gamma*)`, que se
traslada automaticamente a todos los experimentos siguientes.

---

## 5. Metrica de no-estacionariedad (importante)

Un punto clave que ajustamos: la no-estacionariedad **no** se mide como varianza
entre semillas. Se manifiesta como **inestabilidad dentro de un mismo run**: la
curva de recompensa oscila y el TD error no decrece de forma monotona, porque el
target se mueve mientras el otro agente cambia su politica.

Por eso registramos **dos** medidas de estabilidad distintas:
- **Varianza inter-seed**: dispersion entre las 10 semillas (mide robustez).
- **Inestabilidad intra-run**: desviacion de la recompensa en ventana movil
  dentro de un run (mide la firma de la no-estacionariedad de IQL).

La hipotesis es que IQL tendra mayor inestabilidad intra-run que JAL-GT, y que
el efecto se amplifica en mapas grandes y densos.

---

## 6. Metrica de convergencia

En vez de "media de los ultimos N epochs" (que asume que todos los runs convergen
a la vez), usamos una definicion adaptativa: el epoch de convergencia es el
primero donde la media movil (ventana 20) supera el **percentil 90** de la
recompensa del run; la "recompensa en convergencia" es la media desde ahi hasta
el final. Usamos el percentil 90 y no el maximo absoluto para no dispararnos con
un pico espurio.

---

## 7. La bateria: 7 experimentos obligatorios

Se ejecutan **en cadena**, trasladando las conclusiones (los hiperparametros del
EXP1 alimentan a los demas).

| Exp | Que hace | Hipotesis / que buscamos |
|---|---|---|
| **1** | Grid (epsilon x gamma) en JAL-GT-Pareto, 4x4 | Fija (eps*, gamma*). gamma alto necesario; epsilon alto reduce varianza inter-seed |
| **2** | Barrido de gamma en IQL con eps* | Verifica que el gamma optimo transfiere a IQL. Si difiere, usamos el mismo por *fairness* y lo documentamos |
| **3** | Par optimo vs baseline (1.0, 0.95) en 10x10 | El optimo de 4x4 escala a mapas grandes? Limitacion honesta: calibramos en 4x4 y validamos en 10x10 |
| **4** | IQL vs JAL-GT-Pareto, densidad 0.1, tamanos 4/6/10 | JAL-GT >= IQL en recompensa; IQL mas inestable (no-estacionariedad) |
| **5** | Igual que EXP4 con densidad 0.25 | La brecha entre algoritmos deberia crecer con los obstaculos |
| **6** | Nash/Pareto/Welfare/Minimax en 6x6, ambas densidades | Pareto/Welfare mejores; Nash peor (deadlocks); Minimax el peor |
| **7** | Pareto vs Minimax en 4x4 y 10x10 | El ranking de conceptos depende del tamano? |

**Por que Pareto como concepto fijo en EXP4/5:** es el default del baseline y el
mas adecuado para entornos cooperativos. Lo documentamos como tal; si el EXP6
revelara un concepto claramente superior, se puede re-confirmar.

**Por que los 4 conceptos en EXP6** (el enunciado pide "al menos dos"): tener
Minimax demostrado-malo-con-datos es exactamente el tipo de analisis
contextualizado que premia la rubrica, no es desperdicio.

---

## 8. Metricas que se extraen

- **Recompensa colectiva e individual** por epoch (la individual detecta
  asimetrias entre agentes).
- **TD error** por epoch (firma de convergencia / no-estacionariedad).
- **Tiempo de entrenamiento** (JAL-GT actualiza 25 acciones conjuntas por paso
  vs 5 de IQL, asi que el coste-beneficio es un hallazgo relevante).
- **Inestabilidad intra-run y varianza inter-seed** (ver seccion 5).

### Analisis automatico de las trazas SVG

El script reconstruye las trayectorias de los agentes leyendo los SVG de POGEMA
y detecta automaticamente **deadlocks**, **colisiones/intercambios** y **cesiones
de paso** (un agente que rodea para dejar pasar al otro). Dos decisiones de
diseno importantes:

- Los SVG se generan con **vista global** (ambos agentes en el mismo sistema de
  coordenadas), no egocentrica. Las vistas egocentricas no permiten detectar
  colisiones porque cada una tiene su propio sistema de referencia.
- Se **filtran los mapas irresolubles** por BFS antes de analizar: un agente que
  no se mueve en un mapa sin ruta posible no es un deadlock de coordinacion, y
  contarlo como tal ensuciaria las conclusiones (relevante con densidad 0.25).

---

## 9. Ampliaciones (+2 puntos)

Elegimos la combinacion que suma exactamente +2 con el minimo esfuerzo de
implementacion y maxima riqueza analitica.

| Exp | Que hace | Puntos | Hipotesis |
|---|---|---|---|
| **8** | Decay **exponencial** vs lineal de epsilon | +1 | El exponencial concentra exploracion al inicio; deberia converger antes en 6x6 |
| **9** | **Generalizacion** a mapas no vistos (seeds 100..109) | +0.5 | Como `obs_to_state` es local, los mapas comparten estados -> buena generalizacion |
| **10** | **Parejas heterogeneas** de conceptos (Nash+Pareto) vs homogeneas | +0.5 | Conflicto de incentivos -> rendimiento intermedio o inferior |

Detalles de diseno:
- **EXP8** compara apples-to-apples (mismo epsilon_max/min, mismo n de
  episodios; solo cambia la forma de la curva). Es decay de **epsilon**, no de
  alpha (el enunciado dice explicitamente que no hace falta estudiar alpha decay).
- **EXP9** mide, ademas de la recompensa train vs test, la **cobertura de
  estados**: que fraccion de los estados vistos en test ya aparecieron en train.
  Esto convierte "rinde parecido en mapas nuevos" en un hallazgo explicado: rinde
  parecido *porque* la representacion local hace que los mapas compartan estados.
- **EXP10** entrena dos agentes JAL-GT con conceptos **distintos** (uno Nash,
  otro Pareto). Es la interpretacion correcta del enunciado: conceptos distintos,
  no algoritmos distintos.

**Por que descartamos otras ampliaciones:** una representacion de estado
alternativa (+1) obligaria a redisenar todo el pipeline; JAL-GT con red neuronal
(+1) seria enorme de implementar y depurar. La combinacion elegida reutiliza
infraestructura que ya tenemos.

---

## 10. Dos bugs que encontramos y corregimos

Durante la validacion del pipeline detectamos dos problemas reales que habrian
falseado el analisis:

1. **Varianza inter-seed = 0.** `build_algorithms` fijaba la semilla de
   exploracion a `agent_id`, asi que las 10 seeds producian entrenamientos
   identicos. Lo arreglamos con una linea en `main.py` (`base_seed`) que da a cada
   run una semilla distinta, manteniendo retrocompatibilidad.
2. **Analisis SVG sin colisiones.** Se guardaban SVG egocentricos (uno por
   agente), cuyas coordenadas no son comparables. Cambiamos a vista global y
   anadimos el filtro de mapas irresolubles.

---

## 11. Hipotesis centrales (lo que esperamos demostrar)

1. **JAL-GT >= IQL** en recompensa colectiva (modela la coordinacion explicitamente).
2. **IQL mas inestable** intra-run: firma de la no-estacionariedad, peor en mapas
   grandes y densos.
3. **Pareto/Welfare > Nash > Minimax** en entorno cooperativo; Minimax es
   inadecuado porque trata al colaborador como adversario.
4. La **brecha entre algoritmos y entre conceptos crece con tamano y densidad**.

---

## 12. Ejecucion (resumen)

```bash
python experiments.py            # obligatorios + ampliaciones
python experiments.py --base     # solo obligatorios
python experiments.py --extensions   # solo ampliaciones
python experiments.py --sanity   # prueba rapida del pipeline
```

Los experimentos corren en cadena y trasladan los hiperparametros optimos del
EXP1 a los demas. Las salidas (CSV, plots, SVG, analisis) quedan en sus
respectivos directorios (ver `README.md`).
