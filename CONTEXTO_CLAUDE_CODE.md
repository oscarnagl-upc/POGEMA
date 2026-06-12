# CONTEXTO COMPLETO — Práctica 3 SID (MARL)
# Documento de contexto para Claude Code
# Autores: Martí Checa, Óscar Nagl, Andreu Puerto — UPC-FIB, SID Q2 2025-2026
# ============================================================================
#
# INSTRUCCIONES PARA CLAUDE CODE
# ============================================================================
#
# Eres un experto en MARL (Multi-Agent Reinforcement Learning) que va a analizar
# los resultados de nuestra experimentación para la Práctica 3 de SID (Sistemas
# Inteligentes Distribuidos) de la UPC-FIB.
#
# ANTES de leer este documento, te pedimos que:
#   1. Leas el enunciado completo: sid_2526q2_l3.pdf
#   2. Revises los ficheros de código: main.py, algorithms.py,
#      solution_concepts.py, game_model.py, experiments.py, config_exp.py,
#      runner.py (si existe), viz.py, svg_analysis.py
#   3. Leas los resultados de la experimentación en results/*.csv y results/*.json
#   4. Leas la salida por terminal: salida_terminal.txt
#   5. Revises los plots en plots/ y los análisis SVG en svg_analysis/*.csv
#
# NO ejecutes ningún experimento. Ya están ejecutados y los resultados están
# en las carpetas correspondientes. La ejecución tarda varias horas.
#
# TU TAREA es analizar en profundidad los resultados y generar un informe 
# que cubra exactamente los siguientes puntos (en este orden):
#
#   A. Coherencia de los resultados: ¿tienen sentido? ¿hay alguno inconsistente?
#   B. Conclusiones: ¿qué se puede concluir? ¿son correctas?
#   C. Mapeo a los criterios de la práctica: ¿la ejecución satisface los objetivos?
#   D. Evaluación de la resolución: nota sobre 10 con justificación por criterio
#   E. Mejoras: ¿cambiarías algo para maximizar la nota?
#
# Tómate todo el tiempo que necesites. La profundidad del análisis es lo que importa.
#
# ============================================================================


# ============================================================================
# SECCIÓN 1: DESCRIPCIÓN DE LA PRÁCTICA Y RÚBRICA
# ============================================================================

## Objetivo de la práctica

La práctica consiste en comparar dos algoritmos MARL sobre el entorno POGEMA
(grid 2D con navegación multiagente):

- IQL (Independent Q-Learning): Q-Learning tabular estándar donde cada agente
  ignora al otro. El mismo Q-Learning que en la Práctica 2 de FrozenLake,
  trasladado a entorno multiagente. Cada agente mantiene Q-table [num_states x
  num_actions] = [1024 x 5] e ignora la no-estacionariedad causada por el otro
  agente actualizando su política simultáneamente.

- JAL-GT (Joint-Action Learner with Game Theory): mantiene Q-table conjunta
  [num_agents x num_states x num_joint_actions] = [2 x 1024 x 25] y usa un
  concepto de solución de teoría de juegos para derivar la política. Los
  conceptos disponibles son Nash, Pareto, Welfare y Minimax.

## Restricciones obligatorias del enunciado

- Exactamente 2 agentes en todos los experimentos.
- La función obs_to_state del baseline debe usarse tal cual (no modificarla).
  Codifica: obstáculos cardinales (4 bits) + otros agentes cardinales (4 bits)
  + posición relativa objetivo (2 bits) = 1024 estados posibles.
- Solo se pueden usar las dependencias del requirements.txt.

## Rúbrica de evaluación (nota base máxima: 10)

CRITERIO 1 — Ejecución (30%):
  El código funciona a partir del README y permite reproducir los experimentos
  obligatorios. El pipeline debe ser robusto y ejecutable sin modificaciones.

CRITERIO 2 — Implementación (20%):
  La implementación de IQL es consistente con los objetivos de experimentación
  y con el funcionamiento teórico. Cualquier adaptación adicional también debe
  estar bien implementada.

CRITERIO 3 — Diseño de experimentos (25%):
  Los experimentos obligatorios están planteados a partir de hipótesis y
  justificados en base al comportamiento teórico. La selección de parámetros y
  comparativas es razonada y coherente.

CRITERIO 4 — Análisis de resultados (25%):
  Los resultados se analizan de manera correctamente contextualizada con respecto
  a: las hipótesis de partida, la no-estacionariedad del entorno multiagente, y
  los incentivos de coordinación de cada algoritmo/concepto de solución.
  Se usan correctamente los outputs: recompensas, TD errors, animaciones SVG.
  Se concluye con recomendaciones de cómo entrenar agentes en entornos similares.

## Ampliaciones (hasta +2 puntos)

+0.5 cada una (máximo +1 en esta categoría):
  - Parejas heterogéneas de conceptos de solución (comparar contra homogéneas)
  - Generalización a mapas no vistos en entrenamiento
  - Escalar número de agentes

+1 cada una:
  - Representación de estado alternativa a obs_to_state
  - Mejora metodológica o algorítmica sencilla bien justificada (e.g. decay
    distinto o protocolo de evaluación más robusto)
  - JAL-GT con modelo lineal o red neuronal

Para que una extensión puntúe debe incluir: hipótesis, comparación con baseline,
y análisis contextualizado.


# ============================================================================
# SECCIÓN 2: ARQUITECTURA DEL CÓDIGO
# ============================================================================

## Ficheros del codebase base (profesores, en la raíz del proyecto)

- main.py: obs_to_state, RewardWrapper (-0.01/paso sin llegar), create_env,
  build_algorithms, compute_epsilon (decay lineal), train_episode,
  evaluate_episode. El exp_config baseline usa: size=4, epochs=200,
  density=0.1, learning_rate=0.01, epsilon_max=1, epsilon_min=0.1,
  JAL-GT+Pareto.

- algorithms.py: clase base MARLAlgorithm (abstracta), clase JALGT,
  clase IQL (añadida por nosotros).

- solution_concepts.py: SolutionConcept (abstracta), ParetoSolutionConcept,
  NashSolutionConcept, WelfareSolutionConcept, MinimaxSolutionConcept.

- game_model.py: GameModel — genera el espacio de acciones conjuntas mediante
  producto cartesiano (itertools.product). Con 2 agentes y 5 acciones: 25
  acciones conjuntas.

- utils.py: draw_history (única función, genera curva de aprendizaje).

## Ficheros de experimentación (nuestros)

- experiments.py: orquestador completo. Contiene la lógica de entrenamiento
  (runner), todos los experimentos (obligatorios 1-7 + ampliaciones 8-10),
  la salida por terminal (log/banner) y el main() con flags.

- config_exp.py: todas las constantes del estudio. Punto de entrada para
  modificar parámetros sin tocar experiments.py.

- viz.py: funciones de plotting (plot_curve, plot_heatmap, plot_bars,
  plot_grouped_bars, save_json).

- svg_analysis.py: protocolo autónomo de análisis de trazas SVG. Reconstruye
  trayectorias desde los atributos <animate> del SVG y detecta deadlocks,
  colisiones y cesiones de paso. NOTA: los SVG se generan con vista global
  (egocentric_idx=None) para que ambos agentes estén en el mismo sistema de
  coordenadas.

## Modificaciones al código base (mínimas y justificadas)

1. IQL añadido en algorithms.py: Q-table individual [1024 x 5], actualización
   Q-Learning estándar (réplica de FrozenLake), ignorando la no-estacionariedad.
   Interfaz idéntica a JALGT (mismos métodos: learn, select_action, set_epsilon,
   explain) para ser compatible con train_episode y evaluate_episode sin cambios.

2. Una línea cambiada en build_algorithms de main.py:
   ANTES: kwargs.setdefault("seed", agent_id)
   DESPUÉS: base_seed = config.get("base_seed", 0)
            kwargs.setdefault("seed", base_seed * 100 + agent_id)
   RAZÓN: Sin este cambio, todas las seeds producían entrenamientos idénticos
   (varianza inter-seed = 0.000, bug detectado y corregido durante el desarrollo).
   Es retrocompatible: sin base_seed en el config, base_seed=0 y el
   comportamiento es idéntico al original (seed = agent_id).


# ============================================================================
# SECCIÓN 3: DECISIONES DE DISEÑO Y SU JUSTIFICACIÓN
# (todo lo que debatimos y justificamos antes de ejecutar)
# ============================================================================

## 3.1 Parámetros fijos

- alpha = 0.01: valor de los profesores. Valor conservador deliberado porque la
  Q-table conjunta de JAL-GT tiene un target inestable (depende de la política
  conjunta que cambia en cada paso). Un alpha bajo amortigua las oscilaciones.
  NO lo tocamos porque el enunciado no pide estudiarlo.

- episodes_per_epoch = 10, maps = 10: con esta combinación, cada mapa se ve ~1
  vez por epoch. Equilibrio entre diversidad y frecuencia de visita.

- 10 seeds (0..9): mínimo para estimar media y varianza inter-seed sin depender
  de una semilla afortunada.

## 3.2 Densidades: 0.1 y 0.25

Verificadas empíricamente con BFS sobre 300+ mapas por configuración. Medimos:
  - % de mapas resolubles (ambos agentes pueden alcanzar su objetivo)
  - Ratio de rodeo (longitud del camino / distancia Manhattan): mide si los
    obstáculos realmente fuerzan coordinación

Resultados del análisis BFS:
  density=0.1:  98% resoluble, rodeo=1.05-1.08 (diferencia mínima vs 0.0)
  density=0.2:  87% resoluble, rodeo=1.08-1.13 (contraste casi imperceptible)
  density=0.25: 84% resoluble, rodeo=1.13-1.21 (contraste visible en 6x6/10x10)
  density=0.3:  63-76% resoluble, rodeo=1.17-1.30 (demasiados mapas muertos)

Elegimos 0.25 porque: tiene contraste visible (especialmente en 10x10 donde
rodeo=1.21), mantiene ≥84% resolubilidad, y los mapas completamente muertos
son solo 2-5% (vs 7-9% con 0.3). Descartamos 0.3 porque triplicaba los mapas
muertos, que contaminarían la recompensa media del bloque IQL vs JAL-GT.

IMPORTANTE para el análisis: los mapas irresolubles se filtran del análisis SVG
por BFS. Un agente sin ruta posible que no se mueve NO es un deadlock de
coordinación y no debe contarse como tal.

## 3.3 Grid de hiperparámetros

Grid conjunto 3x3 (no secuencial): epsilon y gamma están acoplados, estudiarlos
de forma independiente puede llevar a un par subóptimo.

gamma ∈ {0.90, 0.95, 0.99}: Rango validado en Práctica 2 (FrozenLake con mismo
Q-Learning). En FrozenLake: tasa de éxito de 62.5% con gamma=0.9 vs 71.6% con
gamma=0.99. Eliminamos gamma=0.8 porque a 10 pasos solo queda 11% de la señal
de la recompensa terminal (0.8^10 = 0.107), insuficiente para aprender
trayectorias largas. Mantenemos 0.95 como ancla al valor del baseline.

epsilon_max ∈ {0.5, 0.75, 1.0}: Saltos uniformes de 0.25 en la zona útil.
Descartamos 0.3 del grid (aunque lo debatimos) porque en Práctica 2 demostró
producir bimodalidad y rendimiento pésimo — sabíamos que daría mal resultado y
habría desperdiciado 3 runs confirmando lo predecible. Los 3 valores elegidos
cubren "exploración moderada, alta y máxima" sin descartes teóricos seguros.

## 3.4 Métrica de convergencia (percentil 90)

Razón: no podemos comparar algoritmos con "media de los últimos N epochs" porque
IQL converge en ~epoch 90 y JAL-GT en ~epoch 200. Una ventana fija mediría a
JAL-GT en plena rampa de aprendizaje (injusto).

Solución: para cada run, calcular el percentil 90 de toda la curva de recompensa.
El "epoch de convergencia" es el primero donde la media móvil (ventana 20) supera
el P90. La "recompensa en convergencia" es la media desde ese epoch hasta el final.

Usamos P90 (no máximo) para evitar picos espurios que inflarían el umbral.

## 3.5 Métrica de no-estacionariedad

La no-estacionariedad NO se mide como varianza inter-seed. Se manifiesta como
inestabilidad DENTRO de un mismo run: curva oscilante y TD error no monótono.

Dos métricas separadas:
  - Varianza inter-seed: dispersión entre 10 seeds (mide robustez del algoritmo)
  - Inestabilidad intra-run: desviación de la recompensa en ventana móvil dentro
    de un run (mide la firma de no-estacionariedad de IQL)

Hipótesis: IQL tendrá mayor inestabilidad intra-run que JAL-GT.

## 3.6 Concepto de solución en EXP4/5

Usamos Pareto como concepto fijo en la comparativa IQL vs JAL-GT porque:
  1. Es el default del baseline de los profesores
  2. Es el más adecuado para entornos cooperativos (maximiza bienestar colectivo
     sin que ningún agente empeore)
  3. El EXP6 estudia los conceptos específicamente; mezclarlos aquí añadiría
     variables no controladas

## 3.7 Número de epochs: 150 (o el valor que se haya usado)

Basado en curvas de convergencia reales ejecutadas previamente (250 epochs,
2 seeds, 4x4 y 6x6):
  - IQL 4x4:   convergencia práctica en epoch ~87, completamente plano desde 150
  - IQL 6x6:   convergencia práctica en epoch ~89, plano desde 150
  - JAL-GT 4x4: convergencia práctica en epoch ~204 (sigue subiendo lentamente)
  - JAL-GT 6x6: convergencia práctica en epoch ~158

Los profesores usaron 200 en el baseline (solo para JAL-GT 4x4). Para nuestra
batería mixta (IQL + JAL-GT + múltiples tamaños), 150 es el punto donde todas
las configuraciones han mostrado su comportamiento característico. Las
configuraciones de IQL ya están completamente convergidas y las de JAL-GT han
pasado su rampa principal de aprendizaje.


# ============================================================================
# SECCIÓN 4: DESCRIPCIÓN DE LOS EXPERIMENTOS
# ============================================================================

## Experimentos obligatorios (ejecutados con --base o sin flags)

EXP1 — Grid (epsilon × gamma):
  Qué: 9 runs de JAL-GT+Pareto en 4x4, densidad 0.1, 10 seeds.
  Para qué: fijar el par óptimo (epsilon*, gamma*) que se traslada a los demás.
  Hipótesis: epsilon alto → mejor cobertura, menos varianza inter-seed.
             gamma alto → necesario para propagar recompensa terminal a través
             de trayectorias largas (recompensa de POGEMA al llegar: +1.0).
  Salida: heatmap de recompensa en convergencia (plots/exp1_heatmap_reward.png)
          y heatmap de varianza inter-seed (plots/exp1_heatmap_std.png).
  JSON: results/exp1_summary.json (incluye best_epsilon, best_gamma)

EXP2 — Transferencia de gamma a IQL:
  Qué: 3 runs de IQL en 4x4 con epsilon* fijo, variando gamma.
  Para qué: verificar que el gamma óptimo de JAL-GT también es razonable para
  IQL, garantizando fairness en la comparativa principal.
  Si los gammas difieren: usar el de JAL-GT en ambos (decisión de fairness
  documentada), no el óptimo de cada uno por separado.
  JSON: results/exp2_summary.json

EXP3 — Validación del par óptimo en 10x10:
  Qué: 2 runs de JAL-GT+Pareto en 10x10: (epsilon*, gamma*) vs baseline (1.0, 0.95).
  Para qué: comprobar si el par calibrado en 4x4 transfiere a mapas grandes.
  Limitación honesta: calibramos en 4x4 y validamos en 10x10 sin re-optimizar.
  Si el par no transfiere: documentar como limitación del diseño.
  JSON: results/exp3_summary.json

EXP4 — IQL vs JAL-GT, densidad 0.1:
  Qué: (IQL + JAL-GT+Pareto) × (4x4, 6x6, 10x10), densidad 0.1, 10 seeds.
  Métricas: recompensa colectiva/individual, TD error, tiempo de entrenamiento,
            inestabilidad intra-run, varianza inter-seed.
  Hipótesis principal: JAL-GT >= IQL en recompensa colectiva. IQL más inestable
  intra-run (firma de no-estacionariedad), efecto amplificado en mapas grandes.
  SVG: generados para 6x6 (densidad media es donde más se ve la coordinación).
  CSVs por run: results/exp4_IQL_size{4,6,10}.csv, results/exp4_JAL-GT_size{4,6,10}.csv
  JSON: results/exp4_summary.json

EXP5 — IQL vs JAL-GT, densidad 0.25:
  Mismo diseño que EXP4 con densidad 0.25.
  Hipótesis adicional: la brecha JAL-GT - IQL crece con la densidad (más
  obstáculos → coordinación más necesaria → JAL-GT se beneficia más).
  SVG generados para 6x6 + análisis automático en svg_analysis/.
  JSON: results/exp5_summary.json

EXP6 — Conceptos de solución en JAL-GT (6x6):
  Qué: (Nash + Pareto + Welfare + Minimax) × (densidad 0.1 + 0.25), 6x6, 10 seeds.
  Hipótesis:
    Pareto/Welfare: mejores en entorno cooperativo, pueden coincidir.
    Nash: algo peor, riesgo de deadlock si "ambos quietos" es equilibrio local.
    Minimax: el peor, trata al colaborador como adversario (inadecuado aquí).
  Para cada densidad: plot de curvas (exp6_concepts_d{density}.png) y barras
  de ranking (exp6_concepts_bars_d{density}.png).
  SVG + análisis automático para densidad 0.25 (donde más se diferencian).
  JSON: results/exp6_summary.json

EXP7 — Contraste situacional Pareto vs Minimax:
  Qué: Pareto y Minimax en 4x4 y 10x10, densidad 0.1.
  Para qué: verificar si el ranking de conceptos depende del tamaño. Hipótesis:
  en mapas pequeños las diferencias se diluyen (alta densidad de visita de
  estados); en mapas grandes se amplifican.
  JSON: results/exp7_summary.json

## Ampliaciones (+2 puntos)

EXP8 — Decay exponencial vs lineal de epsilon (+1 punto):
  Qué: comparar decay lineal (baseline) vs exponencial en IQL y JAL-GT, 4x4 y 6x6.
  La k del exponencial se calibra para que la curva alcance epsilon_min al final.
  Apples-to-apples: mismo epsilon_max, epsilon_min y nº de episodios.
  NOTA IMPORTANTE: es decay de EPSILON, no de alpha. El enunciado dice
  explícitamente que no hace falta estudiar alpha decay.
  Implementación: monkeypatch acotado y reversible de compute_epsilon, restaurado
  con try/finally. Verificado que se restaura correctamente tras la ejecución.
  Incluye plot del perfil epsilon(t) de ambas estrategias (exp8_epsilon_profile.png).
  JSON: results/exp8_summary.json

EXP9 — Generalización a mapas no vistos (+0.5 puntos):
  Qué: entrenar en seeds 0..9, evaluar en seeds 100..109 (nunca vistas).
  Métrica núcleo: cobertura de estados (fracción de estados del test que ya
  aparecieron en train). POGEMA usa radio 1 en obs_to_state → la observación
  es local e independiente de la posición absoluta → los mapas comparten muchos
  estados. Si la cobertura es alta, la generalización es casi trivial.
  Hipótesis: buena generalización porque obs_to_state es local.
  JSON: results/exp9_summary.json

EXP10 — Parejas heterogéneas de conceptos (+0.5 puntos):
  Qué: dos agentes JAL-GT con CONCEPTOS DISTINTOS (agente0=Nash, agente1=Pareto)
  comparado contra homogéneas (Nash+Nash, Pareto+Pareto) en 6x6.
  NOTA CRÍTICA: el enunciado pide "conceptos de solución diferentes para ambos",
  NO mezclar algoritmos distintos. Esta es la interpretación correcta.
  Hipótesis: conflicto de incentivos → rendimiento inferior a la mejor homogénea.
  JSON: results/exp10_summary.json


# ============================================================================
# SECCIÓN 5: SISTEMA DE RECOMPENSAS Y MÉTRICAS
# ============================================================================

## Cómo funciona la recompensa en POGEMA

- Al llegar al objetivo: +1.0 (recompensa de POGEMA, on_target='finish')
- Por cada paso sin llegar: -0.01 (aplicado por RewardWrapper de main.py)
- Peor caso (16 pasos, nadie llega): 10 mapas × 2 agentes × (-0.16) = -3.20
- La recompensa colectiva por epoch es la SUMA sobre los 10 mapas de evaluación
  de las recompensas de ambos agentes.

## Interpretación de rangos

- Recompensa colectiva muy negativa (~-3): los agentes casi nunca llegan.
  Normal con pocos epochs o en mapas grandes donde el objetivo está lejos.
- Recompensa ~0: equilibrio entre llegadas exitosas y penalizaciones.
- Recompensa positiva y alta: los agentes llegan frecuentemente y rápido.

## TD error

- TD error = r + gamma * max Q(s',a') - Q(s,a): diferencia entre lo esperado
  y lo observado. Se guarda la suma de valores absolutos por epoch.
- TD error decreciente: el algoritmo converge (la Q-table se estabiliza).
- TD error oscilante o no-decreciente en IQL: firma de no-estacionariedad.
  El target cambia porque el otro agente también actualiza su política.

## Análisis SVG automático

El módulo svg_analysis.py reconstruye trayectorias y detecta:
- Deadlocks: agente sin moverse (moves=0) en toda la traza
- Colisiones potenciales: dos agentes en la misma celda o que intercambian celdas
- Cesión de paso: un agente con eficiencia <0.5 (rodea) mientras el otro >0.7
  (va casi directo)
- Eficiencia de trayectoria: desplazamiento_neto / movimientos (1.0 = directa)

IMPORTANTE: los mapas irresolubles (BFS confirma que no hay ruta) se filtran
antes del análisis para no contaminar estas métricas.


# ============================================================================
# SECCIÓN 6: BUGS CORREGIDOS DURANTE EL DESARROLLO
# (relevantes para entender las decisiones de implementación)
# ============================================================================

## Bug 1: Varianza inter-seed = 0.000

Síntoma: en el --sanity, todas las seeds producían std inter-seed = 0.000.
Causa: build_algorithms en main.py fijaba la semilla de exploración a agent_id
(0 o 1), independientemente del master_seed del bucle de seeds. El RNG de
exploración es random.Random (independiente de np.random), así que
np.random.seed(master_seed) no lo afectaba.
Corrección: añadir base_seed al config, con la fórmula:
  seed = base_seed * 100 + agent_id
Así cada run tiene una semilla de exploración distinta, y dentro de cada run
los dos agentes también tienen semillas distintas.
Verificación: después del fix, std inter-seed = 0.53 (valor real).

## Bug 2: Análisis SVG sin colisiones (0%)

Síntoma: colisiones y cesiones de paso siempre 0% en el análisis automático.
Causa: se guardaban SVGs egocéntricos (un fichero por agente), cada uno con
su propio sistema de referencia centrado en ese agente. Las coordenadas de
los dos SVGs no son comparables → imposible detectar colisiones.
Corrección: cambiar a vista global (egocentric_idx=None), un único SVG por
mapa con ambos agentes en el mismo sistema de coordenadas.
Verificación: prueba sintética confirmó que el detector de colisiones funciona
correctamente con la vista global.


# ============================================================================
# SECCIÓN 7: ESTRUCTURA DE ARCHIVOS DE SALIDA
# (dónde encontrar cada resultado)
# ============================================================================

results/
  exp1_summary.json        -> best_epsilon, best_gamma, matriz de convergencia
  exp2_summary.json        -> gamma óptimo IQL vs JAL-GT, decisión de fairness
  exp3_summary.json        -> transferencia del par óptimo a 10x10
  exp4_summary.json        -> comparativa IQL vs JAL-GT, density=0.1
  exp5_summary.json        -> comparativa IQL vs JAL-GT, density=0.25
  exp6_summary.json        -> ranking de conceptos, ambas densidades
  exp7_summary.json        -> Pareto vs Minimax en 4x4 y 10x10
  exp8_summary.json        -> decay exponencial vs lineal
  exp9_summary.json        -> generalización a mapas no vistos
  exp10_summary.json       -> parejas heterogéneas de conceptos

  exp{N}_{CONFIG}_size{S}.csv  -> curva época a época: collective_reward_mean/std,
                                   td_error_mean, individual_reward_agent{0,1}_mean

plots/
  exp1_heatmap_reward.png  -> grid epsilon x gamma (recompensa en convergencia)
  exp1_heatmap_std.png     -> grid epsilon x gamma (varianza inter-seed)
  exp2_iql_gamma.png       -> efecto del gamma en IQL
  exp3_validation_10x10.png -> par óptimo vs baseline en 10x10
  exp4_reward_size{4,6,10}.png -> curvas IQL vs JAL-GT por tamaño
  exp4_tderror_size{4,6,10}.png -> TD error por tamaño
  exp5_*.png               -> igual que exp4 con density=0.25
  exp6_concepts_d{density}.png -> curvas por concepto de solución
  exp6_concepts_bars_d{density}.png -> ranking de conceptos
  exp7_pareto_minimax_size{4,10}.png -> contraste situacional
  exp8_{algo}_size{4,6}.png -> decay expo vs lineal por algoritmo/tamaño
  exp8_epsilon_profile.png  -> perfil epsilon(t) de ambas estrategias
  exp9_generalization.png   -> recompensa train vs test
  exp10_heterogeneous_d{density}.png -> parejas homogéneas vs heterogénea

renders/
  exp{N}_{CONCEPT}-size{S}-d{D}-map{M}.svg -> animaciones globales (vista global)

svg_analysis/
  exp5_{IQL,JAL-GT}_svg.csv  -> análisis trazas EXP5
  exp6_{Nash,Pareto,Welfare,Minimax}_svg.csv -> análisis trazas EXP6

qtables/
  exp9_{run}_agent{0,1}.npy  -> Q-tables persistidas para el análisis de
                                  generalización


# ============================================================================
# SECCIÓN 8: HIPÓTESIS CENTRALES A VERIFICAR EN LOS RESULTADOS
# ============================================================================

H1: JAL-GT >= IQL en recompensa colectiva (modela coordinación explícitamente).

H2: IQL más inestable intra-run que JAL-GT (no-estacionariedad). El efecto
    debe ser más visible en mapas grandes (10x10) y densidad alta (0.25).

H3: Ranking de conceptos en entorno cooperativo:
    Pareto/Welfare > Nash > Minimax.
    Minimax debe ser el peor (trata al colaborador como adversario).

H4: La brecha entre algoritmos y entre conceptos crece con tamaño y densidad.

H5 (EXP7): El ranking Pareto > Minimax se amplifica en 10x10 vs 4x4.

H6 (EXP8): El decay exponencial converge antes en 6x6 (efecto marginal en 4x4).

H7 (EXP9): Buena generalización a mapas no vistos por la representación local
    de obs_to_state (alta cobertura de estados).

H8 (EXP10): La pareja heterogénea Nash+Pareto rinde por debajo de la mejor
    pareja homogénea (conflicto de incentivos).


# ============================================================================
# SECCIÓN 9: PREGUNTAS ESPECÍFICAS PARA EL ANÁLISIS
# ============================================================================

Además del análisis general, presta atención específica a:

1. ¿Los valores de std inter-seed son distintos de 0 en todos los experimentos?
   Si alguno es exactamente 0.000, podría indicar que el fix del base_seed no
   se aplicó correctamente (o que hay determinismo accidental).

2. ¿Las recompensas negativas (si las hay) son coherentes con el sistema de
   recompensas? Una recompensa colectiva de -3 es el peor caso teórico
   (nadie llega en ningún mapa). ¿En qué configuraciones y por qué?

3. ¿El análisis SVG detecta diferencias reales entre conceptos de solución?
   Especialmente: ¿Minimax tiene más deadlocks que Pareto? ¿Welfare tiene
   más cesiones de paso?

4. ¿El EXP2 confirma o refuta que el gamma óptimo transfiere de JAL-GT a IQL?
   Esto afecta directamente a la fairness de la comparativa EXP4/5.

5. ¿El EXP3 confirma o refuta la transferencia del par óptimo a 10x10?
   Si no transfiere, ¿cómo afecta esto a las conclusiones de EXP4/5 en 10x10?

6. ¿Se aprecia la hipótesis H4 (brecha crece con tamaño y densidad)?
   Comparar la brecha JAL-GT - IQL en: 4x4 d=0.1 vs 10x10 d=0.1 vs
   4x4 d=0.25 vs 10x10 d=0.25.

7. Para las ampliaciones: ¿cada una incluye hipótesis + baseline + análisis
   contextualizado? Si falta alguno de los tres, la ampliación no puntúa.


# ============================================================================
# FIN DEL CONTEXTO
# ============================================================================
#
# Ahora que tienes todo el contexto, procede a:
#   1. Leer el enunciado (sid_2526q2_l3.pdf)
#   2. Revisar los códigos (especialmente algorithms.py, experiments.py)
#   3. Leer todos los results/*.json (resúmenes de cada experimento)
#   4. Leer la salida por terminal adjunta (salida_terminal.txt)
#   5. Revisar los plots en plots/ y svg_analysis/*.csv
#
# Luego produce el informe de análisis cubriendo los puntos A-E descritos
# al inicio. Tómate el tiempo que necesites. No ejecutes nada.
