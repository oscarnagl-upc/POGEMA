# Práctica 3 SID — Aprendizaje por Refuerzo Multiagente (MARL)

Estudio comparativo de **IQL** (Independent Q-Learning) y **JAL-GT** (Joint-Action
Learning con Game Theory) sobre el entorno **POGEMA**, con 2 agentes. Incluye el
estudio de los conceptos de solución de JAL-GT (Nash, Pareto, Welfare, Minimax)
y tres ampliaciones para nota extra.

## Requisitos

- Python 3.10 (recomendado por compatibilidad con `pogema==1.2.2`)
- Dependencias en `requirements.txt`

### Instalación

```bash
conda create -n POGEMA python=3.10 -y
conda activate POGEMA
pip install -r requirements.txt

# RECOMENDADO (comprueba que el entorno baseline funciona): 
bash python sanity_check.py
```

> **Nota sobre POGEMA.** El código es compatible con `pogema==1.2.2` (la del
> `requirements.txt`) y con versiones 1.4.x, que cambiaron la ruta del módulo de
> animación. El import de `AnimationMonitor` está protegido con un `try/except`
> en `main.py` y `experiments.py` para funcionar en ambas.

## Estructura del proyecto

### Código base (proporcionado por los profesores)

| Fichero | Contenido |
|---|---|
| `game_model.py` | Modelo del juego: espacio de acciones conjuntas |
| `algorithms.py` | `MARLAlgorithm`, `JALGT` y **`IQL`** (añadido por nosotros) |
| `solution_concepts.py` | Nash, Pareto, Welfare, Minimax |
| `main.py` | `obs_to_state`, wrappers, pipeline de entrenamiento/evaluación |
| `utils.py` | Visualización básica |

### Código de experimentación (nuestro)

| Fichero | Responsabilidad |
|---|---|
| `experiments.py` | Batería completa: entrenamiento, los 10 experimentos y el orquestador |
| `config.py` | Constantes y parámetros del estudio (densidades, grids, tamaños) |
| `plots.py` | Generación de plots y persistencia de resultados en JSON |
| `svg.py` | Análisis automático de trazas SVG (deadlocks, colisiones, cesiones) |
| `sanity_check.py` | Verificación rápida de que el entorno baseline funciona |

## Cambios aplicados al código base

Para que la experimentación funcione, hay **dos añadidos mínimos** al código de
los profesores (permitidos por el enunciado):

### 1. Clase `IQL` en `algorithms.py`

Se añade al final del fichero la clase `IQL`, que implementa Q-Learning tabular
independiente (el mismo de la Práctica 2 sobre FrozenLake), con cada agente
ignorando la no-estacionariedad del entorno.

### 2. Una línea en `build_algorithms` de `main.py`

Para poder variar la exploración entre runs (y que el estudio de varianza
inter-seed sea significativo):

```python
# ANTES
kwargs.setdefault("seed", agent_id)

# DESPUÉS
base_seed = config.get("base_seed", 0)
kwargs.setdefault("seed", base_seed * 100 + agent_id)
```

Es **retrocompatible**: sin `base_seed` en el config, `base_seed=0` y el
comportamiento es idéntico al original (`seed = agent_id`).

## Uso

### Verificación previa (recomendado)

```bash
python sanity_check.py          # comprueba que el entorno baseline funciona
python experiments.py --sanity  # ejecuta toda la batería con pocos epochs
```

El modo `--sanity` corre la cadena completa en pocos minutos para confirmar que
el pipeline funciona antes del run real (que dura más).

### Ejecución de experimentos

```bash
python experiments.py               # obligatorios + ampliaciones (run completo)
python experiments.py --base        # solo experimentos obligatorios
python experiments.py --extensions  # solo ampliaciones (+2 puntos)
python experiments.py --sanity      # prueba rápida del pipeline
python experiments.py --epochs 150  # nº de epochs personalizado
```

Los experimentos se ejecutan **en cadena**: el par óptimo de hiperparámetros
`(epsilon*, gamma*)` que encuentra el primer experimento se traslada
automáticamente a todos los siguientes.

## Salidas generadas

| Directorio | Contenido |
|---|---|
| `results/` | CSV por epoch de cada run + JSON con el resumen y las decisiones |
| `plots/` | Figuras para la documentación (curvas, heatmaps, barras) |
| `renders/` | Animaciones SVG (vista global) del último epoch |
| `svg_analysis/` | CSV con el análisis cuantitativo de las trazas SVG |
| `qtables/` | Q-tables persistidas (usadas por la ampliación de generalización) |

## Experimentos

### Obligatorios

1. **Estudio (epsilon, gamma)** — grid en JAL-GT, fija el par óptimo.
2. **Transferencia a IQL** — verifica que el gamma óptimo sirve para IQL.
3. **Validación en 10x10** — comprueba si el par óptimo escala a mapas grandes.
4. **IQL vs JAL-GT, densidad 0.1** — comparativa algorítmica principal.
5. **IQL vs JAL-GT, densidad 0.25** — igual con más obstáculos.
6. **Conceptos de solución** — Nash/Pareto/Welfare/Minimax en 6x6.
7. **Contraste situacional** — Pareto vs Minimax en 4x4 y 10x10.

### Ampliaciones (+2 puntos)

8. **Decay exponencial vs lineal** de epsilon (+1).
9. **Generalización** a mapas no vistos (+0.5).
10. **Parejas heterogéneas** de conceptos de solución (+0.5).

Para más detalle sobre el diseño y las hipótesis, ver `RESUMEN_experimentos.md`.

## Autores

Práctica 3 — Sistemas Inteligentes Distribuidos (SID), UPC-FIB.
