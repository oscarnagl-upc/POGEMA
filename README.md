# Práctica 3 SID — Aprendizaje por Refuerzo Multiagente (MARL)

Estudio comparativo de **IQL** (Independent Q-Learning) y **JAL-GT** (Joint-Action
Learning con Game Theory) sobre el entorno **POGEMA**, con 2 agentes. Incluye el
estudio de los cuatro conceptos de solución de JAL-GT (Nash, Pareto, Welfare,
Minimax) y tres ampliaciones para nota extra.

## Requisitos

- Python 3.10 (recomendado por compatibilidad con `pogema==1.2.2`)
- Dependencias en `requirements.txt`

### Instalación

```bash
conda create -n POGEMA python=3.10 -y
conda activate POGEMA
pip install -r requirements.txt
```

> **Compatibilidad POGEMA.** El código funciona con `pogema==1.2.2` (la del
> `requirements.txt`) y con versiones 1.4.x, que cambiaron la ruta del módulo de
> animación. El import de `AnimationMonitor` está protegido con `try/except` en
> `main.py` y `experiments.py`.

---

## Estructura del repositorio

### Código base (proporcionado por los profesores)

| Fichero | Contenido |
|---|---|
| `game_model.py` | Modelo del juego: espacio de acciones conjuntas |
| `algorithms.py` | `MARLAlgorithm`, `JALGT` y **`IQL`** (añadido por nosotros) |
| `solution_concepts.py` | Nash, Pareto, Welfare, Minimax |
| `main.py` | `obs_to_state`, wrappers, pipeline de entrenamiento/evaluación |
| `utils.py` | Visualización básica |
| `baseline.ipynb` | Notebook de referencia de los profesores |

### Código de experimentación (nuestro)

| Fichero | Responsabilidad |
|---|---|
| `experiments.py` | Batería completa: los 10 experimentos y el orquestador |
| `config.py` | Constantes del estudio (densidades, grids, semillas, tamaños) |
| `plots.py` | Generación de plots y persistencia de resultados en JSON |
| `svg.py` | Análisis automático de trazas SVG (deadlocks, colisiones, cesiones) |

### Salidas generadas

| Directorio | Contenido |
|---|---|
| `results/` | CSV por epoch de cada run + JSON con el resumen y las decisiones |
| `plots/` | Figuras para la documentación (curvas, heatmaps, barras) |
| `renders/` | Animaciones SVG (vista global) del último epoch |
| `svg_analysis/` | CSV con el análisis cuantitativo de las trazas SVG |
| `qtables/` | Q-tables persistidas (ampliación de generalización) |

---

## Cambios aplicados al código base

Para que la experimentación funcione, hay **dos añadidos mínimos** al código de
los profesores (permitidos por el enunciado):

### 1. Clase `IQL` en `algorithms.py`

Se añade la clase `IQL`, que implementa Q-Learning tabular independiente (el mismo
de la Práctica 2 sobre FrozenLake), con cada agente ignorando la no-estacionariedad.
Hereda de `MARLAlgorithm` con la misma interfaz (`learn`, `select_action`, `explain`)
y acepta `**kwargs` para ser compatible con `build_algorithms` sin cambios.

### 2. Una línea en `build_algorithms` de `main.py`

Para obtener varianza inter-seed significativa (sin esto todos los runs son idénticos):

```python
# ANTES
kwargs.setdefault("seed", agent_id)

# DESPUÉS
base_seed = config.get("base_seed", 0)
kwargs.setdefault("seed", base_seed * 100 + agent_id)
```

Es **retrocompatible**: sin `base_seed` en el config, `base_seed=0` y el
comportamiento es idéntico al original.

---

## Ejecución de experimentos

### Flags de modo (mutuamente exclusivos)

| Invocación | Qué ejecuta |
|---|---|
| `python experiments.py` | Todos los experimentos: obligatorios (EXP1–7) + ampliaciones (EXP8–10) |
| `python experiments.py --base` | Solo obligatorios (EXP1–7, incluye EXP2 y EXP3) |
| `python experiments.py --extensions` | Solo ampliaciones (EXP8–10); EXP1 se corre para obtener los hiperparámetros |

### Parámetros numéricos opcionales

| Flag | Descripción | Default |
|---|---|---|
| `--epochs N` | Epochs de entrenamiento por run | 200 |
| `--episodes N` | Episodios por epoch | 10 |
| `--seeds N` | Número de semillas (usa `range(N)`) | 10 |

### Ejemplos

```bash
# Run completo con los parámetros por defecto
python experiments.py

# Solo obligatorios
python experiments.py --base

# Solo ampliaciones, reduciendo epochs para pruebas
python experiments.py --extensions --epochs 50

# Run completo con parámetros reducidos (útil para verificar el pipeline)
python experiments.py --epochs 10 --episodes 3 --seeds 2

# Solo obligatorios con parámetros personalizados
python experiments.py --base --epochs 50 --episodes 5 --seeds 2
```

Los experimentos se ejecutan **en cadena**: el par óptimo `(epsilon*, gamma*)`
encontrado en EXP1 se traslada automáticamente a todos los experimentos siguientes.

---

## Experimentos

### Obligatorios (EXP1–7)

1. **Estudio (ε, γ)** — grid 3×3 en JAL-GT, fija el par óptimo.
2. **Transferencia a IQL** — verifica que el γ óptimo sirve también para IQL.
3. **Validación en 10×10** — comprueba si el par óptimo escala a mapas grandes.
4. **IQL vs JAL-GT, densidad 0.1** — comparativa algorítmica en baja densidad.
5. **IQL vs JAL-GT, densidad 0.25** — ídem con más obstáculos.
6. **Conceptos de solución** — Nash/Pareto/Welfare/Minimax en 6×6, dos densidades.
7. **Contraste situacional** — Pareto vs Minimax en 4×4 y 10×10.

### Ampliaciones (EXP8–10, +2 puntos)

8. **Decay exponencial vs lineal de ε** (+1 pto): comparativa apples-to-apples del perfil de exploración.
9. **Generalización a mapas no vistos** (+0.5 pto): evaluación en semillas 100–109 tras entrenar en 0–9.
10. **Parejas heterogéneas de conceptos** (+0.5 pto): agentes JAL-GT con conceptos distintos (Nash+Pareto) frente a parejas homogéneas.

---

## Tiempo de ejecución

El run completo (`python experiments.py`) con los parámetros por defecto tarda
aproximadamente **5 horas y 30 minutos** en nuestra máquina:

- Epochs: **200**
- Episodios por epoch: **10**
- Seeds: **10** (`range(10)`)

---

## Autores

Práctica 3 — Sistemas Inteligentes Distribuidos (SID), UPC-FIB.  
Martí Checa · Óscar Nagl · Andreu Puerto
