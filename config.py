"""
config_exp.py -- Constantes y parámetros del estudio experimental.

Centraliza todos los "magic numbers" del experimento en un único módulo, para
que cambiar el alcance del estudio (densidades, grids de hiperparámetros,
tamaños, epochs) sea una edición localizada y no haya valores dispersos.

No importa nada del codebase ni de los demás módulos: es la raíz del árbol de
dependencias.
"""

from solution_concepts import (
    NashSolutionConcept,
    ParetoSolutionConcept,
    WelfareSolutionConcept,
    MinimaxSolutionConcept,
)

# --- Directorios de salida ---------------------------------------------------
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
RENDERS_DIR = "renders"
SVG_ANALYSIS_DIR = "svg_analysis"
QTABLES_DIR = "qtables"          # Q-tables persistidas (ampliación de generalización)

# --- Parámetros fijos en TODOS los experimentos ------------------------------
# (justificados en la documentación)
BASE = {
    "num_agents": 2,                  # restricción del enunciado
    "num_states": 16 * 16 * 4,        # consecuencia de obs_to_state (1024)
    "maps": 10,                       # 10 semillas de mapa por run
    "episodes_per_epoch": 10,         # configuración baseline
    "episode_length": 16,             # configuración baseline
    "learning_rate": 0.01,            # alpha calibrado por los profesores; NO se toca
    "epsilon_min": 0.1,               # cota inferior de epsilon (baseline)
    "save_every": None,               # no generar SVGs durante el sweep
    "algorithm_kwargs": {},
}

# --- Semillas de entrenamiento/evaluación ------------------------------------
TRAIN_SEEDS = list(range(10))         # 0..9  (también usadas como eval en EXP9)
# Semillas NO vistas en entrenamiento (ampliación de generalización)
UNSEEN_SEEDS = list(range(100, 110))  # 100..109

# --- Seeds por experimento (tabla de reducción de tiempo) --------------------
EXP1_SEEDS  = list(range(5))   # grid hiperparámetros: 5 seeds
EXP2_SEEDS  = list(range(3))   # transferencia gamma a IQL: 3 seeds
EXP3_SEEDS  = list(range(3))   # validación par óptimo 10x10: 3 seeds
EXP45_SEEDS = list(range(3))   # IQL vs JAL-GT comparativa: 3 seeds
EXP6_SEEDS  = list(range(3))   # conceptos de solución: 3 seeds
EXP7_SEEDS  = list(range(3))   # contraste situacional: 3 seeds
EXP8_SEEDS  = list(range(5))   # decay exponencial vs lineal: 5 seeds
EXP9_SEEDS  = list(range(3))   # generalización (política, no inter-seed): 3 seeds
EXP10_SEEDS = list(range(3))   # parejas heterogéneas: 3 seeds

# --- Espacio de hiperparámetros ----------------------------------------------
# (decidido tras análisis crítico + datos de la Práctica 2)
EPSILON_GRID = [0.5, 0.75, 1.0]       # saltos uniformes de 0.25, zona sana
GAMMA_GRID = [0.90, 0.95, 0.99]       # rango validado en la Práctica 2

# --- Densidades --------------------------------------------------------------
# (verificadas empíricamente por BFS: resolubilidad y rodeo)
DENSITY_LOW = 0.1                     # 98% resoluble, aísla coordinación pura
DENSITY_MID = 0.25                    # contraste de rodeo visible en 6x6 y 10x10

# --- Tamaños obligatorios ----------------------------------------------------
SIZES = [4, 6, 10]

# --- Epochs ------------------------------------------------------------------
DEFAULT_EPOCHS = 200
SANITY_EPOCHS = 8

# --- Conceptos de solución disponibles, indexados por nombre legible ---------
SOLUTION_CONCEPTS = {
    "Nash": NashSolutionConcept,
    "Pareto": ParetoSolutionConcept,
    "Welfare": WelfareSolutionConcept,
    "Minimax": MinimaxSolutionConcept,
}
