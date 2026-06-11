"""
================================================================================
  experiments.py  --  Batería completa de experimentos
  Práctica 3 SID -- MARL: IQL vs JAL-GT sobre POGEMA
================================================================================

Contiene la salida por terminal, el núcleo de entrenamiento, todos los
experimentos (obligatorios y ampliaciones) y el orquestador. Se apoya en tres
módulos auxiliares con separación de responsabilidades clara:

  config_exp.py    constantes y parámetros del estudio
  viz.py           plots y persistencia JSON
  svg_analysis.py  protocolo de análisis automático de trazas SVG (autónomo)

Flujo de experimentos OBLIGATORIOS:
  EXP 1  Estudio conjunto (epsilon, gamma)   -> fija (eps*, gamma*)
  EXP 2  Transferencia de gamma a IQL
  EXP 3  Validación del par óptimo en 10x10
  EXP 4  IQL vs JAL-GT, densidad baja (0.1)
  EXP 5  IQL vs JAL-GT, densidad media (0.25)
  EXP 6  Conceptos de solución (Nash/Pareto/Welfare/Minimax) en 6x6
  EXP 7  Conceptos contrastantes (Pareto/Minimax) en 4x4 y 10x10

Ampliaciones (+2 puntos), lanzadas con --extensions:
  EXP 8  Decay exponencial vs lineal de epsilon          (+1)
  EXP 9  Generalización a mapas no vistos                (+0.5)
  EXP 10 Parejas heterogéneas de conceptos de solución   (+0.5)

Uso:
  python experiments.py             # obligatorios + ampliaciones (run completo)
  python experiments.py --base      # solo experimentos obligatorios
  python experiments.py --extensions  # solo ampliaciones (+2 puntos)
  python experiments.py --sanity    # verificación rápida del pipeline (8 epochs)
"""

import os
import time
import math
import argparse

import numpy as np
import pandas as pd

from config import (
    RESULTS_DIR, PLOTS_DIR, RENDERS_DIR, SVG_ANALYSIS_DIR, QTABLES_DIR, BASE,
    TRAIN_SEEDS, UNSEEN_SEEDS, EPSILON_GRID, GAMMA_GRID,
    DENSITY_LOW, DENSITY_MID, SIZES, DEFAULT_EPOCHS, SANITY_EPOCHS,
    SOLUTION_CONCEPTS,
)
from plots import plot_curve, plot_heatmap, plot_bars, plot_grouped_bars, save_json
from svg import analyze_svgs

# --- Codebase de los profesores ---------------------------------------------
from game_model import GameModel
from algorithms import JALGT, IQL
from solution_concepts import NashSolutionConcept, ParetoSolutionConcept
from main import (
    obs_to_state,
    RewardWrapper,
    build_algorithms,
    compute_epsilon,
    train_episode,
    evaluate_episode,
)
from pogema import pogema_v0, GridConfig
try:
    from pogema.animation import AnimationMonitor, AnimationConfig
except ModuleNotFoundError:
    from pogema.svg_animation.animation_wrapper import AnimationMonitor, AnimationConfig

ALL_DIRS = (RESULTS_DIR, PLOTS_DIR, RENDERS_DIR, SVG_ANALYSIS_DIR, QTABLES_DIR)


# ============================================================================
#  SALIDA POR TERMINAL
# ============================================================================

def log(msg, level=0):
    """Impresión estructurada por terminal con sangría según el nivel."""
    prefix = {0: "", 1: "  ", 2: "    ", 3: "      "}.get(level, "  " * level)
    print(f"{prefix}{msg}", flush=True)


def banner(title):
    """Encabezado visual para separar experimentos en la salida de terminal."""
    line = "=" * 78
    print("\n" + line)
    print(f"  {title}")
    print(line, flush=True)


def make_dirs(dirs):
    """Crea los directorios de salida si no existen."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def report_svg_findings(label, summary):
    """Imprime por terminal los hallazgos automáticos del análisis SVG."""
    print()  # separación visual entre conclusiones experimentales y análisis SVG
    if summary is None:
        log(f"[SVG] {label}: no se generaron/encontraron SVGs analizables.", 1)
        return
    log(f"[SVG] Hallazgos automáticos para {label}  (n={summary['n_svgs']}"
        + (f", {summary['n_skipped_unsolvable']} mapas irresolubles excluidos"
           if summary.get("n_skipped_unsolvable") else "") + "):", 1)
    log(f"- deadlocks (algún agente quieto):  {summary['pct_any_deadlock']}%", 2)
    log(f"- deadlocks dobles (ambos quietos): {summary['pct_both_deadlock']}%", 2)
    log(f"- colisiones/intercambios potenciales: {summary['pct_collision']}%", 2)
    log(f"- patrón de cesión de paso:         {summary['pct_yielding']}%", 2)
    log(f"- eficiencia media de trayectoria:  {summary['mean_efficiency']} "
        f"(1.0 = directa)", 2)
    log("Comprueba estos valores abriendo los SVG correspondientes en el navegador.", 2)
# ============================================================================
#  INFRAESTRUCTURA DE ENTORNO Y CONFIG
# ============================================================================

def make_anim_config(directory, save_every=None, egocentric_idx=None):
    """Construye AnimationConfig de forma compatible con ambas APIs de pogema."""
    try:
        return AnimationConfig(directory=directory, static=False, show_agents=True,
                               egocentric_idx=egocentric_idx,
                               save_every_idx_episode=save_every,
                               show_border=True, show_lines=True)
    except TypeError:
        return AnimationConfig(directory=directory, static=False, show_agents=True,
                               egocentric_idx=egocentric_idx,
                               save_every_idx_episode=save_every,
                               show_grid_lines=True)


def make_save_anim_config(egocentric_idx):
    """AnimationConfig para save_animation (sin directory), compatible ambas APIs."""
    try:
        return AnimationConfig(egocentric_idx=egocentric_idx,
                               show_border=True, show_lines=True)
    except TypeError:
        return AnimationConfig(egocentric_idx=egocentric_idx, show_grid_lines=True)


def build_config(size, density, epochs, epsilon_max, gamma,
                 algorithm_cls, solution_concept_cls, renders_dir,
                 per_agent_concepts=None):
    """Construye un exp_config completo a partir de los parámetros variables.

    Reutiliza la estructura exacta que esperan las funciones de main.py.
    'per_agent_concepts' (opcional) es una lista de clases de concepto por
    agente, usada por la ampliación de parejas heterogéneas.
    """
    cfg = dict(BASE)
    cfg.update({
        "size": size,
        "obstacle_density": density,
        "epochs": epochs,
        "epsilon_max": epsilon_max,
        "gamma": gamma,
        "renders": renders_dir,
        "algorithm_cls": algorithm_cls,
        "solution_concept": solution_concept_cls,
        "algorithm_kwargs": {"gamma": gamma},
    })
    if per_agent_concepts is not None:
        cfg["per_agent_concepts"] = per_agent_concepts
    return cfg


def create_env_for(config, seed):
    """Crea un entorno POGEMA con la densidad/tamaño del config y la seed dada."""
    grid_config = GridConfig(num_agents=config["num_agents"],
                             size=config["size"],
                             density=config["obstacle_density"],
                             seed=seed,
                             max_episode_steps=config["episode_length"],
                             obs_radius=1,
                             on_target="finish",
                             render_mode=None)
    anim = make_anim_config(config["renders"], save_every=config["save_every"])
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=anim)
    return RewardWrapper(env)


def build_agents(config, game, master_seed):
    """Construye los algoritmos para un run.

    Caso normal: delega en build_algorithms de main.py, inyectando base_seed
    para que cada run explore distinto (varianza inter-seed significativa).

    Caso heterogéneo (ampliación): si el config trae 'per_agent_concepts', crea
    cada agente con un concepto de solución distinto. Lo construye a mano porque
    build_algorithms aplica el mismo concepto a todos los agentes.
    """
    config["base_seed"] = master_seed
    per_agent = config.get("per_agent_concepts")
    if per_agent is None:
        return build_algorithms(config, game)

    # Construcción manual con un concepto distinto por agente
    algorithm_cls = config["algorithm_cls"]
    gamma = config["algorithm_kwargs"].get("gamma", 0.95)
    algorithms = []
    for agent_id in range(game.num_agents):
        concept_cls = per_agent[agent_id]
        algorithms.append(algorithm_cls(
            agent_id, game,
            solution_concept=concept_cls(),
            gamma=gamma,
            epsilon=config["epsilon_max"],
            alpha=config["learning_rate"],
            seed=master_seed * 100 + agent_id))
    return algorithms


# ============================================================================
#  MÉTRICAS Y NÚCLEO DE ENTRENAMIENTO
# ============================================================================

def convergence_epoch(rewards, window=20):
    """Métrica de convergencia rigurosa.

    El epoch de convergencia es el primero donde la media móvil (ventana 'window')
    supera el percentil 90 de la recompensa del run. Usamos percentil 90 en lugar
    del máximo absoluto para no dispararnos con un pico espurio.
    """
    rewards = np.asarray(rewards, dtype=float)
    if len(rewards) < window:
        return len(rewards) - 1, float(np.mean(rewards))
    p90 = np.percentile(rewards, 90)
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
    idx = np.argmax(moving >= p90)
    if moving[idx] < p90:
        conv_epoch = len(rewards) - window
    else:
        conv_epoch = idx
    conv_reward = float(np.mean(rewards[conv_epoch:]))
    return int(conv_epoch), conv_reward


def evaluate_on_seeds(config, algorithms, game, seeds):
    """Evalúa una política ya entrenada sobre un conjunto de seeds dado.

    Usado por la ampliación de generalización: medir rendimiento en mapas no
    vistos sin entrenar de nuevo. Devuelve recompensa colectiva e individual
    promediadas sobre las seeds.
    """
    collective, individual = 0.0, [0.0] * config["num_agents"]
    for ep in seeds:
        env = create_env_for(config, seed=ep)
        rewards = evaluate_episode(env, algorithms, game)
        collective += sum(rewards)
        for i in range(config["num_agents"]):
            individual[i] += rewards[i]
    n = len(seeds)
    return collective / n, [x / n for x in individual]


def run_training(config, seeds, run_name, save_svgs=False, svg_concept_name=None,
                 svg_maps=(0, 1, 2)):
    """Entrena y evalúa UN run completo (una configuración, varias seeds).

    Reutiliza train_episode y evaluate_episode de main.py sin modificarlos.
    Promedia las métricas sobre todas las seeds. Devuelve un dict con métricas
    agregadas y deja persistido el CSV por-epoch.
    """
    num_agents = config["num_agents"]
    epochs = config["epochs"]

    per_seed_collective, per_seed_td = [], []
    per_seed_individual, per_seed_time = [], []
    final_algorithms = None

    for master_seed in seeds:
        np.random.seed(master_seed)
        game = GameModel(num_agents=num_agents,
                         num_states=config["num_states"], num_actions=5)
        algorithms = build_agents(config, game, master_seed)

        collective_hist, td_hist, individual_hist = [], [], []
        t0 = time.time()
        for epoch in range(epochs):
            # --- Entrenamiento ---
            all_td = []
            for ep in range(config["episodes_per_epoch"]):
                global_episode = epoch * config["episodes_per_epoch"] + ep
                epsilon = compute_epsilon(config, global_episode)
                env = create_env_for(config, seed=ep % config["maps"])
                _, td_errors = train_episode(env, algorithms, game, epsilon)
                all_td.extend(td_errors)
            td_hist.append(float(np.sum(np.abs(all_td))))

            # --- Evaluación ---
            eval_collective = 0.0
            eval_individual = [0.0] * num_agents
            for ep in range(config["maps"]):
                env = create_env_for(config, seed=ep)
                total_rewards = evaluate_episode(env, algorithms, game)
                eval_collective += sum(total_rewards)
                for i in range(num_agents):
                    eval_individual[i] += total_rewards[i]
            collective_hist.append(eval_collective)
            individual_hist.append(eval_individual)

        per_seed_collective.append(collective_hist)
        per_seed_td.append(td_hist)
        per_seed_individual.append(individual_hist)
        per_seed_time.append(time.time() - t0)
        final_algorithms = algorithms

    # --- Agregación sobre seeds ---
    collective_arr = np.array(per_seed_collective)
    td_arr = np.array(per_seed_td)
    individual_arr = np.array(per_seed_individual)

    mean_collective = collective_arr.mean(axis=0)
    std_collective = collective_arr.std(axis=0)
    mean_td = td_arr.mean(axis=0)

    conv_epochs, conv_rewards = [], []
    for s in range(len(seeds)):
        ce, cr = convergence_epoch(per_seed_collective[s])
        conv_epochs.append(ce)
        conv_rewards.append(cr)

    # Inestabilidad intra-run (firma de no-estacionariedad)
    intra_run_std = []
    for s in range(len(seeds)):
        r = np.asarray(per_seed_collective[s], dtype=float)
        if len(r) >= 20:
            half = len(r) // 2
            windows = [np.std(r[i:i + 20]) for i in range(half, len(r) - 20)]
            intra_run_std.append(float(np.mean(windows)) if windows else 0.0)
        else:
            intra_run_std.append(float(np.std(r)))

    # --- Persistir CSV por-epoch ---
    df = pd.DataFrame({
        "epoch": range(epochs),
        "collective_reward_mean": mean_collective,
        "collective_reward_std": std_collective,
        "td_error_mean": mean_td,
    })
    for i in range(num_agents):
        df[f"individual_reward_agent{i}_mean"] = individual_arr.mean(axis=0)[:, i]
    csv_path = os.path.join(RESULTS_DIR, f"{run_name}.csv")
    df.to_csv(csv_path, index=False)

    if save_svgs and final_algorithms is not None:
        save_final_svgs(config, final_algorithms, svg_concept_name or run_name,
                        svg_maps)

    return {
        "run_name": run_name,
        "mean_collective": mean_collective,
        "std_collective": std_collective,
        "mean_td": mean_td,
        "final_collective": float(mean_collective[-1]),
        "conv_epoch_mean": float(np.mean(conv_epochs)),
        "conv_reward_mean": float(np.mean(conv_rewards)),
        "inter_seed_std_final": float(std_collective[-1]),
        "intra_run_std_mean": float(np.mean(intra_run_std)),
        "train_time_mean": float(np.mean(per_seed_time)),
        "individual_final": individual_arr.mean(axis=0)[-1].tolist(),
        "csv_path": csv_path,
        "final_algorithms": final_algorithms,
    }


def save_final_svgs(config, algorithms, concept_name, svg_maps):
    """Genera los SVGs de evaluación greedy con vista GLOBAL para unos mapas.

    Vista global (egocentric_idx=None), un único SVG por mapa, para que ambos
    agentes queden en el MISMO sistema de coordenadas. Es lo que permite detectar
    colisiones, intercambios y cesiones de paso reales en el análisis posterior.
    """
    num_agents = config["num_agents"]
    game = GameModel(num_agents=num_agents,
                     num_states=config["num_states"], num_actions=5)
    for ep in svg_maps:
        env = create_env_for(config, seed=ep)
        evaluate_episode(env, algorithms, game)
        fname = os.path.join(
            config["renders"],
            f"{concept_name}-size{config['size']}-d{config['obstacle_density']}"
            f"-map{ep}.svg")
        try:
            env.save_animation(fname, make_save_anim_config(None))
        except Exception as e:
            log(f"[aviso] no se pudo guardar SVG {fname}: {e}", 2)


def save_qtables(algorithms, run_name):
    """Persiste las Q-tables de los algoritmos de un run (ampliación generalización)."""
    os.makedirs(QTABLES_DIR, exist_ok=True)
    for i, algo in enumerate(algorithms):
        path = os.path.join(QTABLES_DIR, f"{run_name}_agent{i}.npy")
        np.save(path, algo.q_table)
# ============================================================================
#  EXPERIMENTO 1 -- Estudio conjunto de (epsilon, gamma)
# ============================================================================

def experiment_1_hyperparams(epochs):
    banner("EXPERIMENTO 1 -- Estudio conjunto de (epsilon, gamma)")
    log("Objetivo: caracterizar el efecto de epsilon y gamma y fijar el par óptimo")
    log("Algoritmo: JAL-GT (Pareto) | Mapa: 4x4 | Densidad: 0.1 | Seeds: 0..9")
    log("Hipótesis:")
    log("- epsilon alto reduce la varianza inter-seed (mejor cobertura inicial).", 1)
    log("- gamma alto es necesario para propagar la recompensa terminal (objetivo", 1)
    log("  a varios pasos). El step penalty refuerza esta preferencia.", 1)
    log(f"Grid: epsilon in {EPSILON_GRID} x gamma in {GAMMA_GRID}  "
        f"({len(EPSILON_GRID)*len(GAMMA_GRID)} runs)")

    conv_matrix = np.zeros((len(EPSILON_GRID), len(GAMMA_GRID)))
    std_matrix = np.zeros((len(EPSILON_GRID), len(GAMMA_GRID)))
    all_runs = []

    for i, eps in enumerate(EPSILON_GRID):
        for j, gamma in enumerate(GAMMA_GRID):
            run_name = f"exp1_eps{eps}_gamma{gamma}"
            log(f"\n>> Run: epsilon_max={eps}, gamma={gamma}", 1)
            cfg = build_config(size=4, density=DENSITY_LOW, epochs=epochs,
                               epsilon_max=eps, gamma=gamma,
                               algorithm_cls=JALGT,
                               solution_concept_cls=ParetoSolutionConcept,
                               renders_dir=RENDERS_DIR)
            res = run_training(cfg, TRAIN_SEEDS, run_name)
            conv_matrix[i, j] = res["conv_reward_mean"]
            std_matrix[i, j] = res["inter_seed_std_final"]
            all_runs.append(res)
            log(f"   recompensa en convergencia = {res['conv_reward_mean']:.3f} "
                f"| std inter-seed = {res['inter_seed_std_final']:.3f}", 1)

    best_idx = np.unravel_index(np.argmax(conv_matrix), conv_matrix.shape)
    best_eps = EPSILON_GRID[best_idx[0]]
    best_gamma = GAMMA_GRID[best_idx[1]]

    plot_heatmap(conv_matrix,
                 [f"eps={e}" for e in EPSILON_GRID],
                 [f"gamma={g}" for g in GAMMA_GRID],
                 "EXP1: Recompensa en convergencia (JAL-GT Pareto, 4x4)",
                 "exp1_heatmap_reward.png",
                 row_name="epsilon_max", col_name="gamma")
    plot_heatmap(std_matrix,
                 [f"eps={e}" for e in EPSILON_GRID],
                 [f"gamma={g}" for g in GAMMA_GRID],
                 "EXP1: Varianza inter-seed final (menor es mejor)",
                 "exp1_heatmap_std.png",
                 row_name="epsilon_max", col_name="gamma", fmt=".3f")

    log(f"\n>> CONCLUSIÓN EXP1: par óptimo encontrado = "
        f"(epsilon_max={best_eps}, gamma={best_gamma})")
    log("   Este par se trasladará a los experimentos siguientes.")

    save_json({
        "experiment": "exp1_hyperparams",
        "grid_epsilon": EPSILON_GRID, "grid_gamma": GAMMA_GRID,
        "conv_reward_matrix": conv_matrix.tolist(),
        "inter_seed_std_matrix": std_matrix.tolist(),
        "best_epsilon": best_eps, "best_gamma": best_gamma,
        "runs": [{k: v for k, v in r.items()
                  if k not in ("final_algorithms", "mean_collective",
                               "std_collective", "mean_td")} for r in all_runs],
    }, "exp1_summary.json")

    return best_eps, best_gamma


# ============================================================================
#  EXPERIMENTO 2 -- Transferencia de gamma a IQL
# ============================================================================

def experiment_2_iql_transfer(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 2 -- Transferencia de gamma a IQL")
    log("Objetivo: verificar que el gamma óptimo de JAL-GT es razonable para IQL")
    log(f"Algoritmo: IQL | epsilon fijo = {best_eps} | Mapa 4x4 | Densidad 0.1")
    log("Hipótesis: IQL tiene dinámica más estable (5 vs 25 valores por estado),")
    log("por lo que su gamma óptimo debería coincidir o ser cercano al de JAL-GT.", 1)

    results = {}
    for gamma in GAMMA_GRID:
        run_name = f"exp2_iql_gamma{gamma}"
        log(f"\n>> Run IQL: gamma={gamma}", 1)
        cfg = build_config(size=4, density=DENSITY_LOW, epochs=epochs,
                           epsilon_max=best_eps, gamma=gamma,
                           algorithm_cls=IQL, solution_concept_cls=None,
                           renders_dir=RENDERS_DIR)
        res = run_training(cfg, TRAIN_SEEDS, run_name)
        results[gamma] = res["conv_reward_mean"]
        log(f"   recompensa en convergencia = {res['conv_reward_mean']:.3f}", 1)

    best_gamma_iql = max(results, key=results.get)
    plot_bars([f"gamma={g}" for g in GAMMA_GRID],
              [results[g] for g in GAMMA_GRID],
              "Recompensa en convergencia", "EXP2: gamma en IQL (4x4)",
              "exp2_iql_gamma.png")

    if best_gamma_iql == best_gamma:
        decision = (f"El gamma óptimo de IQL ({best_gamma_iql}) COINCIDE con el de "
                    f"JAL-GT ({best_gamma}). Usamos el mismo par: fairness perfecta.")
    else:
        decision = (f"El gamma óptimo de IQL ({best_gamma_iql}) DIFIERE del de JAL-GT "
                    f"({best_gamma}). Por fairness usamos el de JAL-GT en la comparativa, "
                    f"pero documentamos este hallazgo.")
    log(f"\n>> CONCLUSIÓN EXP2: {decision}")

    save_json({"experiment": "exp2_iql_transfer", "epsilon_used": best_eps,
               "iql_conv_by_gamma": results, "best_gamma_iql": best_gamma_iql,
               "best_gamma_jalgt": best_gamma, "decision": decision},
              "exp2_summary.json")
    return best_gamma_iql


# ============================================================================
#  EXPERIMENTO 3 -- Validación del par óptimo en 10x10
# ============================================================================

def experiment_3_validate_large(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 3 -- Validación del par óptimo en mapa grande (10x10)")
    log("Objetivo: comprobar si el par óptimo de 4x4 transfiere a 10x10")
    log(f"Comparamos (eps={best_eps}, gamma={best_gamma}) contra el baseline (1.0, 0.95)")
    log("Hipótesis: los mapas grandes podrían necesitar más exploración; el par")
    log("óptimo de 4x4 podría no ser óptimo en 10x10.", 1)

    runs, results = [], {}
    for label, eps, gamma in [("optimo", best_eps, best_gamma),
                              ("baseline", 1.0, 0.95)]:
        run_name = f"exp3_{label}"
        log(f"\n>> Run {label}: epsilon={eps}, gamma={gamma}", 1)
        cfg = build_config(size=10, density=DENSITY_LOW, epochs=epochs,
                           epsilon_max=eps, gamma=gamma, algorithm_cls=JALGT,
                           solution_concept_cls=ParetoSolutionConcept,
                           renders_dir=RENDERS_DIR)
        res = run_training(cfg, TRAIN_SEEDS, run_name)
        results[label] = res["conv_reward_mean"]
        runs.append({"label": f"{label} (eps={eps},g={gamma})",
                     "mean": res["mean_collective"], "std": res["std_collective"]})
        log(f"   recompensa en convergencia = {res['conv_reward_mean']:.3f}", 1)

    plot_curve(runs, "Recompensa colectiva",
               "EXP3: par óptimo vs baseline en 10x10", "exp3_validation_10x10.png")

    if results["optimo"] >= results["baseline"]:
        decision = (f"El par óptimo se mantiene mejor o igual en 10x10 "
                    f"({results['optimo']:.3f} vs {results['baseline']:.3f}). "
                    f"Confirmada la transferencia a mapas grandes.")
    else:
        decision = (f"El par óptimo de 4x4 NO supera al baseline en 10x10 "
                    f"({results['optimo']:.3f} vs {results['baseline']:.3f}). "
                    f"Hallazgo: los mapas grandes requieren más exploración.")
    log(f"\n>> CONCLUSIÓN EXP3: {decision}")
    save_json({"experiment": "exp3_validate_large", "results": results,
               "decision": decision}, "exp3_summary.json")


# ============================================================================
#  EXPERIMENTOS 4 y 5 -- IQL vs JAL-GT (comparativa principal)
# ============================================================================

def experiment_4_5_iql_vs_jalgt(best_eps, best_gamma, epochs):
    results_by_density = {}
    for density, exp_id in [(DENSITY_LOW, "exp4"), (DENSITY_MID, "exp5")]:
        banner(f"EXPERIMENTO {'4' if density == DENSITY_LOW else '5'} -- "
               f"IQL vs JAL-GT (densidad {density})")
        log("Objetivo: comparativa algorítmica principal (recompensa individual y")
        log("colectiva, estabilidad, efecto de la no-estacionariedad sobre IQL).")
        log(f"Tamaños: {SIZES} | epsilon={best_eps} | gamma={best_gamma}")
        log("Hipótesis:")
        log("- JAL-GT supera a IQL en recompensa colectiva (modela coordinación).", 1)
        log("- IQL muestra MAYOR inestabilidad intra-run: firma de no-estacionariedad.", 1)
        log("- La brecha crece con el tamaño del mapa y con la densidad.", 1)

        density_summary = {}
        for size in SIZES:
            log(f"\n--- Tamaño {size}x{size} ---", 1)
            runs_for_plot, td_runs_for_plot, size_data = [], [], {}

            for algo_label, algo_cls, concept in [
                ("IQL", IQL, None),
                ("JAL-GT", JALGT, ParetoSolutionConcept)]:
                run_name = f"{exp_id}_{algo_label}_size{size}"
                save_svgs = (density == DENSITY_MID and size == 6)
                cfg = build_config(size=size, density=density, epochs=epochs,
                                   epsilon_max=best_eps, gamma=best_gamma,
                                   algorithm_cls=algo_cls, solution_concept_cls=concept,
                                   renders_dir=RENDERS_DIR)
                res = run_training(cfg, TRAIN_SEEDS, run_name, save_svgs=save_svgs,
                                   svg_concept_name=f"{exp_id}_{algo_label}")
                runs_for_plot.append({"label": algo_label,
                                      "mean": res["mean_collective"],
                                      "std": res["std_collective"]})
                td_runs_for_plot.append({"label": algo_label, "mean": res["mean_td"]})
                size_data[algo_label] = {
                    "final_collective": res["final_collective"],
                    "conv_reward": res["conv_reward_mean"],
                    "intra_run_std": res["intra_run_std_mean"],
                    "inter_seed_std": res["inter_seed_std_final"],
                    "train_time": res["train_time_mean"],
                    "individual_final": res["individual_final"],
                }
                log(f"  {algo_label}: recompensa final={res['final_collective']:.3f} "
                    f"| convergencia={res['conv_reward_mean']:.3f} "
                    f"| inestabilidad intra-run={res['intra_run_std_mean']:.3f} "
                    f"| tiempo={res['train_time_mean']:.1f}s", 1)

            plot_curve(runs_for_plot, "Recompensa colectiva",
                       f"{exp_id.upper()}: IQL vs JAL-GT recompensa ({size}x{size}, d={density})",
                       f"{exp_id}_reward_size{size}.png")
            plot_curve(td_runs_for_plot, "TD error (suma abs por epoch)",
                       f"{exp_id.upper()}: IQL vs JAL-GT TD error ({size}x{size}, d={density})",
                       f"{exp_id}_tderror_size{size}.png", show_band=False)

            jal, iql = size_data["JAL-GT"], size_data["IQL"]
            gap = jal["conv_reward"] - iql["conv_reward"]
            log(f"  -> brecha JAL-GT - IQL (convergencia) = {gap:+.3f}", 1)
            if iql["intra_run_std"] > jal["intra_run_std"]:
                log(f"  -> IQL más inestable que JAL-GT (no-estacionariedad confirmada): "
                    f"{iql['intra_run_std']:.3f} > {jal['intra_run_std']:.3f}", 1)
            else:
                log(f"  -> IQL NO más inestable que JAL-GT aquí "
                    f"({iql['intra_run_std']:.3f} <= {jal['intra_run_std']:.3f})", 1)
            density_summary[f"size{size}"] = size_data

        if density == DENSITY_MID:
            for algo_label in ("IQL", "JAL-GT"):
                summary = analyze_svgs(
                    os.path.join(RENDERS_DIR, f"{exp_id}_{algo_label}-size6-*.svg"),
                    os.path.join(SVG_ANALYSIS_DIR, f"{exp_id}_{algo_label}_svg.csv"),
                    size=6, density=density)
                report_svg_findings(f"{exp_id} {algo_label} (6x6, d={density})", summary)

        results_by_density[exp_id] = density_summary
        save_json({"experiment": exp_id, "density": density, "epsilon": best_eps,
                   "gamma": best_gamma, "data": density_summary}, f"{exp_id}_summary.json")
    return results_by_density


# ============================================================================
#  EXPERIMENTO 6 -- Conceptos de solución (comparación central, 6x6)
# ============================================================================

def experiment_6_concepts(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 6 -- Conceptos de solución en JAL-GT (6x6)")
    log("Objetivo: comparar Nash/Pareto/Welfare/Minimax: comportamiento inducido,")
    log("coordinación y recompensas. Dos densidades para contraste situacional.")
    log(f"epsilon={best_eps} | gamma={best_gamma}")
    log("Hipótesis:")
    log("- Pareto y Welfare: mejores en entorno cooperativo (pueden coincidir).", 1)
    log("- Nash: algo peor, riesgo de deadlock si 'ambos quietos' es equilibrio.", 1)
    log("- Minimax: el peor, trata al colaborador como adversario.", 1)

    summary = {}
    for density in [DENSITY_LOW, DENSITY_MID]:
        log(f"\n--- Densidad {density} ---", 1)
        runs_for_plot, bar_labels, bar_values, density_data = [], [], [], {}

        for concept_name, concept_cls in SOLUTION_CONCEPTS.items():
            run_name = f"exp6_{concept_name}_d{density}"
            save_svgs = (density == DENSITY_MID)
            cfg = build_config(size=6, density=density, epochs=epochs,
                               epsilon_max=best_eps, gamma=best_gamma,
                               algorithm_cls=JALGT, solution_concept_cls=concept_cls,
                               renders_dir=RENDERS_DIR)
            res = run_training(cfg, TRAIN_SEEDS, run_name, save_svgs=save_svgs,
                               svg_concept_name=f"exp6_{concept_name}")
            runs_for_plot.append({"label": concept_name,
                                  "mean": res["mean_collective"],
                                  "std": res["std_collective"]})
            bar_labels.append(concept_name)
            bar_values.append(res["conv_reward_mean"])
            density_data[concept_name] = {"conv_reward": res["conv_reward_mean"],
                                          "final_collective": res["final_collective"],
                                          "train_time": res["train_time_mean"]}
            log(f"  {concept_name}: convergencia={res['conv_reward_mean']:.3f} "
                f"| tiempo={res['train_time_mean']:.1f}s", 1)

        plot_curve(runs_for_plot, "Recompensa colectiva",
                   f"EXP6: conceptos de solución (6x6, d={density})",
                   f"exp6_concepts_d{density}.png")
        plot_bars(bar_labels, bar_values, "Recompensa en convergencia",
                  f"EXP6: ranking de conceptos (6x6, d={density})",
                  f"exp6_concepts_bars_d{density}.png")

        if density == DENSITY_MID:
            for concept_name in SOLUTION_CONCEPTS:
                s = analyze_svgs(
                    os.path.join(RENDERS_DIR, f"exp6_{concept_name}-size6-*.svg"),
                    os.path.join(SVG_ANALYSIS_DIR, f"exp6_{concept_name}_svg.csv"),
                    size=6, density=density)
                report_svg_findings(f"EXP6 {concept_name} (6x6, d={density})", s)

        ranking = sorted(density_data.items(),
                         key=lambda kv: kv[1]["conv_reward"], reverse=True)
        log(f"  -> Ranking (d={density}): " +
            " > ".join(f"{n}({d['conv_reward']:.2f})" for n, d in ranking), 1)
        summary[f"density{density}"] = density_data

    save_json({"experiment": "exp6_concepts", "data": summary}, "exp6_summary.json")
    return summary


# ============================================================================
#  EXPERIMENTO 7 -- Conceptos contrastantes en tamaños extremos
# ============================================================================

def experiment_7_concepts_scaling(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 7 -- Contraste situacional Pareto vs Minimax (4x4 y 10x10)")
    log("Objetivo: sustentar 'en qué situaciones es más adecuado cada concepto'")
    log("añadiendo variación de tamaño barata (los dos conceptos más contrastantes).")
    log(f"epsilon={best_eps} | gamma={best_gamma} | densidad={DENSITY_LOW}")
    log("Hipótesis: el ranking puede depender del tamaño; en mapas pequeños las")
    log("diferencias se diluyen, en grandes se amplifican.", 1)

    summary = {}
    for size in [4, 10]:
        log(f"\n--- Tamaño {size}x{size} ---", 1)
        bar_labels, bar_values = [], []
        for concept_name in ("Pareto", "Minimax"):
            run_name = f"exp7_{concept_name}_size{size}"
            cfg = build_config(size=size, density=DENSITY_LOW, epochs=epochs,
                               epsilon_max=best_eps, gamma=best_gamma,
                               algorithm_cls=JALGT,
                               solution_concept_cls=SOLUTION_CONCEPTS[concept_name],
                               renders_dir=RENDERS_DIR)
            res = run_training(cfg, TRAIN_SEEDS, run_name)
            bar_labels.append(concept_name)
            bar_values.append(res["conv_reward_mean"])
            summary[f"{concept_name}_size{size}"] = res["conv_reward_mean"]
            log(f"  {concept_name}: convergencia={res['conv_reward_mean']:.3f}", 1)
        plot_bars(bar_labels, bar_values, "Recompensa en convergencia",
                  f"EXP7: Pareto vs Minimax ({size}x{size})",
                  f"exp7_pareto_minimax_size{size}.png")

    gap_small = summary["Pareto_size4"] - summary["Minimax_size4"]
    gap_large = summary["Pareto_size10"] - summary["Minimax_size10"]
    log(f"\n>> CONCLUSIÓN EXP7: brecha Pareto-Minimax en 4x4 = {gap_small:+.3f}, "
        f"en 10x10 = {gap_large:+.3f}")
    if abs(gap_large) > abs(gap_small):
        log("   La diferencia entre conceptos SE AMPLIFICA con el tamaño.")
    else:
        log("   La diferencia entre conceptos NO se amplifica con el tamaño.")
    save_json({"experiment": "exp7_concepts_scaling", "data": summary,
               "gap_small": gap_small, "gap_large": gap_large}, "exp7_summary.json")
    return summary


# ============================================================================
#  EXPERIMENTO 8 -- Decay exponencial vs lineal de epsilon  (+1)
# ============================================================================
#
#  El baseline usa decay LINEAL de epsilon (compute_epsilon de main.py).
#  Implementamos decay EXPONENCIAL y comparamos ambas estrategias de forma
#  apples-to-apples: mismo epsilon_max, mismo epsilon_min, mismo nº de episodios.
#  Solo cambia la FORMA de la curva. La k del exponencial se calibra para que
#  la curva alcance epsilon_min al final del entrenamiento.
#
#  Para inyectar el decay exponencial sin tocar main.py, parcheamos en caliente
#  la función compute_epsilon que usa train_episode (monkeypatch acotado y
#  reversible). Es la forma menos invasiva de cambiar SOLO la regla de decay.
# ============================================================================

def _exponential_epsilon(config, global_episode):
    """Decay exponencial: eps_t = eps_min + (eps_max-eps_min) * exp(-k t).

    k se elige para que en el último episodio el término exponencial sea ~1%
    del rango, es decir la curva llega prácticamente a eps_min al final.
    """
    total = config["epochs"] * config["episodes_per_epoch"]
    k = -math.log(0.01) / max(total - 1, 1)   # exp(-k*(total-1)) = 0.01
    eps_max, eps_min = config["epsilon_max"], config["epsilon_min"]
    return eps_min + (eps_max - eps_min) * math.exp(-k * global_episode)


def _run_with_decay(config, seeds, run_name, decay_fn):
    """Versión de run_training que usa una función de decay concreta.

    run_training (en este mismo módulo) llama a compute_epsilon, que importamos
    arriba. Para cambiar SOLO la regla de decay sin tocar el bucle, parcheamos
    temporalmente la referencia global de este módulo, restaurándola al terminar.
    """
    global compute_epsilon
    original = compute_epsilon
    compute_epsilon = decay_fn
    try:
        return run_training(config, seeds, run_name)
    finally:
        compute_epsilon = original


def experiment_8_decay(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 8 (+1) -- Decay exponencial vs lineal de epsilon")
    log("Objetivo: mejora metodológica sobre el decay de EPSILON (no alpha).")
    log("Comparamos el decay lineal del baseline contra un decay exponencial,")
    log("apples-to-apples (mismo eps_max, eps_min y nº de episodios).")
    log(f"epsilon_max={best_eps} | gamma={best_gamma} | densidad={DENSITY_LOW}")
    log("Hipótesis: el exponencial concentra exploración al inicio y cae más")
    log("rápido a explotación; debería converger antes en mapas medianos (6x6),", 1)
    log("con diferencia marginal en 4x4.", 1)

    summary = {}
    for size in [4, 6]:
        log(f"\n--- Tamaño {size}x{size} ---", 1)
        for algo_label, algo_cls, concept in [
            ("IQL", IQL, None),
            ("JAL-GT", JALGT, ParetoSolutionConcept)]:
            runs_for_plot = []
            for decay_label, decay_fn in [("lineal", compute_epsilon),
                                          ("exponencial", _exponential_epsilon)]:
                run_name = f"exp8_{algo_label}_size{size}_{decay_label}"
                cfg = build_config(size=size, density=DENSITY_LOW, epochs=epochs,
                                   epsilon_max=best_eps, gamma=best_gamma,
                                   algorithm_cls=algo_cls,
                                   solution_concept_cls=concept,
                                   renders_dir=RENDERS_DIR)
                res = _run_with_decay(cfg, TRAIN_SEEDS, run_name, decay_fn)
                runs_for_plot.append({"label": f"{decay_label}",
                                      "mean": res["mean_collective"],
                                      "std": res["std_collective"]})
                summary[f"{algo_label}_size{size}_{decay_label}"] = res["conv_reward_mean"]
                log(f"  {algo_label} {decay_label}: "
                    f"convergencia={res['conv_reward_mean']:.3f} "
                    f"(epoch conv={res['conv_epoch_mean']:.0f})", 1)
            plot_curve(runs_for_plot, "Recompensa colectiva",
                       f"EXP8: decay lineal vs exponencial ({algo_label}, {size}x{size})",
                       f"exp8_{algo_label}_size{size}.png")

    # Perfil de epsilon de ambas estrategias (apples-to-apples)
    dummy = build_config(size=4, density=DENSITY_LOW, epochs=epochs,
                         epsilon_max=best_eps, gamma=best_gamma,
                         algorithm_cls=IQL, solution_concept_cls=None,
                         renders_dir=RENDERS_DIR)
    total = epochs * dummy["episodes_per_epoch"]
    eps_lin = [compute_epsilon(dummy, t) for t in range(total)]
    eps_exp = [_exponential_epsilon(dummy, t) for t in range(total)]
    plot_curve([{"label": "lineal", "mean": eps_lin},
                {"label": "exponencial", "mean": eps_exp}],
               "epsilon", "EXP8: perfil de epsilon(t) (apples-to-apples)",
               "exp8_epsilon_profile.png", show_band=False)

    log("\n>> CONCLUSIÓN EXP8: comparar epoch de convergencia entre estrategias.")
    log("   Si el exponencial converge antes en 6x6, confirma la hipótesis.")
    save_json({"experiment": "exp8_decay", "data": summary}, "exp8_summary.json")
    return summary


# ============================================================================
#  EXPERIMENTO 9 -- Generalización a mapas no vistos  (+0.5)
# ============================================================================
#
#  Entrenamos en seeds 0..9 y evaluamos en seeds 100..109 (nunca vistas).
#  Métrica núcleo: cobertura de estados (qué fracción de los estados vistos en
#  test ya se vieron en train). Como obs_to_state es independiente de la posición
#  absoluta (radio 1), los mapas comparten muchos estados; la cobertura explica
#  si la generalización es trivial o si los mapas grandes generan estados nuevos.
# ============================================================================

def _state_coverage(config, algorithms, game, train_seeds, test_seeds):
    """Fracción de estados visitados en test que también se visitaron en train."""
    # obs_to_state y create_env_for ya están disponibles en este módulo.

    def visited_states(seeds):
        seen = set()
        for ep in seeds:
            env = create_env_for(config, seed=ep)
            obs, _ = env.reset()
            states = [obs_to_state(obs[i]) for i in range(game.num_agents)]
            seen.update(states)
            term = [False] * game.num_agents
            trunc = [False] * game.num_agents
            while not all(term) and not all(trunc):
                actions = tuple(algorithms[i].select_action(states[i], train=False)
                                for i in range(game.num_agents))
                obs, _, term, trunc, _ = env.step(actions)
                states = [obs_to_state(obs[i]) for i in range(game.num_agents)]
                seen.update(states)
        return seen

    train_states = visited_states(train_seeds)
    test_states = visited_states(test_seeds)
    if not test_states:
        return 0.0
    overlap = len(test_states & train_states) / len(test_states)
    return overlap


def experiment_9_generalization(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 9 (+0.5) -- Generalización a mapas no vistos")
    log("Objetivo: medir rendimiento de políticas entrenadas en mapas NUEVOS.")
    log(f"Train seeds: {TRAIN_SEEDS[:3]}... | Test seeds: {UNSEEN_SEEDS[:3]}...")
    log("Hipótesis: como obs_to_state es independiente de la posición absoluta,")
    log("los mapas comparten muchos estados y esperamos buena generalización.", 1)
    log("La cobertura de estados es la clave: si es alta, generalizar es trivial.", 1)

    summary = {}
    for algo_label, algo_cls, concept in [
        ("IQL", IQL, None),
        ("JAL-GT", JALGT, ParetoSolutionConcept)]:
        for density in [DENSITY_LOW, DENSITY_MID]:
            for size in [4, 6, 10]:
                run_name = f"exp9_{algo_label}_size{size}_d{density}"
                cfg = build_config(size=size, density=density, epochs=epochs,
                                   epsilon_max=best_eps, gamma=best_gamma,
                                   algorithm_cls=algo_cls,
                                   solution_concept_cls=concept,
                                   renders_dir=RENDERS_DIR)
                # Entrenamos un run (1 seed para abaratar; generalización no
                # necesita estadística inter-seed, sino el contraste train/test)
                res = run_training(cfg, [0], run_name)
                algorithms = res["final_algorithms"]
                game = GameModel(num_agents=cfg["num_agents"],
                                 num_states=cfg["num_states"], num_actions=5)
                # Recompensa en train-seeds vs test-seeds
                r_train, _ = evaluate_on_seeds(cfg, algorithms, game, TRAIN_SEEDS)
                r_test, _ = evaluate_on_seeds(cfg, algorithms, game, UNSEEN_SEEDS)
                coverage = _state_coverage(cfg, algorithms, game,
                                           TRAIN_SEEDS, UNSEEN_SEEDS)
                gap = r_train - r_test
                summary[run_name] = {"reward_train": r_train, "reward_test": r_test,
                                     "gap": gap, "state_coverage": coverage}
                log(f"  {algo_label} {size}x{size} d={density}: "
                    f"train={r_train:.2f} test={r_test:.2f} gap={gap:+.2f} "
                    f"cobertura={coverage:.1%}", 1)

    # Plot: gap de generalización por configuración
    labels = list(summary.keys())
    plot_grouped_bars(
        [l.replace("exp9_", "") for l in labels],
        {"train": [summary[l]["reward_train"] for l in labels],
         "test": [summary[l]["reward_test"] for l in labels]},
        "Recompensa", "EXP9: generalización train vs test (mapas no vistos)",
        "exp9_generalization.png")

    log("\n>> CONCLUSIÓN EXP9: si la cobertura de estados es alta, la generalización")
    log("   es casi trivial (consecuencia de la representación local de obs_to_state).")
    log("   Donde la cobertura baja (mapas grandes/densos), el gap train-test crece.")
    save_json({"experiment": "exp9_generalization", "data": summary},
              "exp9_summary.json")
    return summary


# ============================================================================
#  EXPERIMENTO 10 -- Parejas heterogéneas de conceptos de solución  (+0.5)
# ============================================================================
#
#  El enunciado pide entrenar parejas con CONCEPTOS DE SOLUCIÓN distintos por
#  agente (ambos JAL-GT, uno Nash y otro Pareto), comparadas contra parejas
#  homogéneas. NO es mezclar algoritmos (IQL+JAL-GT), sino conceptos.
# ============================================================================

def experiment_10_heterogeneous(best_eps, best_gamma, epochs):
    banner("EXPERIMENTO 10 (+0.5) -- Parejas heterogéneas de conceptos de solución")
    log("Objetivo: entrenar parejas JAL-GT con conceptos DISTINTOS por agente")
    log("(Nash + Pareto) y compararlas contra homogéneas (Nash+Nash, Pareto+Pareto).")
    log(f"epsilon={best_eps} | gamma={best_gamma}")
    log("Hipótesis: un agente que busca el óptimo de Pareto conviviendo con uno")
    log("que busca el equilibrio de Nash genera conflicto de incentivos; esperamos", 1)
    log("rendimiento intermedio o inferior al de la mejor pareja homogénea.", 1)

    configs = {
        "Nash+Nash": [NashSolutionConcept, NashSolutionConcept],
        "Pareto+Pareto": [ParetoSolutionConcept, ParetoSolutionConcept],
        "Nash+Pareto": [NashSolutionConcept, ParetoSolutionConcept],
    }

    summary = {}
    for density in [DENSITY_LOW, DENSITY_MID]:
        log(f"\n--- Densidad {density} ---", 1)
        bar_labels, bar_values = [], []
        for pair_name, concepts in configs.items():
            run_name = f"exp10_{pair_name.replace('+','_')}_d{density}"
            cfg = build_config(size=6, density=density, epochs=epochs,
                               epsilon_max=best_eps, gamma=best_gamma,
                               algorithm_cls=JALGT,
                               solution_concept_cls=concepts[0],  # fallback
                               renders_dir=RENDERS_DIR,
                               per_agent_concepts=concepts)
            res = run_training(cfg, TRAIN_SEEDS, run_name)
            summary[f"{pair_name}_d{density}"] = res["conv_reward_mean"]
            bar_labels.append(pair_name)
            bar_values.append(res["conv_reward_mean"])
            log(f"  {pair_name}: convergencia={res['conv_reward_mean']:.3f}", 1)
        plot_bars(bar_labels, bar_values, "Recompensa en convergencia",
                  f"EXP10: parejas homogéneas vs heterogénea (6x6, d={density})",
                  f"exp10_heterogeneous_d{density}.png")

        het = summary[f"Nash+Pareto_d{density}"]
        best_homo = max(summary[f"Nash+Nash_d{density}"],
                        summary[f"Pareto+Pareto_d{density}"])
        if het < best_homo:
            log(f"  -> La pareja heterogénea ({het:.2f}) rinde por debajo de la mejor "
                f"homogénea ({best_homo:.2f}): conflicto de incentivos (hipótesis).", 1)
        else:
            log(f"  -> La pareja heterogénea ({het:.2f}) iguala o supera a la mejor "
                f"homogénea ({best_homo:.2f}): no hay penalización por mezclar.", 1)

    save_json({"experiment": "exp10_heterogeneous", "data": summary},
              "exp10_summary.json")
    return summary


def run_all_extensions(best_eps, best_gamma, epochs):
    """Lanza las tres ampliaciones en orden."""
    experiment_8_decay(best_eps, best_gamma, epochs)
    experiment_9_generalization(best_eps, best_gamma, epochs)
    experiment_10_heterogeneous(best_eps, best_gamma, epochs)


# ============================================================================
#  ORQUESTADOR PRINCIPAL
# ============================================================================

def run_obligatory(epochs):
    """Lanza la batería obligatoria y devuelve los hiperparámetros óptimos."""
    best_eps, best_gamma = experiment_1_hyperparams(epochs)
    experiment_2_iql_transfer(best_eps, best_gamma, epochs)
    experiment_3_validate_large(best_eps, best_gamma, epochs)
    experiment_4_5_iql_vs_jalgt(best_eps, best_gamma, epochs)
    experiment_6_concepts(best_eps, best_gamma, epochs)
    experiment_7_concepts_scaling(best_eps, best_gamma, epochs)
    return best_eps, best_gamma


def main():
    parser = argparse.ArgumentParser(
        description="Batería de experimentos -- Práctica 3 SID")
    parser.add_argument("--sanity", action="store_true",
                        help="Verificación rápida del pipeline (pocos epochs)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Número de epochs (sobreescribe el valor por defecto)")
    parser.add_argument("--base", action="store_true",
                        help="Ejecutar SOLO los experimentos obligatorios")
    parser.add_argument("--extensions", action="store_true",
                        help="Ejecutar SOLO las ampliaciones (+2 puntos)")
    args = parser.parse_args()

    if args.epochs is not None:
        epochs = args.epochs
    elif args.sanity:
        epochs = SANITY_EPOCHS
    else:
        epochs = DEFAULT_EPOCHS

    make_dirs(ALL_DIRS)

    banner("BATERÍA DE EXPERIMENTOS -- Práctica 3 SID")
    mode = "SANITY (verificación del pipeline)" if args.sanity else "COMPLETO"
    log(f"Modo: {mode} | epochs={epochs}")
    log(f"Seeds de entrenamiento: {TRAIN_SEEDS}")
    log(f"Densidades: baja={DENSITY_LOW}, media={DENSITY_MID}")
    log(f"Grid hiperparámetros: epsilon in {EPSILON_GRID}, gamma in {GAMMA_GRID}")
    log("Las conclusiones se trasladan automáticamente entre experimentos.")

    t_start = time.time()

    # Esquema de flags:
    #   (sin flag)    -> obligatorios + ampliaciones
    #   --base        -> solo obligatorios
    #   --extensions  -> solo ampliaciones (fija hiperparám. con EXP1 primero)
    if args.extensions and not args.base:
        # Solo ampliaciones: necesitan (eps*, gamma*), que sale del EXP1.
        best_eps, best_gamma = experiment_1_hyperparams(epochs)
        run_all_extensions(best_eps, best_gamma, epochs)
    else:
        best_eps, best_gamma = run_obligatory(epochs)
        if not args.base:
            # Sin --base, tras los obligatorios corremos también las ampliaciones.
            run_all_extensions(best_eps, best_gamma, epochs)

    elapsed = time.time() - t_start
    banner("BATERÍA COMPLETADA")
    log(f"Tiempo total: {elapsed/60:.1f} minutos")
    log(f"Hiperparámetros óptimos usados: epsilon_max={best_eps}, gamma={best_gamma}")
    log(f"Resultados en: {RESULTS_DIR}/ | Plots en: {PLOTS_DIR}/")
    log(f"Análisis SVG en: {SVG_ANALYSIS_DIR}/ | Animaciones en: {RENDERS_DIR}/")


if __name__ == "__main__":
    main()
