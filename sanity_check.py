"""
=============================================================
  SANITY CHECK — Práctica 3 SID (MARL con JAL-GT e IQL)
  Ejecutar ANTES de cualquier experimento real.
  Tiempo estimado: < 10 segundos
=============================================================
"""

import sys
import time

print("=" * 60)
print("  SANITY CHECK — Baseline POGEMA + JAL-GT")
print("=" * 60)

# ── PASO 1: Importación de dependencias ──────────────────────
print("\n[1/4] Verificando importaciones...")
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Sin interfaz gráfica (no bloquea)
    import seaborn
    import tqdm
    print(f"      numpy     {np.__version__}  ✅")
    print(f"      matplotlib {matplotlib.__version__}  ✅")
    print(f"      seaborn   {seaborn.__version__}  ✅")
    print(f"      tqdm               ✅")
except ImportError as e:
    print(f"  ❌ Error importando dependencia base: {e}")
    print("     Ejecuta: pip install -r requirements.txt")
    sys.exit(1)

try:
    import pogema
    print(f"      pogema    {pogema.__version__}  ✅")
except ImportError:
    print("  ❌ pogema no encontrado. Ejecuta: pip install pogema==1.2.2")
    sys.exit(1)

try:
    import gymnasium
    print(f"      gymnasium {gymnasium.__version__}  ✅")
except ImportError:
    print("  ❌ gymnasium no encontrado.")
    sys.exit(1)

# ── PASO 2: Creación del entorno POGEMA ──────────────────────
print("\n[2/4] Verificando entorno POGEMA...")
try:
    from pogema import pogema_v0, GridConfig

    gc = GridConfig(
        num_agents=2,
        size=4,
        density=0.1,
        seed=42,
        max_episode_steps=16,
        obs_radius=1,
        on_target="finish",
        render_mode=None,
    )
    env = pogema_v0(gc)
    obs, _ = env.reset()

    assert len(obs) == 2,          "Número de observaciones incorrecto"
    assert len(obs[0]) == 3,       "Observación no tiene 3 matrices (obstáculos, agentes, objetivo)"
    assert obs[0][0].shape == (3, 3), "Matriz de obstáculos no es 3×3"

    actions = (0, 0)  # STAY, STAY
    obs2, rewards, terminated, truncated, _ = env.step(actions)

    assert len(rewards) == 2, "Número de rewards incorrecto"
    print(f"      Entorno 4×4 creado y reseteado  ✅")
    print(f"      Observación shape: {[m.shape for m in obs[0]]}")
    print(f"      Rewards tras STAY: {rewards}")
except Exception as e:
    print(f"  ❌ Error al crear el entorno: {e}")
    sys.exit(1)

# ── PASO 3: Verificación de obs_to_state ─────────────────────
print("\n[3/4] Verificando obs_to_state...")
try:
    def obs_to_state(obs):
        matrix_obstacles = obs[0]
        matrix_agents    = obs[1]
        matrix_target    = obs[2]
        target    = (np.max(matrix_target[2]) * 1
                     + matrix_target[1][0] * 2
                     + matrix_target[1][2] * 3)
        obstacles = (matrix_obstacles[0][1] * 2**9
                     + matrix_obstacles[1][0] * 2**8
                     + matrix_obstacles[1][2] * 2**7
                     + matrix_obstacles[2][1] * 2**6)
        agents    = (matrix_agents[0][1] * 2**5
                     + matrix_agents[1][0] * 2**4
                     + matrix_agents[1][2] * 2**3
                     + matrix_agents[2][1] * 2**2)
        return int(obstacles + agents + target)

    NUM_STATES = 16 * 16 * 4  # 1024

    # Testar sobre 50 pasos aleatorios
    env2 = pogema_v0(gc)
    obs_test, _ = env2.reset()
    out_of_range = 0
    for _ in range(50):
        for i in range(2):
            s = obs_to_state(obs_test[i])
            if not (0 <= s < NUM_STATES):
                out_of_range += 1
        a = tuple(np.random.randint(0, 5) for _ in range(2))
        obs_test, _, term, trunc, _ = env2.step(a)
        if all(term) or all(trunc):
            obs_test, _ = env2.reset()

    if out_of_range > 0:
        print(f"  ⚠️  {out_of_range} estados fuera de rango [0, {NUM_STATES-1}]")
    else:
        print(f"      obs_to_state: todos los estados en [0, {NUM_STATES-1}]  ✅")
        print(f"      num_states = {NUM_STATES} es correcto para size=4  ✅")
except Exception as e:
    print(f"  ❌ Error en obs_to_state: {e}")
    sys.exit(1)

# ── PASO 4: Ejecución mínima del baseline (3 epochs) ─────────
print("\n[4/4] Ejecutando baseline mínimo (3 epochs × 3 episodios)...")
try:
    import os, importlib.util

    # Cargar módulos del proyecto
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from algorithms import JALGT
    from solution_concepts import ParetoSolutionConcept
    from game_model import GameModel
    from gymnasium import Wrapper

    # Importar AnimationMonitor (compatible con pogema 1.2.2 y 1.4.x)
    try:
        from pogema.animation import AnimationMonitor, AnimationConfig
    except ModuleNotFoundError:
        from pogema.svg_animation.animation_wrapper import AnimationMonitor, AnimationConfig

    class RewardWrapper(Wrapper):
        def step(self, joint_action):
            observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
            for i in range(len(joint_action)):
                if not terminated[i] and not truncated[i]:
                    if rewards[i] == 0:
                        rewards[i] -= 0.01
            return observations, rewards, terminated, truncated, infos

    RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders_sanity")
    os.makedirs(RENDER_DIR, exist_ok=True)

    exp_config = {
        "num_agents": 2,
        "size": 4,
        "maps": 3,
        "num_states": 16 * 16 * 4,
        "epochs": 3,
        "episodes_per_epoch": 3,
        "episode_length": 16,
        "obstacle_density": 0.1,
        "save_every": None,
        "learning_rate": 0.1,
        "epsilon_max": 1.0,
        "epsilon_min": 0.1,
        "renders": RENDER_DIR,
        "algorithm_cls": JALGT,
        "algorithm_kwargs": {},
        "solution_concept": ParetoSolutionConcept,
    }

    def create_env_sanity(config, seed=42):
        # Compatible AnimationConfig (ignorar params renombrados según versión)
        try:
            anim_cfg = AnimationConfig(
                directory=config["renders"],
                static=False,
                show_agents=True,
                egocentric_idx=None,
                save_every_idx_episode=config["save_every"],
                show_border=True,
                show_lines=True,
            )
        except TypeError:
            anim_cfg = AnimationConfig(
                directory=config["renders"],
                static=False,
                show_agents=True,
                egocentric_idx=None,
                save_every_idx_episode=config["save_every"],
                show_grid_lines=True,
            )
        gc = GridConfig(
            num_agents=config["num_agents"], size=config["size"],
            density=config["obstacle_density"], seed=seed,
            max_episode_steps=config["episode_length"],
            obs_radius=1, on_target="finish", render_mode=None,
        )
        env = pogema_v0(gc)
        env = AnimationMonitor(env, animation_config=anim_cfg)
        return RewardWrapper(env)

    game = GameModel(
        num_agents=exp_config["num_agents"],
        num_states=exp_config["num_states"],
        num_actions=5,
    )
    algorithms = [
        JALGT(i, game, ParetoSolutionConcept(),
              epsilon=exp_config["epsilon_max"],
              alpha=exp_config["learning_rate"])
        for i in range(game.num_agents)
    ]

    t0 = time.time()
    reward_history = []

    for epoch in range(exp_config["epochs"]):
        # Entrenamiento
        for ep in range(exp_config["episodes_per_epoch"]):
            env = create_env_sanity(exp_config, seed=ep % exp_config["maps"])
            obs, _ = env.reset()
            states = [obs_to_state(obs[i]) for i in range(2)]
            term = [False, False]
            trunc = [False, False]
            while not all(term) and not all(trunc):
                actions = tuple(algorithms[i].select_action(states[i]) for i in range(2))
                obs, rewards, term, trunc, _ = env.step(actions)
                nxt = [obs_to_state(obs[i]) for i in range(2)]
                for i in range(2):
                    algorithms[i].learn(actions, rewards, states[i], nxt[i])
                states = nxt

        # Evaluación
        epoch_reward = 0
        for ep in range(exp_config["maps"]):
            env = create_env_sanity(exp_config, seed=ep)
            obs, _ = env.reset()
            states = [obs_to_state(obs[i]) for i in range(2)]
            term = [False, False]
            trunc = [False, False]
            total = [0, 0]
            while not all(term) and not all(trunc):
                actions = tuple(algorithms[i].select_action(states[i], train=False) for i in range(2))
                obs, rewards, term, trunc, _ = env.step(actions)
                total = [total[i] + rewards[i] for i in range(2)]
                states = [obs_to_state(obs[i]) for i in range(2)]
            # Guardar SVG del último epoch
            if epoch == exp_config["epochs"] - 1:
                for agent_i in range(2):
                    try:
                        env.save_animation(
                            os.path.join(RENDER_DIR, f"ParetoSolutionConcept-map{ep}-agent{agent_i}-epoch{epoch}.svg"),
                            AnimationConfig(egocentric_idx=agent_i, show_grid_lines=True),
                        )
                    except TypeError:
                        env.save_animation(
                            os.path.join(RENDER_DIR, f"ParetoSolutionConcept-map{ep}-agent{agent_i}-epoch{epoch}.svg"),
                            AnimationConfig(egocentric_idx=agent_i, show_border=True, show_lines=True),
                        )
            epoch_reward += sum(total)
        reward_history.append(epoch_reward)

    elapsed = time.time() - t0

    # Contar SVGs
    svgs = [f for f in os.listdir(RENDER_DIR) if f.endswith(".svg")]
    assert len(svgs) > 0, "No se generó ningún SVG"

    print(f"      3 epochs completados en {elapsed:.1f}s  ✅")
    print(f"      Recompensas por epoch: {[round(r, 3) for r in reward_history]}")
    print(f"      SVGs generados en '{RENDER_DIR}': {len(svgs)} ficheros  ✅")
    svg_example = os.path.join(RENDER_DIR, svgs[0])
    svg_size = os.path.getsize(svg_example)
    assert svg_size > 500, f"SVG parece vacío ({svg_size} bytes)"
    print(f"      Ejemplo SVG: {svgs[0]} ({svg_size} bytes)  ✅")

except Exception as e:
    import traceback
    print(f"  ❌ Error en ejecución baseline: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── RESUMEN FINAL ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  SANITY CHECK COMPLETADO — El entorno funciona.")
print("  Puedes proceder con los experimentos reales.")
print("=" * 60)
