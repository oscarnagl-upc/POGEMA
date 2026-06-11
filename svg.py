"""
svg_analysis.py -- Protocolo de análisis automático de trazas SVG de POGEMA.

Los SVG de POGEMA codifican la trayectoria de cada agente en atributos
<animate attributeName="cx"/"cy" values="...">. Esto permite reconstruir la
posición de cada agente en cada paso y derivar métricas de comportamiento de
forma 100% automática y cuantitativa, sin inspección manual.

IMPORTANTE: el análisis requiere SVGs con vista GLOBAL (un fichero por mapa con
ambos agentes en el mismo sistema de coordenadas). Las vistas egocéntricas (un
SVG por agente) NO sirven para detectar colisiones porque cada una tiene su
propio sistema de referencia.

Métricas extraídas por SVG:
  - movimientos efectivos por agente
  - desplazamiento neto (Manhattan inicio->fin)
  - eficiencia = desplazamiento_neto / movimientos  (1.0 = trayectoria directa)
  - deadlock: un agente sin moverse durante toda la traza
  - colisión potencial: dos agentes que ocupan/intercambian la misma celda
  - cesión de paso: un agente que rodea mientras el otro llega directo

Filtrado de resolubilidad: con densidad media algunos mapas son irresolubles.
Un agente que no se mueve en un mapa sin ruta posible NO es un deadlock de
coordinación, así que esos mapas se excluyen del análisis (map_is_solvable).

Módulo autónomo: solo depende de numpy y pandas (testeable en aislamiento).
"""

import os
import re
import glob
from collections import deque as _deque

import numpy as np
import pandas as pd



def map_is_solvable(size, density, seed, num_agents=2):
    """Comprueba con BFS si en un mapa generado AMBOS agentes pueden alcanzar su
    objetivo (ignorando al otro agente). Replica la generación binomial de POGEMA.
    """
    rnd = np.random.default_rng(seed)
    grid = rnd.binomial(1, density, (size, size))
    free = [(r, c) for r in range(size) for c in range(size) if grid[r, c] == 0]
    if len(free) < 2 * num_agents:
        return False
    rnd2 = np.random.default_rng(seed + 10000)
    idx = rnd2.choice(len(free), size=2 * num_agents, replace=False)
    pts = [free[i] for i in idx]
    starts = pts[:num_agents]
    goals = pts[num_agents:]

    def bfs(start, goal):
        q = _deque([start]); seen = {start}
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 \
                        and (nr, nc) not in seen:
                    seen.add((nr, nc)); q.append((nr, nc))
        return False

    return all(bfs(starts[i], goals[i]) for i in range(num_agents))


def parse_svg_trajectories(svg_path, grid_size_px=100):
    """Reconstruye las trayectorias (lista de celdas) de cada agente del SVG."""
    with open(svg_path) as f:
        content = f.read()
    cx_lists = re.findall(r'attributeName="cx"[^>]*values="([^"]*)"', content)
    cy_lists = re.findall(r'attributeName="cy"[^>]*values="([^"]*)"', content)

    trajectories = []
    for cx_str, cy_str in zip(cx_lists, cy_lists):
        xs = [float(v) for v in cx_str.split(";")]
        ys = [float(v) for v in cy_str.split(";")]
        traj = []
        for i in range(max(len(xs), len(ys))):
            x = xs[min(i, len(xs) - 1)]
            y = ys[min(i, len(ys) - 1)]
            cell = (round(x / grid_size_px), round(-y / grid_size_px))
            if not traj or traj[-1] != cell:
                traj.append(cell)
        trajectories.append(traj)
    return trajectories


def classify_svg_behavior(svg_path):
    """Analiza un SVG y devuelve un dict con métricas de comportamiento."""
    trajs = parse_svg_trajectories(svg_path)
    if len(trajs) < 2:
        return None

    result = {"svg": os.path.basename(svg_path)}
    move_counts, efficiencies = [], []
    deadlock_flags = []

    for t in trajs[:2]:  # exactamente 2 agentes
        moves = len(t) - 1
        move_counts.append(moves)
        if moves == 0:
            deadlock_flags.append(True)
            efficiencies.append(0.0)
        else:
            deadlock_flags.append(False)
            net = abs(t[-1][0] - t[0][0]) + abs(t[-1][1] - t[0][1])
            efficiencies.append(net / moves)

    # Colisión potencial: misma celda en el mismo índice temporal, o swap
    collision = False
    t0, t1 = trajs[0], trajs[1]
    n = min(len(t0), len(t1))
    for k in range(n):
        if t0[k] == t1[k]:
            collision = True
            break
        if k > 0 and t0[k] == t1[k - 1] and t1[k] == t0[k - 1]:
            collision = True  # intercambio de celdas
            break

    result["agent0_moves"] = move_counts[0]
    result["agent1_moves"] = move_counts[1]
    result["agent0_efficiency"] = round(efficiencies[0], 3)
    result["agent1_efficiency"] = round(efficiencies[1], 3)
    result["any_deadlock"] = any(deadlock_flags)
    result["both_deadlock"] = all(deadlock_flags)
    result["potential_collision"] = collision
    # Cesión de paso: un agente con eficiencia baja (rodea) mientras el otro
    # llega directo. Heurística: efic < 0.5 para uno y > 0.7 para el otro.
    low, high = sorted(efficiencies)
    result["yielding_pattern"] = (low < 0.5 and high > 0.7)
    return result


def analyze_svgs(pattern, out_csv, size=None, density=None):
    """Analiza todos los SVGs que casen con el patrón y guarda un CSV resumen.

    Si se proporcionan size y density, FILTRA los mapas irresolubles (BFS) para
    que no ensucien las conclusiones. Devuelve un dict con agregados.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    rows = []
    n_skipped_unsolvable = 0
    for f in files:
        m = re.search(r'-map(\d+)\.svg$', os.path.basename(f))
        if m and size is not None and density is not None:
            map_seed = int(m.group(1))
            if not map_is_solvable(size, density, map_seed):
                n_skipped_unsolvable += 1
                continue
        try:
            r = classify_svg_behavior(f)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"    [aviso] no se pudo analizar {f}: {e}", flush=True)

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    return {
        "n_svgs": len(df),
        "n_skipped_unsolvable": n_skipped_unsolvable,
        "pct_any_deadlock": round(100 * df["any_deadlock"].mean(), 1),
        "pct_both_deadlock": round(100 * df["both_deadlock"].mean(), 1),
        "pct_collision": round(100 * df["potential_collision"].mean(), 1),
        "pct_yielding": round(100 * df["yielding_pattern"].mean(), 1),
        "mean_efficiency": round(
            df[["agent0_efficiency", "agent1_efficiency"]].values.mean(), 3),
    }
