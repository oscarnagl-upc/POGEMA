"""
viz.py -- Utilidades de visualización y persistencia de resultados.

Agrupa la generación de figuras (curvas de aprendizaje, heatmaps, barras) y el
guardado de resúmenes en JSON. Depende de config_exp para los directorios de
salida, pero no sabe nada de experimentos concretos: recibe datos ya agregados.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin ventana: guardamos a fichero, no mostramos
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid")


def plot_curve(runs, ylabel, title, fname, show_band=True):
    """Dibuja una o varias curvas de aprendizaje en una sola figura.

    'runs' es una lista de dicts con claves 'label', 'mean' y opcionalmente 'std'.
    """
    plt.figure(figsize=(10, 6))
    for run in runs:
        epochs = range(len(run["mean"]))
        plt.plot(epochs, run["mean"], label=run["label"], linewidth=1.8)
        if show_band and "std" in run and run["std"] is not None:
            plt.fill_between(epochs,
                             np.array(run["mean"]) - np.array(run["std"]),
                             np.array(run["mean"]) + np.array(run["std"]),
                             alpha=0.15)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_heatmap(matrix, row_labels, col_labels, title, fname,
                 row_name="", col_name="", fmt=".2f"):
    """Dibuja un heatmap (usado para el grid de hiperparámetros)."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=fmt,
                xticklabels=col_labels, yticklabels=row_labels,
                cmap="viridis", cbar_kws={"label": "Recompensa en convergencia"})
    plt.xlabel(col_name)
    plt.ylabel(row_name)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_bars(labels, values, ylabel, title, fname, errors=None):
    """Diagrama de barras (usado para comparar conceptos / algoritmos)."""
    plt.figure(figsize=(9, 6))
    x = range(len(labels))
    plt.bar(x, values, yerr=errors, capsize=4,
            color=sns.color_palette("deep", len(labels)))
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_grouped_bars(group_labels, series, ylabel, title, fname):
    """Barras agrupadas: 'series' es dict {nombre_serie: [valores por grupo]}."""
    plt.figure(figsize=(10, 6))
    n_groups = len(group_labels)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups)
    palette = sns.color_palette("deep", n_series)
    for i, (name, values) in enumerate(series.items()):
        plt.bar(x + i * width, values, width, label=name, color=palette[i])
    plt.xticks(x + width * (n_series - 1) / 2, group_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, fname)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def save_json(obj, fname):
    """Guarda un resumen en JSON (limpiando objetos no serializables)."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()
                    if k not in ("final_algorithms",)}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2, ensure_ascii=False)
    return path
