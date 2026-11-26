"""Figuras de la Fase 4 a partir de experiments/results y experiments/logs → assets/modelado/.

Uso:  python -m src.evaluation.figuras_modelado
"""
from __future__ import annotations

import json

import numpy as np

from src import config

OUT = config.ROOT_DIR / "assets" / "modelado"
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_NAMES = {"mayoria": "Mayoría", "azar_estratificado": "Azar estratificado",
               "regresion_logistica": "Regresión logística", "svm": "SVM (RBF)", "random_forest": "Random Forest"}
CNN_NAMES = {
    "cnn_busqueda": ["Base", "Sin aumento de datos", "Red pequeña", "Más regularización", "Tasa de aprendizaje 3e-4"],
    "cnn_estable": ["Lote 128 + coseno", "Sin aumento de datos", "Pesos 1/√grupo", "Pesos 1/√grupo, lr 2e-3"],
}
FINAL_CNN_LABEL = "CNN (sin aumento, estable)"


def _load(name: str):
    with open(config.RESULTS_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", facecolor=SURFACE)
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor=SURFACE)


def compare_models(plt) -> None:
    """Barras horizontales: F1-macro en validación de cada modelo."""
    classic = _load("classic")
    items = [(MODEL_NAMES[k], v["val"]["f1_macro"]) for k, v in classic.items()]
    items.append((FINAL_CNN_LABEL, _load(config.FINAL_MODEL)["val"]["f1_macro"]))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    _style(ax)
    ax.barh([n for n, _ in items], [v for _, v in items], height=0.55, color=BLUE)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.012, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=8, color=INK)
    ax.axvline(0.85, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.85, len(items) - 0.35, "meta 0,85", fontsize=7.5, color=MUTED, ha="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-macro en validación", fontsize=8, color=MUTED)
    ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout()
    _save(fig, "comparacion_modelos")


def cnn_curves(plt, search: str = "cnn_busqueda", name: str = "cnn_curvas_validacion") -> None:
    """Múltiplos pequeños: F1-macro de validación por época de cada configuración de una búsqueda."""
    import pandas as pd

    names = CNN_NAMES[search]
    svm = _load("classic")["svm"]["val"]["f1_macro"]
    fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.4), sharey=True)
    for i, ax in enumerate(axes):
        _style(ax)
        f1 = pd.read_csv(config.LOGS_DIR / f"{search}_{i}_historia.csv")["val_f1_macro"]
        ax.axhline(svm, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.plot(range(1, len(f1) + 1), f1, color=BLUE, lw=2, solid_capstyle="round")
        k = int(f1.idxmax())
        ax.plot(k + 1, f1[k], "o", ms=6, color=BLUE, mec=SURFACE, mew=1.5)
        ax.text(k + 1, f1[k] + 0.04, f"{f1[k]:.2f}".replace(".", ","), ha="center", fontsize=8, color=INK)
        ax.set_title(names[i], fontsize=8.5, color=INK, loc="left")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("época", fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("F1-macro en validación", fontsize=8, color=MUTED)
    axes[0].text(1, svm - 0.07, "SVM", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, name)


def confusion_matrices(plt) -> None:
    """Matrices de confusión en validación (normalizadas por fila) del SVM y la CNN final."""
    panels = [("SVM (RBF)", _load("classic")["svm"]["val"]["matriz_confusion"]),
              (FINAL_CNN_LABEL, _load(config.FINAL_MODEL)["val"]["matriz_confusion"])]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, (title, cm) in zip(axes, panels):
        cm = np.array(cm)
        frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{frac[i, j]:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), config.QUALITY_CLASSES, fontsize=8)
        ax.set_yticks(range(3), config.QUALITY_CLASSES, fontsize=8)
        ax.tick_params(length=0, colors=INK)
        ax.set_xlabel("predicha", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("real", fontsize=8, color=MUTED)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    _save(fig, "matrices_confusion")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compare_models(plt)
    cnn_curves(plt)
    cnn_curves(plt, "cnn_estable", "cnn_curvas_estable")
    confusion_matrices(plt)
    print(f"Figuras → {OUT}")


if __name__ == "__main__":
    main()
