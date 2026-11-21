# %% [markdown]
# # 04 · CNN desde cero
#
# **Fase 4 de CRISP-DM.** Red convolucional pequeña entrenada desde cero sobre la fruta recortada en
# fondo blanco (notebook 02, §4). Este notebook **lee los resultados guardados**; para regenerarlos:
#
# ```bash
# python -m src.training.train --model cnn_search   # 5 configuraciones → experiments/results/cnn_busqueda*.json
# python -m src.training.train --model cnn_stable   # 4 configuraciones estables (en Colab con GPU: notebooks/colab_entrenar_cnn)
# python -m src.training.train --model cnn          # solo la configuración por defecto (CNN_DEFAULT)
# ```
#
# Mismo protocolo que los modelos clásicos: entrena con `train`, se elige con `val`. Detalle en
# `docs/crisp-dm.md` §4.2.

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.models.cnn import CNN_DEFAULT, CNN_GRID, build_cnn  # noqa: E402

pd.set_option("display.precision", 3)
NAMES = ["Base", "Sin aumento de datos", "Red pequeña", "Más regularización", "Tasa de aprendizaje 3e-4"]
runs = [json.load(open(config.RESULTS_DIR / f"cnn_busqueda_{i}.json", encoding="utf-8")) for i in range(len(CNN_GRID))]
histories = [pd.read_csv(config.LOGS_DIR / f"cnn_busqueda_{i}_historia.csv") for i in range(len(CNN_GRID))]

# %% [markdown]
# ## 1. Arquitectura
#
# Normalización a [0, 1] y aumento de datos van **dentro del modelo** (el aumento solo actúa al
# entrenar). Bloques Conv-BatchNorm-ReLU-MaxPool y GlobalAveragePooling en lugar de Flatten para
# tener pocos parámetros.

# %%
print(CNN_DEFAULT)
build_cnn().summary()

# %% [markdown]
# ### Aumento de datos
#
# Ejemplo: la misma imagen de entrenamiento pasada 6 veces por las capas de aumento (volteos,
# rotación, zoom, traslación con relleno blanco, brillo y contraste ±10 %).

# %%
import tensorflow as tf  # noqa: E402

from src.data.preprocess import load_split  # noqa: E402

model = build_cnn()
augment = tf.keras.Sequential([layer for layer in model.layers if layer.name.startswith("random_")])
X_tr, rows_tr = load_split("train")
sample = X_tr[[i for i, r in enumerate(rows_tr) if r["calidad"] == "Podrida"][0]][None]
fig, axes = plt.subplots(1, 7, figsize=(12, 2))
axes[0].imshow(sample[0])
axes[0].set_title("original", fontsize=8)
for ax in axes[1:]:
    ax.imshow(np.clip(augment(sample / 255.0, training=True)[0].numpy(), 0, 1))
for ax in axes:
    ax.set_axis_off()
plt.tight_layout()

# %% [markdown]
# ## 2. Búsqueda de configuraciones
#
# Una variación a la vez sobre la configuración base. Cada corrida conserva la época con mejor
# F1-macro en validación (máx. 40 épocas, parada tras 8 sin mejorar).

# %%
pd.DataFrame([{
    "configuración": NAMES[i], "cambios": CNN_GRID[i] or "—", "parámetros": r["parametros"], "épocas": r["epocas"],
    "F1 train": r["train"]["f1_macro"], "F1 val": r["val"]["f1_macro"],
    "F1 val Kaggle": r["val"]["por_fuente"]["kaggle"], "F1 val HDF": r["val"]["por_fuente"]["hdf"],
    "recall Podrida": r["val"]["por_clase"]["Podrida"]["recall"],
} for i, r in enumerate(runs)])

# %% [markdown]
# ## 3. Curvas de entrenamiento
#
# Pérdida de entrenamiento y F1-macro de validación por época. La línea punteada es el SVM (mejor
# modelo clásico).

# %%
svm = json.load(open(config.RESULTS_DIR / "classic.json", encoding="utf-8"))["svm"]["val"]["f1_macro"]
fig, axes = plt.subplots(2, len(runs), figsize=(14, 4.5), sharex="col")
for i, h in enumerate(histories):
    epochs = h["epoca"] + 1
    axes[0, i].plot(epochs, h["loss"], color="#52514e", lw=1.5)
    axes[0, i].set_title(NAMES[i], fontsize=9, loc="left")
    axes[1, i].plot(epochs, h["val_f1_macro"], color="#2a78d6", lw=1.5)
    axes[1, i].axhline(svm, color="#52514e", ls=":", lw=1)
    axes[1, i].set_ylim(0, 1)
    axes[1, i].set_xlabel("época", fontsize=8)
axes[0, 0].set_ylabel("pérdida train", fontsize=8)
axes[1, 0].set_ylabel("F1-macro val", fontsize=8)
plt.tight_layout()

# %% [markdown]
# **Entrenamiento inestable:** el F1 de validación oscila ±0,1 entre épocas consecutivas en todas las
# configuraciones, aunque la pérdida de entrenamiento baja de forma suave. Como se elige la mejor de
# ~35 épocas ruidosas con la propia validación, el F1 de validación de la CNN es **optimista** y las
# diferencias entre configuraciones (0,81–0,89) no son concluyentes. La cifra fiable sale del
# conjunto de prueba (Fase 5).

# %%
osc = pd.DataFrame({
    "configuración": NAMES,
    "mejor época": [int(h["val_f1_macro"].idxmax()) + 1 for h in histories],
    "F1 val máx.": [h["val_f1_macro"].max() for h in histories],
    "F1 val media últimas 10": [h["val_f1_macro"].tail(10).mean() for h in histories],
    "desv. últimas 10": [h["val_f1_macro"].tail(10).std() for h in histories],
})
osc

# %% [markdown]
# ## 4. Mejor configuración vs. SVM

# %%
best = int(np.argmax([r["val"]["f1_macro"] for r in runs]))
classic = json.load(open(config.RESULTS_DIR / "classic.json", encoding="utf-8"))
panels = [("SVM (RBF)", classic["svm"]["val"]), (f"CNN · {NAMES[best]}", runs[best]["val"])]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
for ax, (title, res) in zip(axes, panels):
    cm = np.array(res["matriz_confusion"])
    frac = cm / cm.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{frac[i, j]:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), config.QUALITY_CLASSES, fontsize=8)
    ax.set_yticks(range(3), config.QUALITY_CLASSES, fontsize=8)
    ax.set_title(f"{title} · F1 {res['f1_macro']:.2f}", fontsize=9, loc="left")
    ax.set_xlabel("predicha", fontsize=8)
axes[0].set_ylabel("real", fontsize=8)
plt.tight_layout()

# %%
pd.concat({"SVM": pd.DataFrame(classic["svm"]["val"]["por_clase"]).T,
           f"CNN · {NAMES[best]}": pd.DataFrame(runs[best]["val"]["por_clase"]).T})

# %% [markdown]
# - La CNN supera al SVM en F1-macro de validación, pero la ventaja viene de HDF; en Kaggle queda por
#   debajo.
# - El error más costoso (Podrida → Buena) es mayor en la CNN (9 % vs. 4 %) y su recall de Podrida no
#   cumple la meta de 0,90.
# - Pendiente: estabilizar el entrenamiento antes de fijar la configuración final (§5).

# %% [markdown]
# ## 5. Estabilización (Colab, GPU T4)
#
# Cambios: lotes de **128** y tasa de aprendizaje con **decaimiento coseno** hasta 0 en 40 épocas,
# sin parada temprana (`CNN_GRID_STABLE` en `src/models/cnn.py`). Además del F1 de la mejor época se
# mira la **media de las últimas 5 épocas**, que no depende de elegir una época con la propia
# validación. El IC95 se calcula remuestreando grupos (`src/evaluation/evaluate.py`).

# %%
from src.models.cnn import CNN_GRID_STABLE  # noqa: E402

STABLE_NAMES = ["Lote 128 + coseno", "Sin aumento de datos", "Pesos 1/√grupo", "Pesos 1/√grupo, lr 2e-3"]
stable = [json.load(open(config.RESULTS_DIR / f"cnn_estable_{i}.json", encoding="utf-8")) for i in range(len(CNN_GRID_STABLE))]
stable_hist = [pd.read_csv(config.LOGS_DIR / f"cnn_estable_{i}_historia.csv") for i in range(len(CNN_GRID_STABLE))]
stable_eval = [json.load(open(config.RESULTS_DIR / f"eval_cnn_estable_{i}_val.json", encoding="utf-8")) for i in range(len(CNN_GRID_STABLE))]
pd.DataFrame([{
    "configuración": STABLE_NAMES[i], "mejor época": r["mejor_epoca"], "minutos (T4)": r["minutos"],
    "F1 val": r["val"]["f1_macro"], "IC95": "[{:.2f}, {:.2f}]".format(*e["ic95"]["f1_macro"]),
    "últimas 5 media": r["f1_val_ultimas_5_media"], "últimas 5 desv.": r["f1_val_ultimas_5_desv"],
    "F1 val Kaggle": r["val"]["por_fuente"]["kaggle"], "F1 val HDF": r["val"]["por_fuente"]["hdf"],
    "recall Podrida": r["val"]["por_clase"]["Podrida"]["recall"],
} for i, (r, e) in enumerate(zip(stable, stable_eval))])

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True)
for ax, hists, names, title in [(axes[0], histories, NAMES, "Antes: lote 32 + ReduceLROnPlateau"),
                                (axes[1], stable_hist, STABLE_NAMES, "Después: lote 128 + coseno")]:
    for h, n in zip(hists, names):
        ax.plot(h["epoca"] + 1, h["val_f1_macro"], lw=1.3, label=n)
    ax.axhline(svm, color="#52514e", ls=":", lw=1)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("época", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("F1-macro val", fontsize=8)
plt.tight_layout()

# %% [markdown]
# Con el decaimiento coseno el F1 converge: en las últimas 5 épocas varía ±0,002–0,007 (antes ±0,1),
# y la mejor época queda a ≤ 0,02 de la media final. **Sin aumento de datos** vuelve a ser la mejor
# configuración, ahora también en Kaggle (0,83 vs. 0,79 del SVM).
#
# ## 6. Selección del modelo final
#
# Diferencia pareada (mismas imágenes de validación, bootstrap por grupos): si el IC95 no contiene
# 0, la diferencia no se explica por el azar de qué frutas cayeron en validación.

# %%
from src.evaluation.evaluate import f1_macro, paired_diff_ci, predict, recall_of  # noqa: E402

y, pred_final, rows_val = predict(config.FINAL_MODEL, "val")
groups = [r["grupo"] for r in rows_val]
rivals = {"SVM": predict("svm", "val")[1],
          **{STABLE_NAMES[i]: predict(f"cnn_estable_{i}", "val")[1] for i in (0, 2, 3)}}
pd.DataFrame([{"frente a": k, "métrica": m, **dict(zip(["diferencia", "IC95 inf.", "IC95 sup."],
                                                       paired_diff_ci(y, pred_final, p, groups, fn)))}
              for k, p in rivals.items()
              for m, fn in [("F1-macro", f1_macro), ("recall Podrida", recall_of(2))]])

# %% [markdown]
# **Modelo final: `cnn_estable_1`** (lote 128, coseno, sin aumento; `config.FINAL_MODEL`). Supera al
# SVM en F1-macro con diferencia significativa y empata en recall de Podrida; el SVM se mantiene como
# comparación. La elección se registró antes de abrir `test` (`docs/crisp-dm.md` §4.3).
