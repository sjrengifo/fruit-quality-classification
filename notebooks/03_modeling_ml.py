# %% [markdown]
# # 03 · Modelado con ML tradicional
#
# **Fase 4 de CRISP-DM.** Líneas base y tres modelos sobre el vector de características φ(x) ∈ ℝ⁶⁸
# (notebook 02, §6). Este notebook **lee los resultados guardados**; para regenerarlos:
#
# ```bash
# python -m src.training.train --model classic    # GridSearchCV → experiments/results/classic.json, cv_*.csv
# python -m src.evaluation.ablation               # → experiments/results/ablacion_svm.json
# ```
#
# Protocolo: se ajusta con `train` (validación cruzada de 5 pliegues **por grupo**), se compara con
# `val`; `test` y las fotos propias no se usan hasta la Fase 5. Métrica principal: **F1-macro**.
# Pesos por muestra = 1/tamaño del grupo, con las tres clases equilibradas. Detalle en
# `docs/crisp-dm.md` §4.1.

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
from src.models.classic_ml import build_models  # noqa: E402

pd.set_option("display.precision", 3)
classic = json.load(open(config.RESULTS_DIR / "classic.json", encoding="utf-8"))
NAMES = {"mayoria": "Mayoría", "azar_estratificado": "Azar estratificado",
         "regresion_logistica": "Regresión logística", "svm": "SVM (RBF)", "random_forest": "Random Forest"}

# %% [markdown]
# ## 1. Modelos y rejillas de hiperparámetros
#
# Cada modelo es un pipeline `StandardScaler → clasificador`, así la estandarización se estima solo
# con los datos de entrenamiento de cada pliegue.

# %%
pd.DataFrame([{"modelo": NAMES[k], "clasificador": type(est.named_steps["clf"]).__name__,
               "rejilla": {p.removeprefix("clf__"): v for p, v in grid.items()},
               "combinaciones": int(np.prod([len(v) for v in grid.values()]))}
              for k, (est, grid) in build_models().items()])

# %% [markdown]
# ### Pesos por muestra (desbalanceo y copias)
#
# Peso = 1/tamaño del grupo, reescalado para que las tres clases sumen lo mismo
# (`sample_weights`, `src/training/train.py`). Así 30 copias de una foto cuentan como una, y
# Defectuosa/Podrida pesan tanto como Buena aunque tengan ~5× menos imágenes.

# %%
from src.features.extract_features import load_features  # noqa: E402
from src.training.train import sample_weights  # noqa: E402

_, rows_tr = load_features("train")
w = pd.DataFrame({"calidad": [r["calidad"] for r in rows_tr], "fuente": [r["fuente"] for r in rows_tr],
                  "peso": sample_weights(rows_tr)})
w.groupby("calidad").agg(imagenes=("peso", "size"), peso_total=("peso", "sum"), peso_medio=("peso", "mean"))

# %% [markdown]
# ## 2. Resultados

# %%
summary = pd.DataFrame([{
    "modelo": NAMES[k],
    "mejores hiperparámetros": {p.removeprefix("clf__"): v for p, v in r.get("mejores_hiperparametros", {}).items()} or "—",
    "F1 CV": r.get("f1_macro_cv"), "± CV": r.get("f1_macro_cv_std"),
    "F1 train": r.get("train", {}).get("f1_macro"),
    "F1 val": r["val"]["f1_macro"], "exactitud val": r["val"]["accuracy"],
    "F1 val Kaggle": r["val"]["por_fuente"].get("kaggle"), "F1 val HDF": r["val"]["por_fuente"].get("hdf"),
} for k, r in classic.items()])
summary

# %% [markdown]
# - La regresión logística (frontera lineal) queda muy por debajo de SVM y Random Forest: la
#   separación entre clases en φ(x) no es lineal.
# - SVM y Random Forest empatan en validación (0,84); el SVM es más estable entre pliegues (± 0,01).
# - F1 train ≫ F1 val en SVM/RF → sobreajuste, acotado por la validación cruzada por grupos.

# %% [markdown]
# ### Por clase (validación)

# %%
per_class = pd.concat({NAMES[k]: pd.DataFrame(r["val"]["por_clase"]).T
                       for k, r in classic.items() if k in ("regresion_logistica", "svm", "random_forest")})
per_class

# %% [markdown]
# Meta de negocio (§1.4): **recall de Podrida ≥ 0,90** (no dejar pasar fruta podrida como buena) y
# F1-macro ≥ 0,85.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, k in zip(axes, ["regresion_logistica", "svm", "random_forest"]):
    cm = np.array(classic[k]["val"]["matriz_confusion"])
    frac = cm / cm.sum(1, keepdims=True)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{frac[i, j]:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.55 else "black")
    ax.set_xticks(range(3), config.QUALITY_CLASSES, fontsize=8)
    ax.set_yticks(range(3), config.QUALITY_CLASSES, fontsize=8)
    ax.set_title(NAMES[k], fontsize=9, loc="left")
    ax.set_xlabel("predicha", fontsize=8)
axes[0].set_ylabel("real", fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## 3. Ajuste de hiperparámetros
#
# F1-macro de validación cruzada (media de 5 pliegues por grupo) para cada combinación. La brecha con
# el F1 de entrenamiento muestra el sobreajuste de cada configuración.

# %%
cv = pd.read_csv(config.RESULTS_DIR / "cv_svm.csv")
heat = cv.pivot(index="param_clf__C", columns="param_clf__gamma", values="mean_test_score")
fig, ax = plt.subplots(figsize=(4.5, 3))
ax.imshow(heat, cmap="Blues")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(heat.shape[1]), heat.columns)
ax.set_yticks(range(heat.shape[0]), heat.index)
ax.set_xlabel("gamma")
ax.set_ylabel("C")
ax.set_title("SVM: F1-macro CV", loc="left", fontsize=10)
plt.tight_layout()

# %%
for name in ["regresion_logistica", "random_forest"]:
    cv = pd.read_csv(config.RESULTS_DIR / f"cv_{name}.csv")
    cols = [c for c in cv.columns if c.startswith("param_")]
    display(cv[cols + ["mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]]  # noqa: F821
            .rename(columns=lambda c: c.removeprefix("param_clf__"))
            .sort_values("rank_test_score").head(6).assign(modelo=NAMES[name]))

# %% [markdown]
# ## 4. ¿Qué características usa el modelo?
#
# **Ablación** (SVM con los mejores hiperparámetros, entrenado solo con algunos grupos de
# características) e **importancia** en Random Forest (disminución media de impureza).

# %%
abl = json.load(open(config.RESULTS_DIR / "ablacion_svm.json", encoding="utf-8"))
pd.Series(abl["ablacion"], name="F1 val").to_frame()

# %%
import joblib  # noqa: E402

from src.features.extract_features import FEATURE_GROUPS, feature_names  # noqa: E402

rf = joblib.load(config.CHECKPOINTS_DIR / "random_forest.joblib").named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=feature_names())
group = np.repeat(list(FEATURE_GROUPS), list(FEATURE_GROUPS.values()))
print(imp.groupby(group).sum().round(3).to_dict())

top = imp.sort_values().tail(15)
plt.figure(figsize=(6, 4))
plt.barh(top.index, top.values, color="#2a78d6")
plt.title("Random Forest: 15 características más importantes", loc="left", fontsize=10)
plt.tight_layout()

# %% [markdown]
# ## 5. Generalización entre fuentes
#
# SVM entrenado con **una sola** fuente y validado en cada una. Si el modelo aprendiera "calidad" en
# general, debería funcionar razonablemente en la otra fuente.

# %%
cross = pd.Series(abl["entre_fuentes"])
cross.index = pd.MultiIndex.from_tuples([tuple(k.split("→")) for k in cross.index], names=["entrena", "valida"])
cross.unstack()

# %% [markdown]
# Un modelo entrenado con una fuente casi no sirve en la otra: cada dataset tiene su propio "aspecto"
# (frutas, cámaras, luz, criterio de etiquetado). El modelo conjunto funciona en ambas, pero esto
# anticipa que el desempeño con las **fotos propias** puede ser menor que en validación; por eso son
# la evaluación que más importa (Fase 5).
