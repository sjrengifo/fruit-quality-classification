# %% [markdown]
# # 02 · Preparación de los datos
#
# **Fase 3 de CRISP-DM.** Trazabilidad de cómo se pasa de las imágenes originales al conjunto que
# usan los modelos. Cada paso se ejecuta con un comando de `src/` y este notebook **solo lee sus
# resultados** (no repite el trabajo pesado):
#
# | Paso | Comando | Resultado |
# |------|---------|-----------|
# | 1. Índice unificado de fuentes | `python -m src.main index` | `data/processed/index.csv` |
# | 2. Formato y tamaño unificados | `python -m src.main preprocess` | `data/interim/img128/` |
# | 3. Segmentación (U²-Net) | `python -m src.main segment` | `data/interim/seg128/`, `mask128/` |
# | 4. Recorte centrado de la fruta | `python -m src.main crop` | `data/interim/crop128/`, `cropmask128/` |
# | 5. Características φ(x) | `python -m src.main features` | `data/processed/features.npz` |
# | Diagnóstico de atajos | `python -m src.evaluation.shortcut` | `experiments/results/shortcut.json` |
#
# Decisiones y su justificación: `docs/crisp-dm.md` §3. Se genera desde `notebooks/02_preprocessing.py`
# (jupytext).

# %%
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data.preprocess import cache_path  # noqa: E402

ASSETS = ROOT / "assets" / "preparacion"
ASSETS.mkdir(parents=True, exist_ok=True)
pd.set_option("display.precision", 3)
index = pd.read_csv(config.DATASET_INDEX)
print(f"{len(index)} imágenes en el índice")

# %% [markdown]
# ## 1. Fuentes y construcción de las 3 clases
#
# - **Kaggle `Good`** → Buena.
# - **Kaggle `Bad`**: mezcla defectos leves con podredumbre → se re-anotó a mano una muestra de
#   1.200 imágenes (`data/anotaciones/`, criterios en `docs/criterios_anotacion.md`).
# - **Healthy-Defective-Fruits (HDF)**: manzana y mango; `fresh` → Buena, golpes/sarna/manchas →
#   Defectuosa, `rot` → Podrida.
# - **Fotos propias** (`propias`): conjunto de prueba externo, nunca se entrena con ellas.

# %%
pd.crosstab(index["fuente"], index["calidad"], margins=True, margins_name="Total")

# %%
pd.crosstab([index["fuente"], index["fruta"]], index["calidad"])

# %% [markdown]
# ### Re-anotación de `Bad` (Kaggle)
#
# Etiquetas por anotador. `Descartar` = imagen ambigua, con varias frutas o no reconocible (no entra
# al índice). El acuerdo entre anotadores (kappa) se calcula con `python -m src.data.relabel` cuando
# hay al menos dos anotadores.

# %%
anot = pd.concat([pd.read_csv(p).assign(anotador=p.stem.removeprefix("bad_"))
                  for p in sorted(config.ANNOTATIONS_DIR.glob("bad_*.csv"))])
anot["fruta"] = anot["archivo"].str.split("/").str[-2].str.split("_").str[0]
pd.crosstab([anot["anotador"], anot["fruta"]], anot["etiqueta"], margins=True, margins_name="Total")

# %% [markdown]
# ## 2. Imágenes excluidas
#
# Imágenes en escala de grises: solo existían en `Buena` (copias aumentadas), así que el modelo
# aprendería "gris = Buena". Se excluyen al construir el índice.

# %%
excl = pd.read_csv(config.PROCESSED_DIR / "excluidas.csv")
excl["carpeta"] = excl["ruta"].str.split("/").str[-2]
excl.groupby(["motivo", "carpeta"]).size().rename("imágenes").to_frame()

# %% [markdown]
# ## 3. Grupos y splits
#
# Las imágenes que son la misma foto o la misma fruta forman un **grupo**, y cada grupo va completo a
# un solo split (train / val / test ≈ 60/20/20, por hash estable). Así ninguna fruta aparece a la vez
# en entrenamiento y en evaluación.
#
# - **Kaggle:** copias aumentadas de una misma captura (`20190820_143522_22896.jpg`).
# - **HDF:** la misma fruta fotografiada cientos de veces; se agrupan las imágenes con histograma de
#   color casi idéntico (`near_duplicate_groups`). Sin esto, el F1 en HDF salía 0,99 (fuga de datos).

# %%
sizes = index.groupby(["fuente", "grupo"]).size().rename("n").reset_index()
sizes.groupby("fuente")["n"].describe(percentiles=[0.5, 0.9, 0.99])[["count", "mean", "50%", "90%", "99%", "max"]] \
    .rename(columns={"count": "grupos", "mean": "imágenes/grupo (media)"})

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, src in zip(axes, ["kaggle", "hdf"]):
    n = sizes.loc[sizes["fuente"] == src, "n"]
    ax.hist(n, bins=np.logspace(0, np.log10(n.max() + 1), 30), color="#2a78d6")
    ax.set_xscale("log")
    ax.set_title(f"{src}: {len(n)} grupos para {n.sum()} imágenes", loc="left", fontsize=10)
    ax.set_xlabel("imágenes por grupo (escala log)")
axes[0].set_ylabel("grupos")
plt.tight_layout()

# %% [markdown]
# Ejemplo: muestras de los grupos más grandes de HDF (cada fila es un grupo; todas sus imágenes van
# al mismo split).

# %%
def contact_sheet(rows: pd.DataFrame, variant: str = "img", n: int = 10, ax=None):
    tiles = [np.asarray(Image.open(cache_path(i, variant=variant)).convert("RGB")) for i in rows["id"][:n]]
    tiles += [np.full_like(tiles[0], 255)] * (n - len(tiles))
    (ax or plt.gca()).imshow(np.hstack(tiles))


big = sizes[sizes["fuente"] == "hdf"].nlargest(4, "n")["grupo"]
fig, axes = plt.subplots(len(big), 1, figsize=(12, 1.4 * len(big)))
for ax, g in zip(axes, big):
    members = index[index["grupo"] == g].sample(frac=1, random_state=config.RANDOM_SEED)
    contact_sheet(members, ax=ax)
    ax.set_axis_off()
    ax.set_title(f"{len(members)} imágenes · {g.split('/')[-2]} · split {members['split'].iloc[0]}",
                 loc="left", fontsize=9)
plt.tight_layout()
fig.savefig(ASSETS / "casi_duplicados_hdf.png", dpi=150)
fig.savefig(ASSETS / "casi_duplicados_hdf.svg")

# %%
pd.crosstab(index["split"], index["calidad"], margins=True, margins_name="Total") \
    .reindex(["train", "val", "test", "externa", "Total"])

# %%
# Comprobación: ningún grupo cruza splits.
assert (index.groupby("grupo")["split"].nunique() == 1).all()
print("OK: cada grupo está en un solo split")

# %% [markdown]
# ## 4. Del archivo original a la entrada de los modelos
#
# 1. **original preprocesada** (`img`): orientación, RGB, 128×128 sin deformar.
# 2. **segmentada** (`seg`): U²-Net separa la fruta y el fondo pasa a blanco.
# 3. **máscara** (`mask`): sirve para forma y tamaño.
# 4. **recortada** (`crop`): caja de la máscara centrada y reescalada → mismo zoom para todas.
#    Es la entrada de todos los modelos.

# %%
variants = ["img", "seg", "mask", "crop"]
examples = (index[index["split"] != "externa"]
            .groupby(["fuente", "calidad"]).sample(1, random_state=config.RANDOM_SEED))
fig, axes = plt.subplots(len(examples), len(variants), figsize=(7, 1.8 * len(examples)))
for row_axes, (_, r) in zip(axes, examples.iterrows()):
    for ax, v in zip(row_axes, variants):
        ax.imshow(Image.open(cache_path(r["id"], variant=v)), cmap="gray")
        ax.set_xticks([]), ax.set_yticks([])
    row_axes[0].set_ylabel(f"{r['fuente']}\n{r['calidad']}", fontsize=8)
for ax, v in zip(axes[0], ["original", "segmentada", "máscara", "recortada"]):
    ax.set_title(v, fontsize=9)
plt.tight_layout()

# %% [markdown]
# Calidad de la segmentación: fracción de la imagen ocupada por la fruta. Con menos de 2 % se
# considera fallida y se conserva la imagen original.

# %%
stats = pd.read_csv(config.INTERIM_DIR / "seg_stats.csv").merge(index[["id", "fuente", "calidad"]], on="id")
print(f"Segmentaciones fallidas (< 2 %): {(stats['area'] < 0.02).sum()} de {len(stats)}")
stats.groupby(["fuente", "calidad"])["area"].describe(percentiles=[0.05, 0.5, 0.95])[["count", "5%", "50%", "95%"]]

# %% [markdown]
# ## 5. Diagnóstico de atajos
#
# Regresiones logísticas que solo ven información "prohibida": el **fondo** (anillo exterior de la
# imagen) o la **silueta** (área, posición y forma de la máscara, sin color). Si predicen la clase
# mejor que el azar, un modelo puede aprender eso en lugar de la calidad. F1-macro en validación.

# %%
sc = pd.DataFrame(json.load(open(config.RESULTS_DIR / "shortcut.json", encoding="utf-8")))
sc.pivot(index="prueba", columns="subconjunto", values="f1_atajo") \
    .reindex(["fondo (original)", "fondo (segmentada)", "silueta", "silueta (recortada)"])[["todas", "kaggle", "hdf"]] \
    .assign(azar=sc.groupby("prueba")["f1_azar"].first())

# %% [markdown]
# - El fondo original predecía la clase (sobre todo en HDF); tras segmentar queda al nivel del azar.
# - La silueta en HDF (encuadre/zoom) baja con el recorte centrado.
# - La silueta en "todas" sigue por encima del azar porque la forma delata la fruta y cada fuente
#   tiene otra mezcla de frutas y clases; no es un atajo de captura.

# %% [markdown]
# ## 6. Características para ML tradicional
#
# φ(x) ∈ ℝ⁶⁸ calculado solo con los píxeles de la fruta: color (histogramas HSV, media, desviación),
# textura (LBP, GLCM) y forma (área, circularidad, solidez, momentos de Hu).

# %%
from src.features.extract_features import FEATURE_GROUPS, feature_names, load_features  # noqa: E402

X, rows = load_features()
feats = pd.DataFrame(X, columns=feature_names())
feats["calidad"] = [r["calidad"] for r in rows]
feats = feats[[r["split"] != "externa" for r in rows]]  # solo datos de entrenamiento/evaluación
print(FEATURE_GROUPS, "→", X.shape)

fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
for ax, col in zip(axes, ["media_h", "media_v", "glcm_contrast_d1", "circularidad"]):
    for q, color in zip(config.QUALITY_CLASSES, ["#2a78d6", "#eda100", "#e34948"]):
        ax.hist(feats.loc[feats["calidad"] == q, col], bins=40, density=True, histtype="step", lw=2,
                color=color, label=q)
    ax.set_title(col, fontsize=9)
axes[0].legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## 7. Fotos propias (prueba externa)
#
# Tomadas por el equipo; se procesan con exactamente el mismo pipeline. Aquí solo se muestran:
# **no se evalúan hasta tener el modelo final** (Fase 5), para que sigan siendo una prueba externa.

# %%
own = index[index["fuente"] == "propias"]
pd.crosstab(own["fruta"], own["calidad"], margins=True, margins_name="Total")

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 4.5))
for ax, q in zip(axes, config.QUALITY_CLASSES):
    contact_sheet(own[own["calidad"] == q].sample(frac=1, random_state=config.RANDOM_SEED), variant="crop",
                  n=12, ax=ax)
    ax.set_axis_off()
    ax.set_title(q, loc="left", fontsize=9)
plt.tight_layout()

# %% [markdown]
# ## 8. Tercera iteración: imágenes con varias frutas y revisión de etiquetas
#
# Tras la segunda mirada a las fotos propias (`docs/crisp-dm.md` §5.4: 20 de 30 Podrida →
# Defectuosa) se planteó la hipótesis de que **el ruido en las etiquetas** explicaba los errores:
# fotos con varias frutas y una frontera Defectuosa/Podrida mal definida. Se revisaron los datos
# **sin tocar `index.csv`**: el resultado es un índice aparte, `data/processed/index_v3.csv`, con el
# que se entrenaron modelos nuevos desde cero (§4.6 y §5.6 de `docs/crisp-dm.md`).
#
# ```bash
# python -m src.data.revision objetos   # objetos grandes por máscara de U²-Net
# python -m src.data.revision hojas     # hojas de contacto a ciegas, una celda por grupo
# python -m src.data.revision indice    # aplica data/anotaciones/revision_v3.txt → index_v3.csv
# python -m src.data.revision resumen   # acuerdo etiqueta original ↔ revisada
# ```
#
# ### 8.1 Imágenes con varias frutas
#
# Se cuentan las componentes conexas de la máscara de segmentación que miden al menos el 15 % de la
# más grande (`count_objects`). Con 2 o más, la imagen no es "una fruta sola" y se excluye.

# %%
from src.data.revision import OBJECTS_CSV, read_labels  # noqa: E402

objects = pd.read_csv(OBJECTS_CSV).merge(index[["id", "fuente", "calidad", "grupo", "split"]])
objects["varias"] = objects["objetos"] >= 2
pd.crosstab([objects["fuente"], objects["calidad"]], objects["varias"].map({False: "una", True: "varias"}))

# %% [markdown]
# Casi todas están en **Kaggle `Buena`** (1.712 de 1.816): el modelo podía aprender "varias frutas →
# Buena", algo que nunca ocurre en las fotos propias (una fruta por foto). Muestra al azar:

# %%
multi = objects[objects["varias"]].sample(12, random_state=config.RANDOM_SEED)
fig, ax = plt.subplots(figsize=(12, 1.4))
contact_sheet(multi, variant="img", n=12, ax=ax)
ax.set_axis_off()
ax.set_title("Imágenes excluidas por tener varias frutas (muestra)", loc="left", fontsize=9)
fig.savefig(ASSETS / "varias_frutas.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# El detector acertó en una muestra de 30 de Kaggle `Buena` (30/30 con varias frutas). No detecta
# frutas que **se tocan** (la máscara queda unida): esas se marcaron a mano como `Descartar` en la
# revisión visual (sobre todo racimos de banano y grupos de guayabas y limas de Kaggle).
#
# ### 8.2 Revisión visual de etiquetas
#
# - **Qué se revisó:** todos los grupos de Defectuosa y Podrida (Kaggle, HDF, FruQ) y los de `Buena` de
#   HDF y FruQ (los 6.353 grupos de Kaggle `Buena` no: solo se filtraron por varias frutas). Una celda
#   por **grupo** (misma fruta): un representante o, si el grupo tiene más de 5 imágenes, un mosaico
#   2×2. Los grupos de 40 o más imágenes se verificaron después con 24 imágenes cada uno.
# - **A ciegas:** las hojas no muestran la etiqueta original ni la fuente.
# - **Criterio:** el mismo de `docs/criterios_anotacion.md`.
# - **Quién:** las etiquetas las asignó **un modelo de IA** a pedido del equipo, por falta de
#   tiempo para revisarlas a mano. Se declara en el informe.
# - **Formato:** `data/anotaciones/revision_v3.txt`, una línea por hoja con una letra por celda
#   (B/D/P/X = Buena/Defectuosa/Podrida/Descartar).

# %%
Image.open(config.INTERIM_DIR / "revision" / "hojas" / "h000.jpg")

# %%
from sklearn.metrics import cohen_kappa_score  # noqa: E402

revised = read_labels()
first = index.drop_duplicates("grupo").set_index("grupo")
rev = pd.DataFrame({"grupo": list(revised), "revisada": list(revised.values())})
rev["original"] = rev["grupo"].map(first["calidad"])
rev["fuente"] = rev["grupo"].map(first["fuente"])
summary = []
for source, g in rev.groupby("fuente"):
    kept = g[g["revisada"] != "Descartar"]
    summary.append({"fuente": source, "grupos revisados": len(g), "descartados": len(g) - len(kept),
                    "cambian de clase": int((kept["original"] != kept["revisada"]).sum()),
                    "kappa original↔revisada": cohen_kappa_score(kept["original"], kept["revisada"])})
pd.DataFrame(summary).set_index("fuente")

# %%
pd.crosstab([rev["fuente"], rev["original"]], rev["revisada"])

# %% [markdown]
# - **Kaggle (κ = 0,29):** la etiqueta original es la re-anotación de Samuel (§1); κ < 0,6 es el umbral
#   que el propio documento de criterios fija para "discutir y afinar". La mayoría de los desacuerdos
#   son **Podrida → Defectuosa** (248 grupos): la frontera entre las dos clases es ambigua incluso
#   con el mismo criterio escrito.
# - **HDF (κ = 0,40):** la carpeta `rot_defect` (antracnosis del mango) incluye muchas frutas con
#   manchas pequeñas que nuestro criterio llama Defectuosa.
# - **FruQ (κ = 0,86):** las etiquetas por etapa del video coinciden casi siempre.
# - En conjunto, la revisión hizo **Podrida más exigente** (945 imágenes Podrida → Defectuosa contra
#   137 en sentido contrario). Es lo opuesto a lo que pedían los errores con fotos propias (el modelo
#   ya llamaba Defectuosa a fruta podrida): el resultado del reentrenamiento está en `05_evaluation` §6.

# %%
index_v3 = pd.read_csv(config.DATASET_INDEX_V3)
both = pd.concat({"original": index[index["fuente"] != "propias"], "v3": index_v3[index_v3["fuente"] != "propias"]})
pd.crosstab([both.index.get_level_values(0), both["split"]], both["calidad"])

# %% [markdown]
# ## 9. Cuarta iteración: 3 frutas, regla del banano y fotos nuevas
#
# - **Alcance:** solo manzana, banano y naranja (`config.V4_FRUITS`), donde el daño contrasta con la
#   piel. Se aplica al entrenar (clave `"frutas"`), sobre el índice original (`v4_0`) o el v4 (`v4_1`).
# - **Regla del banano** (`docs/criterios_anotacion.md`): las pecas de maduración no son defecto. Sobre
#   la revisión v3 solo cambian 3 grupos de FruQ (`data/anotaciones/revision_v4.txt`;
#   `python -m src.data.revision indice4` → `index_v4.csv`).

# %%
from src.training.train import index_path  # noqa: E402

v4_counts = {}
for name, ind in [("original, 3 frutas", None), ("v4, 3 frutas", "v4")]:
    d = pd.read_csv(index_path(ind))
    d = d[d["fruta"].isin(config.V4_FRUITS) & d["split"].isin(["train", "val", "extra"])]
    v4_counts[name] = d.groupby(["split", "calidad"]).size()
pd.DataFrame(v4_counts).unstack(1)

# %% [markdown]
# **Fotos nuevas** (`data/external/propias/nov24/`, 24 nov. 2026): 35 fotos, prueba limpia de la cuarta
# iteración. No están en el índice: se predicen con el mismo camino que la app. Aquí, lo que ve el
# modelo (fruta segmentada y recortada):

# %%
from src.data.segment import segment_and_crop  # noqa: E402

nov24 = pd.read_csv(config.NOV24_DIR / "anotaciones.csv")
print(pd.crosstab(nov24["fruta"], nov24["calidad"], margins=True, margins_name="Total"))
tiles = [segment_and_crop(config.NOV24_DIR / "imagenes" / f)[0] for f in nov24["archivo"]]
tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 9)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.imshow(np.vstack([np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)]))
ax.set_axis_off()
ax.set_title("Fotos nuevas segmentadas y recortadas (en orden de anotaciones.csv)", loc="left", fontsize=9)
fig.savefig(ASSETS / "fotos_nuevas_nov24.png", dpi=130, bbox_inches="tight")
