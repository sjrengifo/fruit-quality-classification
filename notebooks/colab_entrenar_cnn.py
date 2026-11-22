# %% [markdown]
# # Entrenar la CNN y el transfer learning en Google Colab (GPU)
#
# **Fase 4 de CRISP-DM.** Corre en Colab el mismo código de `src/` que en local, pero con GPU.
#
# **Antes de empezar** (una sola vez):
# 1. En el computador: `python -m src.training.colab` → `experiments/colab/fruit_colab.zip`.
# 2. Subir ese zip a Google Drive, en una carpeta llamada **`fruit-quality`** (en "Mi unidad").
# 3. Aquí en Colab: *Entorno de ejecución → Cambiar tipo de entorno → **T4 GPU*** (o L4).
#
# Luego: *Entorno de ejecución → Ejecutar todas*. Los resultados se copian a
# `Mi unidad/fruit-quality/experiments/` después de **cada** configuración; si la sesión se corta,
# basta con volver a ejecutar todo: las configuraciones ya terminadas se saltan.
#
# Al final: descargar `Mi unidad/fruit-quality/experiments/` y copiar su contenido en `experiments/`
# del repositorio (o pedírselo a Claude).

# %%
# Configuración: qué búsquedas correr, en orden (SEARCHES en src/models/cnn.py y src/models/transfer.py)
#   cnn_busqueda, cnn_estable      → CNN desde cero (ya entrenadas)
#   tl_busqueda                    → transfer learning: aumento de datos y aporte de FruQ (5 config.)
#   tl_entre_fuentes               → generalización: entrena sin una fuente y valida en ella (8 config.)
SEARCH_LIST = ["tl_busqueda", "tl_entre_fuentes"]
DRIVE_DIR = "/content/drive/MyDrive/fruit-quality"

# %% [markdown]
# ## 1. Verificar la GPU

# %%
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "· GPU:", gpus or "NINGUNA")
assert gpus, "No hay GPU: Entorno de ejecución → Cambiar tipo de entorno → T4 GPU"
# !nvidia-smi --query-gpu=name,memory.total --format=csv

# %% [markdown]
# ## 2. Traer código y datos desde Google Drive
#
# Se copia el zip al disco local de Colab antes de descomprimir: leer miles de archivos pequeños
# directamente desde Drive es muy lento.

# %%
import os
import shutil
import sys
import zipfile
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")
drive_dir = Path(DRIVE_DIR)
project = Path("/content/proyecto")
if not (project / "src").exists():
    shutil.copy(drive_dir / "fruit_colab.zip", "/content/fruit_colab.zip")
    with zipfile.ZipFile("/content/fruit_colab.zip") as z:
        z.extractall(project)
print((project / "VERSION.txt").read_text())

os.chdir(project)
sys.path.insert(0, str(project))

# %% [markdown]
# ## 3. Retomar resultados previos (si la sesión se cortó)

# %%
from src import config

if (drive_dir / "experiments").exists():
    shutil.copytree(drive_dir / "experiments", config.EXPERIMENTS_DIR, dirs_exist_ok=True)
for d in (config.CHECKPOINTS_DIR, config.RESULTS_DIR, config.LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)
for search in SEARCH_LIST:
    print(search, "· ya entrenadas:", sorted(p.stem for p in config.RESULTS_DIR.glob(f"{search}_*.json")))

# %% [markdown]
# ## 4. Entrenar
#
# Cada configuración imprime una línea por época (con los segundos acumulados) y al final su F1 en
# validación. Con T4: CNN desde cero ~5–10 min por configuración; transfer learning (EfficientNetB0,
# 18 épocas) ~5–15 min. Las búsquedas de transfer learning descargan los pesos de ImageNet (16 MB).

# %%
from src.training.train import all_searches, search_cnn

summaries = {}
for search in SEARCH_LIST:
    for i, overrides in enumerate(all_searches()[search]):
        print(f"{search}_{i}: {overrides}")
    # isolate=True: cada configuración en su propio proceso (si no, la RAM de Colab se agota tras varias)
    summaries[search] = search_cnn(search, sync_dir=drive_dir / "experiments", isolate=True)

# %% [markdown]
# ## 5. Resumen
#
# `f1_val` es la mejor época (optimista); `f1_val_ultimas_5_media` mide si el entrenamiento quedó
# estable. Una configuración buena tiene ambas cifras altas y parecidas.

# %%
import pandas as pd

for search, summary in summaries.items():
    print(search)
    display(pd.DataFrame(summary))  # noqa: F821

# %% [markdown]
# Listo: los resultados están en `Mi unidad/fruit-quality/experiments/`. Descargar esa carpeta y
# copiar su contenido en `experiments/` del repositorio.
