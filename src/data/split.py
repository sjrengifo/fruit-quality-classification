"""Splits train/val/test estables y agrupados — Fase 3 de CRISP-DM.

Dos decisiones:
- **Agrupados:** Kaggle trae fotos con copias aumentadas (`20190820_143522.jpg`,
  `20190820_143522_22896.jpg`, …) y variantes numeradas (`IMG…_01.jpg`). Todas las imágenes de un
  mismo grupo (carpeta + instante de captura) van al mismo split, para que no haya fuga de
  información entre entrenamiento y prueba.
- **Estables:** el split de cada grupo sale de un hash de su clave, no de un sorteo global. Al
  agregar imágenes nuevas, las existentes no cambian de split y los resultados siguen comparables.

- **Casi-duplicados (HDF):** Healthy-Defective-Fruits no trae metadatos de captura y repite la
  misma fruta decenas o cientos de veces (ráfagas, copias exactas). `near_duplicate_groups` agrupa
  dentro de cada carpeta las imágenes con histograma de color casi idéntico (enlace simple), y el
  grupo completo va a un solo split. Sin esto el F1 en HDF sale inflado (0,99).

La recolección propia siempre va al split `externa` (prueba externa, nunca se entrena con ella).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from src import config

SPLITS = ["train", "val", "test", "externa", "extra"]  # extra: fuentes solo-entrenamiento
_TIMESTAMP = re.compile(r"(\d{8})_?(\d{6})")
_VARIANT = re.compile(r"(_\d{1,2})?$")
DUP_BINS = (12, 4, 4)     # histograma H, S, V de la imagen completa
DUP_THRESHOLD = 0.25      # distancia L1 (0–2) por debajo de la cual dos imágenes son la misma toma


def group_key(ruta: str, fuente: str) -> str:
    """Clave de grupo: carpeta + instante de captura (o nombre sin sufijo de variante)."""
    path = PurePosixPath(ruta)
    if fuente == "kaggle":
        m = _TIMESTAMP.search(path.stem)
        stem = m.group(1) + m.group(2) if m else _VARIANT.sub("", path.stem)
        return f"{path.parent}/{stem}"
    if fuente == "fruq":  # time-lapse: ventana de fotogramas consecutivos del mismo objeto
        frame = int(re.search(r"(\d+)", path.stem).group(1))
        return f"{path.parent}/{frame // (config.FRUQ_STEP * config.FRUQ_GROUP_FRAMES)}"
    return ruta


def color_signature(path) -> "np.ndarray":
    """Histograma HSV 3D normalizado de la imagen (reducida a 128 px)."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        img.draft("RGB", (128, 128))
        img = img.convert("RGB")
        img.thumbnail((128, 128))
        hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=DUP_BINS, range=[(0, 256)] * 3)
    return hist.ravel() / len(hsv)


def near_duplicate_groups(paths: list, threshold: float = DUP_THRESHOLD) -> list[int]:
    """Etiqueta de grupo por imagen: componentes conexos de las parejas a distancia L1 < threshold."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import cdist

    if not paths:
        return []
    with ThreadPoolExecutor() as pool:
        X = np.stack(list(pool.map(color_signature, paths)))
    return connected_components(cdist(X, X, "cityblock") < threshold)[1].tolist()


def assign_split(grupo: str, fuente: str) -> str:
    """Asigna train/val/test según un hash estable del grupo (≈ 60/20/20 por defecto)."""
    if fuente == "propias":
        return "externa"
    if fuente in config.TRAIN_ONLY_SOURCES:
        return "extra"
    digest = hashlib.md5(f"{config.RANDOM_SEED}:{grupo}".encode()).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    if u < config.TEST_SIZE:
        return "test"
    if u < config.TEST_SIZE + config.VAL_SIZE:
        return "val"
    return "train"
