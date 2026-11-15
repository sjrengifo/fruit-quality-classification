"""Splits train/val/test estables y agrupados — Fase 3 de CRISP-DM.

Dos decisiones:
- **Agrupados:** Kaggle trae fotos con copias aumentadas (`20190820_143522.jpg`,
  `20190820_143522_22896.jpg`, …) y variantes numeradas (`IMG…_01.jpg`). Todas las imágenes de un
  mismo grupo (carpeta + instante de captura) van al mismo split, para que no haya fuga de
  información entre entrenamiento y prueba.
- **Estables:** el split de cada grupo sale de un hash de su clave, no de un sorteo global. Al
  agregar imágenes nuevas, las existentes no cambian de split y los resultados siguen comparables.

La recolección propia siempre va al split `externa` (prueba externa, nunca se entrena con ella).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from src import config

SPLITS = ["train", "val", "test", "externa"]
_TIMESTAMP = re.compile(r"(\d{8})_?(\d{6})")
_VARIANT = re.compile(r"(_\d{1,2})?$")


def group_key(ruta: str, fuente: str) -> str:
    """Clave de grupo: carpeta + instante de captura (o nombre sin sufijo de variante)."""
    path = PurePosixPath(ruta)
    if fuente == "kaggle":
        m = _TIMESTAMP.search(path.stem)
        stem = m.group(1) + m.group(2) if m else _VARIANT.sub("", path.stem)
        return f"{path.parent}/{stem}"
    return ruta


def assign_split(grupo: str, fuente: str) -> str:
    """Asigna train/val/test según un hash estable del grupo (≈ 60/20/20 por defecto)."""
    if fuente == "propias":
        return "externa"
    digest = hashlib.md5(f"{config.RANDOM_SEED}:{grupo}".encode()).hexdigest()
    u = int(digest[:8], 16) / 0xFFFFFFFF
    if u < config.TEST_SIZE:
        return "test"
    if u < config.TEST_SIZE + config.VAL_SIZE:
        return "val"
    return "train"
