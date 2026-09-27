"""Utilidades comunes: reproducibilidad y E/S de imágenes."""
from __future__ import annotations

import random

import numpy as np

from src import config


def set_seed(seed: int | None = None) -> None:
    """Fija las semillas para reproducibilidad (Python, NumPy y TF si está disponible)."""
    seed = seed if seed is not None else config.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_image(path):
    """Carga una imagen en RGB como np.ndarray."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
