"""Preprocesamiento de imágenes: redimensionado y normalización.

Fase 3 de CRISP-DM (Preparación de los datos).
"""
from __future__ import annotations

import numpy as np

from src import config


def resize_and_normalize(image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Redimensiona a `size` y normaliza los píxeles al rango [0, 1].

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) en formato BGR/RGB, dtype uint8.
    size  : (alto, ancho) destino; por defecto config.IMAGE_SIZE.
    """
    import cv2  # import diferido para no exigir OpenCV al importar el módulo

    size = size or config.IMAGE_SIZE
    resized = cv2.resize(image, size)
    return resized.astype("float32") / 255.0


# TODO(Fase 3): data augmentation, corrección de color, batch loading.
