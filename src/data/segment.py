"""Segmentación de frutas individuales y estimación de tamaño.

Necesario para la carpeta `mixed_quality` (varias frutas por imagen) y para estimar el
diámetro en píxeles. Fase 3 de CRISP-DM.
"""
from __future__ import annotations

import numpy as np


def segment_fruits(image: np.ndarray):
    """Segmenta frutas de un fondo simple y devuelve sus contornos/recortes.

    Estrategia propuesta (OpenCV): conversión a HSV, umbralización del fondo,
    operaciones morfológicas y detección de contornos.
    """
    raise NotImplementedError("Implementar en Fase 3 (segmentación con OpenCV).")


def estimate_diameter_px(contour) -> float:
    """Estima el diámetro (px) de una fruta a partir de su contorno."""
    raise NotImplementedError("Implementar en Fase 3 (círculo mínimo envolvente).")


def diameter_to_size_class(diameter_norm: float, thresholds=(0.33, 0.66)) -> str:
    """Discretiza el diámetro normalizado en {Pequeño, Mediano, Grande}."""
    from src import config

    t1, t2 = thresholds
    if diameter_norm < t1:
        return config.SIZE_CLASSES[0]
    if diameter_norm < t2:
        return config.SIZE_CLASSES[1]
    return config.SIZE_CLASSES[2]
