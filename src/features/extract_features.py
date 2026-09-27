"""Extracción de características para modelos de ML tradicional.

Color (histogramas HSV), textura (GLCM/LBP) y forma (momentos, área/perímetro).
Fase 3 de CRISP-DM.
"""
from __future__ import annotations

import numpy as np


def extract_color_features(image: np.ndarray) -> np.ndarray:
    """Histogramas de color en espacio HSV."""
    raise NotImplementedError("Implementar en Fase 3.")


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """Descriptores de textura (GLCM / LBP)."""
    raise NotImplementedError("Implementar en Fase 3.")


def extract_shape_features(image: np.ndarray) -> np.ndarray:
    """Descriptores de forma (momentos de Hu, área, perímetro, circularidad)."""
    raise NotImplementedError("Implementar en Fase 3.")


def extract_all(image: np.ndarray) -> np.ndarray:
    """Concatena todas las características en un único vector phi(x)."""
    return np.concatenate([
        extract_color_features(image),
        extract_texture_features(image),
        extract_shape_features(image),
    ])
