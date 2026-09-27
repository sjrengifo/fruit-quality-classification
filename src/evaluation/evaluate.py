"""Evaluación de modelos: métricas, matriz de confusión y análisis de error.

Fase 5 de CRISP-DM. Guarda figuras/tablas en experiments/results/.
"""
from __future__ import annotations

import numpy as np


def classification_report_dict(y_true, y_pred, labels=None) -> dict:
    """Reporte de precision/recall/F1 (por clase y macro)."""
    from sklearn.metrics import classification_report

    return classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )


def confusion(y_true, y_pred) -> np.ndarray:
    """Matriz de confusión."""
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred)


# TODO(Fase 5): curvas ROC-AUC, curvas de aprendizaje (sobreajuste),
# análisis de errores y comparación con la línea base.
