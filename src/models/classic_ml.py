"""Modelos de Machine Learning tradicional y búsqueda de hiperparámetros.

Fase 4 de CRISP-DM. Se requieren AL MENOS 2 modelos tradicionales (enunciado §3).
"""
from __future__ import annotations


def build_models() -> dict:
    """Devuelve un diccionario {nombre: (estimator, param_grid)} para GridSearchCV.

    Descomentar/ajustar los modelos elegidos por el equipo.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    return {
        "svm": (
            SVC(probability=True),
            {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        # TODO: añadir XGBoost, KNN, Regresión Logística, etc.
    }


def tune(estimator, param_grid, X, y, cv: int = 5):
    """Ajuste de hiperparámetros con validación cruzada (GridSearchCV, F1-macro)."""
    from sklearn.model_selection import GridSearchCV

    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    search.fit(X, y)
    return search
