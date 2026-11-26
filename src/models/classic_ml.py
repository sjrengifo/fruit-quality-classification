"""Modelos de Machine Learning tradicional y búsqueda de hiperparámetros.

Fase 4 de CRISP-DM. Se requieren AL MENOS 2 modelos tradicionales (enunciado §3). Todos reciben el
vector φ(x) de src/features/extract_features.py, estandarizado dentro del pipeline (así la media y
la desviación se estiman solo con los datos de entrenamiento de cada pliegue).
"""
from __future__ import annotations

from src import config


def build_baselines() -> dict:
    """Líneas base triviales: {nombre: estimator}."""
    from sklearn.dummy import DummyClassifier

    return {
        "mayoria": DummyClassifier(strategy="most_frequent"),
        "azar_estratificado": DummyClassifier(strategy="stratified", random_state=config.RANDOM_SEED),
    }


def build_models() -> dict:
    """Devuelve {nombre: (pipeline, param_grid)} para GridSearchCV. El clasificador es el paso `clf`."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def pipe(clf):
        """Pipeline estandarización + clasificador (el escalado se ajusta dentro de cada pliegue)."""
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "regresion_logistica": (  # referencia lineal
            pipe(LogisticRegression(max_iter=3000)),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "svm": (
            pipe(SVC(kernel="rbf", random_state=config.RANDOM_SEED)),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.03, 0.003]},
        ),
        "random_forest": (
            pipe(RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_SEED, n_jobs=1)),
            {"clf__max_depth": [None, 12, 20], "clf__min_samples_leaf": [1, 3], "clf__max_features": ["sqrt", 0.3]},
        ),
    }


def tune(estimator, param_grid, X, y, groups=None, sample_weight=None, cv: int = 5):
    """Ajuste de hiperparámetros con validación cruzada (GridSearchCV, F1-macro).

    Con `groups` los pliegues son estratificados **por grupo** (las copias aumentadas de una misma
    foto nunca quedan repartidas entre entrenamiento y validación del pliegue).
    """
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

    splitter = (StratifiedGroupKFold(cv, shuffle=True, random_state=config.RANDOM_SEED) if groups is not None
                else StratifiedKFold(cv, shuffle=True, random_state=config.RANDOM_SEED))
    search = GridSearchCV(estimator, param_grid, cv=splitter, scoring="f1_macro", n_jobs=-1, return_train_score=True)
    fit_params = {} if sample_weight is None else {"clf__sample_weight": sample_weight}
    search.fit(X, y, groups=groups, **fit_params)
    return search
