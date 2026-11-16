"""Orquestación del entrenamiento de modelos (ML y CNN).

Fase 4 de CRISP-DM. Guarda artefactos en experiments/ (modelos en checkpoints/, métricas en results/).
Uso:  python -m src.training.train --model {classic,cnn,cnn_search}

Se ajusta con `train`, se compara con `val`; `test` y `externa` quedan intactos para la Fase 5.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from src import config


def sample_weights(rows: list[dict], power: float = 1.0) -> np.ndarray:
    """Peso por muestra = 1/tamaño del grupo, reescalado para que cada clase pese lo mismo.

    Así 30 copias aumentadas de una foto cuentan como una foto, y las clases minoritarias
    (Defectuosa, Podrida) no quedan opacadas por Buena. Media de los pesos = 1.
    """
    group_size = Counter(r["grupo"] for r in rows)
    w = np.array([group_size[r["grupo"]] ** -power for r in rows])
    y = np.array([r["calidad"] for r in rows])
    for c in set(y):
        w[y == c] /= w[y == c].sum()
    return w * len(w) / w.sum()


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([config.QUALITY_CLASSES.index(r["calidad"]) for r in rows])


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, rows: list[dict]) -> dict:
    """F1-macro, exactitud, métricas por clase, por fuente y matriz de confusión."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    idx = list(range(len(config.QUALITY_CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    out = {
        "f1_macro": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "por_clase": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i, c in enumerate(config.QUALITY_CLASSES)},
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
        "por_fuente": {},
    }
    fuente = np.array([row["fuente"] for row in rows])
    for s in sorted(set(fuente)):
        m = fuente == s
        out["por_fuente"][s] = float(f1_score(y_true[m], y_pred[m], labels=idx, average="macro", zero_division=0))
    return out


def save_results(name: str, results: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def train_classic() -> dict:
    """Líneas base + modelos de ML tradicional ajustados con GridSearchCV (pliegues por grupo)."""
    import joblib
    import pandas as pd

    from src.features.extract_features import load_features
    from src.models.classic_ml import build_baselines, build_models, tune

    X_tr, rows_tr = load_features("train")
    X_va, rows_va = load_features("val")
    y_tr, y_va = labels(rows_tr), labels(rows_va)
    w_tr = sample_weights(rows_tr)
    groups = np.array([r["grupo"] for r in rows_tr])
    results = {}

    for name, estimator in build_baselines().items():
        estimator.fit(X_tr, y_tr)
        results[name] = {"val": evaluate_split(y_va, estimator.predict(X_va), rows_va)}
        print(f"{name:22s} F1-macro val = {results[name]['val']['f1_macro']:.3f}")

    for name, (estimator, grid) in build_models().items():
        search = tune(estimator, grid, X_tr, y_tr, groups=groups, sample_weight=w_tr)
        best = search.best_estimator_
        results[name] = {
            "mejores_hiperparametros": search.best_params_,
            "f1_macro_cv": float(search.best_score_),
            "f1_macro_cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "train": evaluate_split(y_tr, best.predict(X_tr), rows_tr),
            "val": evaluate_split(y_va, best.predict(X_va), rows_va),
        }
        joblib.dump(best, config.CHECKPOINTS_DIR / f"{name}.joblib")
        pd.DataFrame(search.cv_results_).to_csv(config.RESULTS_DIR / f"cv_{name}.csv", index=False)
        print(f"{name:22s} F1-macro cv = {search.best_score_:.3f} · val = {results[name]['val']['f1_macro']:.3f}"
              f" · {search.best_params_}")

    save_results("classic", results)
    return results


def train_cnn(name: str = "cnn", epochs: int | None = None, **overrides) -> dict:
    """Entrena la CNN con pesos por muestra; conserva la época con mejor F1-macro en validación."""
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import f1_score

    from src.data.preprocess import load_split
    from src.models.cnn import CNN_DEFAULT, build_cnn

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    X_tr, rows_tr = load_split("train")
    X_va, rows_va = load_split("val")
    y_tr, y_va = labels(rows_tr), labels(rows_va)
    w_tr = sample_weights(rows_tr).astype("float32")
    checkpoint = config.CHECKPOINTS_DIR / f"{name}.keras"

    class BestF1(tf.keras.callbacks.Callback):
        """F1-macro en validación por época; guarda el mejor modelo y detiene si no mejora."""

        def __init__(self, patience: int = 8):
            super().__init__()
            self.best, self.wait, self.patience = -1.0, 0, patience

        def on_epoch_end(self, epoch, logs=None):
            pred = self.model.predict(X_va, batch_size=256, verbose=0).argmax(1)
            f1 = f1_score(y_va, pred, average="macro")
            logs["val_f1_macro"] = f1
            if f1 > self.best:
                self.best, self.wait = f1, 0
                self.model.save(checkpoint)
            else:
                self.wait += 1
                self.model.stop_training = self.wait >= self.patience

    model = build_cnn(**overrides)
    data = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr, w_tr))
            .shuffle(len(X_tr), seed=config.RANDOM_SEED).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    history = model.fit(
        data, epochs=epochs or config.EPOCHS, verbose=2,
        callbacks=[BestF1(), tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-5)],
    )
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(config.LOGS_DIR / f"{name}_historia.csv", index_label="epoca")

    best = tf.keras.models.load_model(checkpoint)
    predict = lambda X: best.predict(X, batch_size=256, verbose=0).argmax(1)  # noqa: E731
    results = {
        "configuracion": {**CNN_DEFAULT, **overrides},
        "parametros": int(best.count_params()),
        "epocas": len(history.history["loss"]),
        "train": evaluate_split(y_tr, predict(X_tr), rows_tr),
        "val": evaluate_split(y_va, predict(X_va), rows_va),
    }
    save_results(name, results)
    print(f"{name}: F1-macro train = {results['train']['f1_macro']:.3f} · val = {results['val']['f1_macro']:.3f}")
    return results


def search_cnn() -> None:
    """Entrena cada configuración de CNN_GRID y resume el F1-macro en validación."""
    from src.models.cnn import CNN_GRID

    summary = []
    for i, overrides in enumerate(CNN_GRID):
        r = train_cnn(name=f"cnn_busqueda_{i}", **overrides)
        summary.append({"configuracion": overrides, "epocas": r["epocas"], "parametros": r["parametros"],
                        "f1_train": r["train"]["f1_macro"], "f1_val": r["val"]["f1_macro"],
                        "f1_val_por_fuente": r["val"]["por_fuente"]})
        save_results("cnn_busqueda", summary)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos")
    parser.add_argument("--model", choices=["cnn", "cnn_search", "classic"], default="cnn")
    args = parser.parse_args()

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.model == "cnn":
        train_cnn()
    elif args.model == "cnn_search":
        search_cnn()
    else:
        train_classic()


if __name__ == "__main__":
    main()
