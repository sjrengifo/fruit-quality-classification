"""Orquestación del entrenamiento de modelos (ML y CNN).

Fase 4 de CRISP-DM. Guarda artefactos en experiments/ (modelos en checkpoints/, métricas en results/).
Uso:  python -m src.training.train --model {classic,cnn,cnn_search,cnn_stable,transfer,tl_busqueda,tl_entre_fuentes}
      (las búsquedas de CNN y transfer learning se corren en Colab con GPU: notebooks/colab_entrenar_cnn)

Se ajusta con `train`, se compara con `val`; `test` y `externa` quedan intactos para la Fase 5.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

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
    """Índices de clase (0 Buena, 1 Defectuosa, 2 Podrida) de las filas del índice."""
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
    """Guarda un diccionario de resultados como `experiments/results/<name>.json`."""
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


def index_path(indice: str | None = None) -> Path:
    """Índice con el que se entrena: el original o uno de etiquetas revisadas ("v3", "v4"; src/data/revision.py)."""
    return {"v3": config.DATASET_INDEX_V3, "v4": config.DATASET_INDEX_V4}.get(indice, config.DATASET_INDEX)


def load_training_data(extra: bool = False, holdout: str | None = None, indice: str | None = None,
                       frutas: list[str] | None = None):
    """(X_tr, rows_tr, X_va, rows_va) de las imágenes recortadas.

    - `extra`: agrega las fuentes solo-entrenamiento (split "extra", p. ej. FruQ).
    - `holdout`: deja esa fuente fuera del entrenamiento y valida **solo** con sus imágenes de val
      (generalización a una fuente no vista; test y externa no se tocan).
    - `indice`: "v3"/"v4" usan las etiquetas revisadas (train y val); por defecto, el índice original.
    - `frutas`: solo esas frutas, en train y en val (4.ª iteración).
    """
    from src.data.preprocess import load_split

    path = index_path(indice)
    X_tr, rows_tr = load_split("train", path)
    if extra:
        X_ex, rows_ex = load_split("extra", path)
        X_tr, rows_tr = np.concatenate([X_tr, X_ex]), rows_tr + rows_ex
    X_va, rows_va = load_split("val", path)
    if holdout:
        keep_tr = np.array([r["fuente"] != holdout for r in rows_tr])
        keep_va = np.array([r["fuente"] == holdout for r in rows_va])
        X_tr, rows_tr = X_tr[keep_tr], [r for r, k in zip(rows_tr, keep_tr) if k]
        X_va, rows_va = X_va[keep_va], [r for r, k in zip(rows_va, keep_va) if k]
    if frutas:
        keep_tr = np.array([r["fruta"] in frutas for r in rows_tr])
        keep_va = np.array([r["fruta"] in frutas for r in rows_va])
        X_tr, rows_tr = X_tr[keep_tr], [r for r, k in zip(rows_tr, keep_tr) if k]
        X_va, rows_va = X_va[keep_va], [r for r, k in zip(rows_va, keep_va) if k]
    return X_tr, rows_tr, X_va, rows_va


def _best_f1_callback(X_va, y_va, checkpoint: Path, patience: int | None):
    """Callback: F1-macro en validación por época; guarda el mejor modelo y detiene si no mejora."""
    import time

    import tensorflow as tf
    from sklearn.metrics import f1_score

    class BestF1(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.wait = -1.0, 0
            self.start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            """F1-macro de validación por época: guarda el mejor modelo y detiene si deja de mejorar."""
            pred = self.model.predict(X_va, batch_size=512, verbose=0).argmax(1)
            f1 = f1_score(y_va, pred, average="macro")
            logs["val_f1_macro"] = f1
            logs["segundos"] = time.time() - self.start
            if f1 > self.best:
                self.best, self.wait = f1, 0
                self.model.save(checkpoint)
            else:
                self.wait += 1
                self.model.stop_training = patience is not None and self.wait >= patience

    return BestF1()


def _report(name, cfg, hist, checkpoint, gpu, minutes, data) -> dict:
    """Recarga la mejor época, evalúa train/val y guarda historia + resultados."""
    import tensorflow as tf

    X_tr, rows_tr, X_va, rows_va = data
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    hist.to_csv(config.LOGS_DIR / f"{name}_historia.csv", index_label="epoca")
    best = tf.keras.models.load_model(checkpoint)
    predict = lambda X: best.predict(X, batch_size=512, verbose=0).argmax(1)  # noqa: E731
    results = {
        "configuracion": cfg,
        "parametros": int(best.count_params()),
        "epocas": len(hist),
        "mejor_epoca": int(hist["val_f1_macro"].idxmax()) + 1,
        # estabilidad: un F1 alto en una sola época puede ser suerte; la media final es más fiable
        "f1_val_ultimas_5_media": float(hist["val_f1_macro"].tail(5).mean()),
        "f1_val_ultimas_5_desv": float(hist["val_f1_macro"].tail(5).std()),
        "dispositivo": "GPU" if gpu else "CPU",
        "minutos": round(minutes, 1),
        "train": evaluate_split(labels(rows_tr), predict(X_tr), rows_tr),
        "val": evaluate_split(labels(rows_va), predict(X_va), rows_va),
    }
    save_results(name, results)
    print(f"{name}: F1-macro train = {results['train']['f1_macro']:.3f} · val = {results['val']['f1_macro']:.3f}"
          f" (media últimas 5 = {results['f1_val_ultimas_5_media']:.3f}) · {results['minutos']} min")
    return results


def _setup_tf():
    """Semilla fija y precisión mixta (float16) si hay GPU: en la T4 de Colab es ~2× más rápida."""
    import tensorflow as tf

    gpu = bool(tf.config.list_physical_devices("GPU"))
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if gpu else "float32")
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    return gpu


def _dataset(X, y, w, batch_size: int):
    import tensorflow as tf

    return (tf.data.Dataset.from_tensor_slices((X, y, w))
            .shuffle(len(X), seed=config.RANDOM_SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE))


def train_cnn(name: str = "cnn", epochs: int | None = None, extra: bool = False, holdout: str | None = None,
              indice: str | None = None, frutas: list[str] | None = None, **overrides) -> dict:
    """Entrena la CNN desde cero con pesos por muestra; conserva la época con mejor F1-macro en validación."""
    import math
    import time

    import pandas as pd
    import tensorflow as tf

    from src.models.cnn import CNN_DEFAULT, build_cnn

    overrides.pop("modelo", None)
    cfg = {**CNN_DEFAULT, **overrides, "extra": extra, "holdout": holdout, "indice": indice,
           "frutas": frutas}
    epochs = epochs or config.EPOCHS
    gpu = _setup_tf()
    data = X_tr, rows_tr, X_va, rows_va = load_training_data(extra, holdout, indice, frutas)
    w_tr = sample_weights(rows_tr, cfg["weight_power"]).astype("float32")
    checkpoint = config.CHECKPOINTS_DIR / f"{name}.keras"

    # con coseno la tasa baja hasta 0 al final: se entrenan todas las épocas (sin parada temprana)
    callbacks = [_best_f1_callback(X_va, labels(rows_va), checkpoint,
                                   patience=None if cfg["schedule"] == "cosine" else 8)]
    if cfg["schedule"] == "plateau":
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-5))
    model = build_cnn(decay_steps=epochs * math.ceil(len(X_tr) / cfg["batch_size"]), **overrides)
    start = time.time()
    history = model.fit(_dataset(X_tr, labels(rows_tr), w_tr, cfg["batch_size"]), epochs=epochs, verbose=2,
                        callbacks=callbacks)
    return _report(name, cfg, pd.DataFrame(history.history), checkpoint, gpu, (time.time() - start) / 60, data)


def train_transfer(name: str = "tl", **overrides) -> dict:
    """Transfer learning en dos etapas (src/models/transfer.py): cabeza con la base congelada y luego
    ajuste fino de toda la red con decaimiento coseno. Conserva la época con mejor F1-macro en val."""
    import math
    import time

    import pandas as pd
    import tensorflow as tf

    from src.models.transfer import TRANSFER_DEFAULT, build_transfer

    cfg = {**TRANSFER_DEFAULT, **overrides}
    gpu = _setup_tf()
    data = X_tr, rows_tr, X_va, rows_va = load_training_data(cfg["extra"], cfg["holdout"], cfg["indice"], cfg["frutas"])
    y_tr, w_tr = labels(rows_tr), sample_weights(rows_tr, cfg["weight_power"]).astype("float32")
    checkpoint = config.CHECKPOINTS_DIR / f"{name}.keras"
    train_data = _dataset(X_tr, y_tr, w_tr, cfg["batch_size"])
    best_f1 = _best_f1_callback(X_va, labels(rows_va), checkpoint, patience=None)
    model, base = build_transfer(**cfg)
    start = time.time()

    # Etapa 1: base congelada, solo la cabeza
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr_head"]), loss="sparse_categorical_crossentropy",
                  weighted_metrics=["accuracy"])
    h1 = model.fit(train_data, epochs=cfg["head_epochs"], verbose=2, callbacks=[best_f1])

    # Etapa 2: ajuste fino de toda la base (BatchNorm sigue en modo inferencia)
    base.trainable = True
    steps = cfg["fine_epochs"] * math.ceil(len(X_tr) / cfg["batch_size"])
    model.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(cfg["lr_fine"], steps)),
                  loss="sparse_categorical_crossentropy", weighted_metrics=["accuracy"])
    h2 = model.fit(train_data, epochs=cfg["fine_epochs"], verbose=2, callbacks=[best_f1])

    hist = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
    hist["etapa"] = ["cabeza"] * len(h1.history["loss"]) + ["ajuste fino"] * len(h2.history["loss"])
    return _report(name, cfg, hist, checkpoint, gpu, (time.time() - start) / 60, data)


def all_searches() -> dict[str, list[dict]]:
    """Búsquedas de src/models/cnn.py y src/models/transfer.py (cada configuración puede indicar
    "modelo": "cnn" | "transfer"; por defecto la CNN desde cero)."""
    from src.models import cnn, transfer

    return {**cnn.SEARCHES, **transfer.SEARCHES}


def train_config(name: str, overrides: dict) -> dict:
    """Entrena una configuración con el modelo que indica su clave "modelo"."""
    if overrides.get("modelo") == "transfer":
        return train_transfer(name=name, **overrides)
    return train_cnn(name=name, **overrides)


def _train_in_subprocess(search: str, i: int) -> dict:
    """Entrena una configuración en un proceso aparte (su memoria se libera completa al terminar) y
    muestra su salida línea por línea."""
    import subprocess
    import sys

    cmd = [sys.executable, "-u", "-m", "src.training.train", "--busqueda", search, "--indice", str(i)]
    with subprocess.Popen(cmd, cwd=config.ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{search}_{i} terminó con código {proc.returncode}")
    with open(config.RESULTS_DIR / f"{search}_{i}.json", encoding="utf-8") as f:
        return json.load(f)


def search_cnn(search: str = "cnn_busqueda", sync_dir: Path | None = None, isolate: bool = False) -> list[dict]:
    """Entrena cada configuración de una búsqueda (`all_searches()`) y la resume.

    Las configuraciones con resultados ya guardados se saltan (permite retomar si se corta la
    sesión). Con `sync_dir`, experiments/ se copia ahí después de cada configuración (Google Drive
    en Colab). Con `isolate`, cada configuración corre en su propio proceso: TensorFlow no libera
    toda la memoria entre modelos y en Colab (12 GB de RAM) la sesión se caía tras varios.
    """
    import shutil

    summary = []
    for i, overrides in enumerate(all_searches()[search]):
        name = f"{search}_{i}"
        path = config.RESULTS_DIR / f"{name}.json"
        r = None
        if path.exists():
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            # un resultado guardado de otro tipo de modelo (p. ej. una versión anterior de la búsqueda) se reentrena
            if r["configuracion"].get("modelo", "cnn") != overrides.get("modelo", "cnn"):
                print(f"{name}: el resultado guardado es de otro modelo, se reentrena")
                r = None
            else:
                print(f"{name}: ya entrenado, se omite")
        if r is None:
            r = _train_in_subprocess(search, i) if isolate else train_config(name, overrides)
        summary.append({"configuracion": overrides, "epocas": r["epocas"], "parametros": r["parametros"],
                        "f1_train": r["train"]["f1_macro"], "f1_val": r["val"]["f1_macro"],
                        "f1_val_ultimas_5_media": r.get("f1_val_ultimas_5_media"),
                        "f1_val_por_fuente": r["val"]["por_fuente"]})
        save_results(search, summary)
        if sync_dir is not None:
            shutil.copytree(config.EXPERIMENTS_DIR, sync_dir, dirs_exist_ok=True)
    return summary


def main():
    """CLI de entrenamiento: modelos clásicos, CNN, transfer learning o una búsqueda completa."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos")
    parser.add_argument("--model", default="cnn",
                        choices=["cnn", "cnn_search", "cnn_stable", "classic", "transfer", "tl_busqueda", "tl_entre_fuentes",
                                 "v3", "v4"])
    parser.add_argument("--busqueda", help="con --indice: entrena solo esa configuración de la búsqueda")
    parser.add_argument("--indice", type=int)
    args = parser.parse_args()

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.busqueda is not None:
        train_config(f"{args.busqueda}_{args.indice}", all_searches()[args.busqueda][args.indice])
    elif args.model == "cnn":
        train_cnn()
    elif args.model == "cnn_search":
        search_cnn("cnn_busqueda")
    elif args.model == "cnn_stable":
        search_cnn("cnn_estable")
    elif args.model == "transfer":
        train_transfer()
    elif args.model.startswith("tl_") or args.model in ("v3", "v4"):
        search_cnn(args.model)
    else:
        train_classic()


if __name__ == "__main__":
    main()
