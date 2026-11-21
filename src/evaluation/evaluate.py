"""Evaluación de un modelo entrenado: métricas, intervalos de confianza y errores — Fase 5 de CRISP-DM.

- Métricas de `evaluate_split` (src/training/train.py): F1-macro, exactitud, precisión/recall/F1 por
  clase, F1 por fuente y matriz de confusión; además F1 por fruta.
- **Intervalos de confianza al 95 %** por *bootstrap por grupos*: se remuestrean grupos (no
  imágenes) con reemplazo, porque las imágenes de un mismo grupo no son independientes. Con las 81
  fotos propias el intervalo es ancho; hay que reportarlo junto a la cifra.
- Lista de errores (imagen, clase real y predicha) para el análisis de errores.

`test` y `externa` son la evaluación final: se corren **una sola vez**, con el modelo ya elegido,
y exigen `--final`. Durante el desarrollo se usa `val`.

Uso:  python -m src.evaluation.evaluate --model svm                       (val)
      python -m src.evaluation.evaluate --model cnn_estable_0 --split test --final
Salida: experiments/results/eval_<modelo>_<split>.json y errores_<modelo>_<split>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime

import numpy as np

from src import config

FINAL_SPLITS = ("test", "externa")
N_BOOT = 2000


def predict(model_name: str, split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(y_true, y_pred, filas del índice) del modelo guardado en experiments/checkpoints/."""
    from src.training.train import labels

    keras_path = config.CHECKPOINTS_DIR / f"{model_name}.keras"
    joblib_path = config.CHECKPOINTS_DIR / f"{model_name}.joblib"
    if joblib_path.exists():  # ML tradicional: vector de características
        import joblib

        from src.features.extract_features import load_features

        X, rows = load_features(split)
        y_pred = joblib.load(joblib_path).predict(X)
    elif keras_path.exists():  # CNN: imagen recortada
        import tensorflow as tf

        from src.data.preprocess import load_split

        X, rows = load_split(split)
        y_pred = tf.keras.models.load_model(keras_path).predict(X, batch_size=256, verbose=0).argmax(1)
    else:
        raise FileNotFoundError(f"No hay modelo '{model_name}' en {config.CHECKPOINTS_DIR}")
    return labels(rows), np.asarray(y_pred), rows


def f1_macro(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))


def recall_of(cls: int):
    """Recall de una clase; NaN si la clase no aparece (p. ej. en un remuestreo)."""
    def recall(y_true, y_pred) -> float:
        mask = y_true == cls
        return float((y_pred[mask] == cls).mean()) if mask.any() else float("nan")
    return recall


def accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean())


def bootstrap_ci(y_true, y_pred, groups, metric, n_boot: int = N_BOOT, alpha: float = 0.05,
                 seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """Intervalo de confianza (percentiles) remuestreando **grupos** con reemplazo."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate(model_name: str, split: str = "val") -> dict:
    """Evalúa, guarda JSON + CSV de errores y devuelve el resultado."""
    from src.training.train import evaluate_split, save_results

    y_true, y_pred, rows = predict(model_name, split)
    groups = [r["grupo"] for r in rows]
    podrida = config.QUALITY_CLASSES.index("Podrida")
    result = {"modelo": model_name, "split": split, "imagenes": len(rows), "grupos": len(set(groups)),
              "fecha": f"{datetime.now():%Y-%m-%d %H:%M}", **evaluate_split(y_true, y_pred, rows)}
    result["ic95"] = {
        "f1_macro": bootstrap_ci(y_true, y_pred, groups, f1_macro),
        "accuracy": bootstrap_ci(y_true, y_pred, groups, accuracy),
        "recall_podrida": bootstrap_ci(y_true, y_pred, groups, recall_of(podrida)),
    }
    fruta = np.array([r["fruta"] for r in rows])
    result["por_fruta"] = {f: {"imagenes": int((fruta == f).sum()), "f1_macro": f1_macro(y_true[fruta == f], y_pred[fruta == f])}
                           for f in sorted(set(fruta))}
    save_results(f"eval_{model_name}_{split}", result)

    with open(config.RESULTS_DIR / f"errores_{model_name}_{split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ruta", "fuente", "fruta", "real", "predicha"])
        for r, t, p in zip(rows, y_true, y_pred):
            if t != p:
                writer.writerow([r["id"], r["ruta"], r["fuente"], r["fruta"],
                                 config.QUALITY_CLASSES[t], config.QUALITY_CLASSES[p]])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de un modelo guardado")
    parser.add_argument("--model", required=True, help="nombre en experiments/checkpoints/ (sin extensión)")
    parser.add_argument("--split", default="val", choices=["train", "val", *FINAL_SPLITS])
    parser.add_argument("--final", action="store_true", help="confirma la evaluación final en test/externa")
    args = parser.parse_args()
    if args.split in FINAL_SPLITS and not args.final:
        parser.error(f"'{args.split}' es la evaluación final: se corre una sola vez con el modelo elegido. "
                     "Agregar --final para confirmarlo.")

    r = evaluate(args.model, args.split)
    ci = r["ic95"]
    print(f"{r['modelo']} · {r['split']} · {r['imagenes']} imágenes ({r['grupos']} grupos)")
    print(f"  F1-macro       {r['f1_macro']:.3f}  IC95 [{ci['f1_macro'][0]:.3f}, {ci['f1_macro'][1]:.3f}]")
    print(f"  exactitud      {r['accuracy']:.3f}  IC95 [{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]")
    rp = r["por_clase"]["Podrida"]["recall"]
    print(f"  recall Podrida {rp:.3f}  IC95 [{ci['recall_podrida'][0]:.3f}, {ci['recall_podrida'][1]:.3f}]")
    print("  por fuente: " + ", ".join(f"{k} {v:.3f}" for k, v in r["por_fuente"].items()))
    print("  por fruta:  " + ", ".join(f"{k} {v['f1_macro']:.3f} (n={v['imagenes']})" for k, v in r["por_fruta"].items()))
    print(json.dumps(r["matriz_confusion"]))


if __name__ == "__main__":
    main()
