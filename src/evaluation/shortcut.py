"""Diagnóstico de atajos: ¿el fondo de la imagen predice la clase de calidad?

Fase 3/5 de CRISP-DM. Entrena una regresión logística que solo ve el anillo exterior de la imagen
(BORDER px, casi siempre fondo) y compara su F1-macro en validación con un clasificador al azar.
Si el fondo predice la clase, un modelo puede aprender el fondo en lugar de la calidad de la fruta.

Uso:  python -m src.evaluation.shortcut
"""
from __future__ import annotations

import csv

import numpy as np

from src import config
from src.data.preprocess import cache_path

BORDER = 10


def border_features(image_id: str) -> np.ndarray:
    """Media, desviación e histogramas HSV del anillo exterior de la imagen preprocesada."""
    from PIL import Image  # import diferido

    a = np.asarray(Image.open(cache_path(image_id)).convert("HSV"), dtype=np.float32)
    b = BORDER
    ring = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                           a[b:-b, :b].reshape(-1, 3), a[b:-b, -b:].reshape(-1, 3)])
    hists = [np.histogram(ring[:, c], bins=8, range=(0, 256))[0] / len(ring) for c in range(3)]
    return np.concatenate([ring.mean(0) / 255, ring.std(0) / 255, *hists])


def background_f1(rows: list[dict]) -> tuple[float, float]:
    """(F1-macro al azar, F1-macro solo-fondo) en val, entrenando con train."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    X_tr = np.stack([border_features(r["id"]) for r in train])
    X_va = np.stack([border_features(r["id"]) for r in val])
    y_tr, y_va = [r["calidad"] for r in train], [r["calidad"] for r in val]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    dummy = DummyClassifier(strategy="stratified", random_state=config.RANDOM_SEED).fit(X_tr, y_tr)
    return (f1_score(y_va, dummy.predict(X_va), average="macro"),
            f1_score(y_va, clf.predict(X_va), average="macro"))


def main() -> None:
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{'subconjunto':12s} {'F1 azar':>8s} {'F1 solo-fondo':>14s}")
    for name, subset in [("todas", rows),
                         ("kaggle", [r for r in rows if r["fuente"] == "kaggle"]),
                         ("hdf", [r for r in rows if r["fuente"] == "hdf"])]:
        chance, background = background_f1(subset)
        print(f"{name:12s} {chance:8.3f} {background:14.3f}")


if __name__ == "__main__":
    main()
