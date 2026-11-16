"""Diagnóstico de atajos: ¿algo distinto de la fruta predice la clase de calidad?

Fase 3/5 de CRISP-DM. Regresiones logísticas que ven solo información "prohibida" y comparan su
F1-macro en validación con un clasificador al azar:
- **fondo:** anillo exterior de BORDER px de la imagen (casi siempre fondo).
- **silueta:** área, caja envolvente, centroide y forma de la máscara (encuadre / zoom), sin color.
Si alguno predice la clase, un modelo puede aprender eso en lugar de la calidad de la fruta.

Uso:  python -m src.evaluation.shortcut
"""
from __future__ import annotations

import csv
from functools import partial

import numpy as np

from src import config
from src.data.preprocess import cache_path

BORDER = 10


def border_features(image_id: str, variant: str = "img") -> np.ndarray:
    """Media, desviación e histogramas HSV del anillo exterior de la imagen."""
    from PIL import Image  # import diferido

    a = np.asarray(Image.open(cache_path(image_id, variant=variant)).convert("HSV"), dtype=np.float32)
    b = BORDER
    ring = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                           a[b:-b, :b].reshape(-1, 3), a[b:-b, -b:].reshape(-1, 3)])
    hists = [np.histogram(ring[:, c], bins=8, range=(0, 256))[0] / len(ring) for c in range(3)]
    return np.concatenate([ring.mean(0) / 255, ring.std(0) / 255, *hists])


def silhouette_features(image_id: str, variant: str = "mask") -> np.ndarray:
    """Área, caja envolvente, centroide, n.º de componentes y circularidad de la máscara."""
    import cv2
    from PIL import Image

    mask = (np.asarray(Image.open(cache_path(image_id, variant=variant))) > 127).astype(np.uint8)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(9)
    n_comp = cv2.connectedComponents(mask)[0] - 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True) or 1.0
    circularity = 4 * np.pi * cv2.contourArea(c) / perimeter ** 2
    return np.array([mask.mean(), (xs.max() - xs.min()) / w, (ys.max() - ys.min()) / h,
                     xs.mean() / w, ys.mean() / h, xs.std() / w, ys.std() / h, n_comp, circularity])


def shortcut_f1(rows: list[dict], features) -> tuple[float, float]:
    """(F1-macro al azar, F1-macro con `features`) en val, entrenando con train."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    X_tr = np.stack([features(r["id"]) for r in train])
    X_va = np.stack([features(r["id"]) for r in val])
    y_tr, y_va = [r["calidad"] for r in train], [r["calidad"] for r in val]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X_tr, y_tr)
    dummy = DummyClassifier(strategy="stratified", random_state=config.RANDOM_SEED).fit(X_tr, y_tr)
    return (f1_score(y_va, dummy.predict(X_va), average="macro"),
            f1_score(y_va, clf.predict(X_va), average="macro"))


def main() -> None:
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tests = {
        "fondo (original)": partial(border_features, variant="img"),
        "fondo (segmentada)": partial(border_features, variant="seg"),
        "silueta": silhouette_features,
        "silueta (recortada)": partial(silhouette_features, variant="cropmask"),
    }
    subsets = [("todas", rows),
               ("kaggle", [r for r in rows if r["fuente"] == "kaggle"]),
               ("hdf", [r for r in rows if r["fuente"] == "hdf"])]
    print(f"{'prueba':20s} {'subconjunto':12s} {'F1 azar':>8s} {'F1 atajo':>9s}")
    for test_name, features in tests.items():
        for name, subset in subsets:
            chance, f1 = shortcut_f1(subset, features)
            print(f"{test_name:20s} {name:12s} {chance:8.3f} {f1:9.3f}")


if __name__ == "__main__":
    main()
