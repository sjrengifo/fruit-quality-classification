"""Extracción de características para modelos de ML tradicional.

Fase 3/4 de CRISP-DM. De la fruta recortada sobre blanco (`crop`) y su máscara (`cropmask`) se
calcula el vector φ(x), usando **solo los píxeles de la fruta**:
- color: histogramas H/S/V normalizados + media y desviación por canal;
- textura: histograma LBP uniforme + propiedades GLCM (contraste, homogeneidad, energía, correlación);
- forma: área relativa, relación de aspecto, circularidad, solidez, extensión y momentos de Hu.

Uso:  python -m src.main features   → data/processed/features.npz (incremental)
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src import config

HIST_BINS = {"h": 16, "s": 8, "v": 8}
LBP_POINTS, LBP_RADIUS = 8, 1
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 3)
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")

# nombre del grupo → n.º de características (en el orden de extract_all)
FEATURE_GROUPS = {
    "color": sum(HIST_BINS.values()) + 6,
    "textura": LBP_POINTS + 2 + len(GLCM_DISTANCES) * len(GLCM_PROPS),
    "forma": 5 + 7,
}
FEATURES_PATH = config.PROCESSED_DIR / "features.npz"


def _full_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    mask = np.ones(image.shape[:2], dtype=bool) if mask is None else mask.astype(bool)
    return mask if mask.any() else np.ones(image.shape[:2], dtype=bool)


def extract_color_features(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Histogramas H, S y V de la fruta + media y desviación de cada canal."""
    import cv2  # import diferido

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[_full_mask(image, mask)].astype(np.float32)
    ranges = {"h": 180, "s": 256, "v": 256}
    hists = [np.histogram(hsv[:, i], bins=HIST_BINS[c], range=(0, ranges[c]))[0] / len(hsv)
             for i, c in enumerate("hsv")]
    scale = np.array([180, 255, 255], dtype=np.float32)
    return np.concatenate([*hists, hsv.mean(0) / scale, hsv.std(0) / scale])


def extract_texture_features(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Histograma LBP uniforme y propiedades GLCM (promedio de 4 ángulos), solo dentro de la fruta."""
    import cv2
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    mask = _full_mask(image, mask)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # máscara erosionada: los vecinos de un píxel del contorno caen en el fondo
    inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * LBP_RADIUS + 1,) * 2, np.uint8)).astype(bool)
    inner = inner if inner.any() else mask
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")[inner]
    lbp_hist = np.histogram(lbp, bins=LBP_POINTS + 2, range=(0, LBP_POINTS + 2))[0] / inner.sum()

    # nivel 0 = fondo; se eliminan sus co-ocurrencias para que el contorno no cuente como textura
    levels = np.where(mask, 1 + (gray.astype(np.int32) * (GLCM_LEVELS - 1)) // 256, 0).astype(np.uint8)
    glcm = graycomatrix(levels, GLCM_DISTANCES, [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=GLCM_LEVELS, symmetric=True).astype(np.float64)
    glcm[0], glcm[:, 0] = 0, 0
    glcm /= np.maximum(glcm.sum(axis=(0, 1), keepdims=True), 1)
    props = [graycoprops(glcm, prop).mean(axis=1) for prop in GLCM_PROPS]
    return np.concatenate([lbp_hist, *props])


def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """Área relativa, aspecto, circularidad, solidez, extensión y momentos de Hu (log) de la máscara."""
    import cv2

    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(FEATURE_GROUPS["forma"])
    c = max(contours, key=cv2.contourArea)
    area = max(cv2.contourArea(c), 1.0)
    x, y, w, h = cv2.boundingRect(c)
    perimeter = cv2.arcLength(c, True) or 1.0
    hull = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return np.concatenate([[mask.mean(), w / h, 4 * np.pi * area / perimeter ** 2, area / hull, area / (w * h)], hu])


def extract_all(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Concatena todas las características en un único vector φ(x) (orden de FEATURE_GROUPS)."""
    mask = _full_mask(image, mask)
    return np.concatenate([
        extract_color_features(image, mask),
        extract_texture_features(image, mask),
        extract_shape_features(mask),
    ]).astype(np.float32)


def group_columns(groups: list[str]) -> np.ndarray:
    """Índices de columnas de los grupos pedidos (para estudios de ablación)."""
    cols, start = [], 0
    for name, n in FEATURE_GROUPS.items():
        if name in groups:
            cols.extend(range(start, start + n))
        start += n
    return np.array(cols)


def _process(image_id: str) -> np.ndarray:
    from PIL import Image

    from src.data.preprocess import cache_path

    image = np.asarray(Image.open(cache_path(image_id, variant="crop")).convert("RGB"))
    mask = np.asarray(Image.open(cache_path(image_id, variant="cropmask"))) > 127
    return extract_all(image, mask)


def build_features(workers: int | None = None) -> int:
    """Calcula φ(x) de las imágenes del índice que falten y guarda todo en FEATURES_PATH."""
    from tqdm import tqdm

    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    done = {}
    if FEATURES_PATH.exists():
        old = np.load(FEATURES_PATH)
        done = dict(zip(old["ids"], old["X"]))
    todo = [i for i in ids if i not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for image_id, x in zip(todo, tqdm(pool.map(_process, todo, chunksize=64), total=len(todo), desc="Características")):
            done[image_id] = x
    np.savez_compressed(FEATURES_PATH, ids=np.array(ids), X=np.stack([done[i] for i in ids]))
    return len(todo)


def load_features(split: str | None = None) -> tuple[np.ndarray, list[dict]]:
    """(X, filas del índice) de un split (o de todo), en el orden del índice."""
    data = np.load(FEATURES_PATH)
    pos = {i: k for k, i in enumerate(data["ids"])}
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (split is None or r["split"] == split) and r["id"] in pos]
    return data["X"][[pos[r["id"]] for r in rows]], rows


def main() -> None:
    n = build_features()
    print(f"{n} imágenes nuevas · {sum(FEATURE_GROUPS.values())} características → {FEATURES_PATH}")


if __name__ == "__main__":
    main()
