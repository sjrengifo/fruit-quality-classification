"""Preprocesamiento de imágenes: formato, orientación y tamaño unificados.

Fase 3 de CRISP-DM (Preparación de los datos). Todas las imágenes pasan por la misma función
`prepare_image` (también la app en el despliegue), para que ninguna clase se distinga por
propiedades del archivo: en Healthy-Defective-Fruits la mitad de `rot` viene en PNG, y en Kaggle hay
imágenes de 256×256 hasta 8000×6000.

Pasos: orientación EXIF → RGB → lado mayor a IMAGE_SIZE (sin deformar) → relleno hasta cuadrado
replicando el borde (no un color fijo, que podría volverse un atajo) → caché PNG sin pérdida en
data/interim/img<N>/<id>.png.

Uso:  python -m src.main preprocess
"""
from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src import config


def prepare_image(source, size: tuple[int, int] | None = None) -> np.ndarray:
    """Imagen (ruta, archivo o PIL.Image) → np.ndarray uint8 (alto, ancho, 3) en RGB."""
    from PIL import Image, ImageOps  # import diferido

    size = size or config.IMAGE_SIZE
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img.draft("RGB", (size[1] * 2, size[0] * 2))  # JPEG grandes: decodifica ya reducido (mucho más rápido)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((size[1], size[0]), Image.Resampling.LANCZOS)

    arr = np.asarray(img)
    pad_h, pad_w = size[0] - arr.shape[0], size[1] - arr.shape[1]
    pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
    return np.pad(arr, pad, mode="edge")


def normalize(image: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] → float32 [0, 1] (entrada de la CNN)."""
    return image.astype("float32") / 255.0


def resize_and_normalize(image, size: tuple[int, int] | None = None) -> np.ndarray:
    """Atajo para inferencia: prepare_image + normalize."""
    return normalize(prepare_image(image, size))


def cache_dir(size: tuple[int, int] | None = None, variant: str = "img") -> Path:
    """Carpeta de caché: `img` (preprocesada), `seg` (sin fondo), `mask` (máscara), `crop` (fruta
    recortada y centrada) o `cropmask` (su máscara)."""
    size = size or config.IMAGE_SIZE
    return config.INTERIM_DIR / f"{variant}{size[0]}"


def cache_path(image_id: str, size: tuple[int, int] | None = None, variant: str = "img") -> Path:
    return cache_dir(size, variant) / f"{image_id}.png"


def _process(task: tuple[str, str, tuple[int, int]]) -> str | None:
    """Procesa una imagen; devuelve un mensaje de error o None."""
    from PIL import Image

    ruta, image_id, size = task
    out = cache_path(image_id, size)
    if out.exists():
        return None
    try:
        Image.fromarray(prepare_image(config.DATA_DIR / ruta, size)).save(out)
    except Exception as exc:  # una imagen corrupta no debe detener todo el lote
        return f"{ruta}: {exc}"
    return None


def preprocess_index(index_path: Path | None = None, size: tuple[int, int] | None = None,
                     workers: int | None = None) -> tuple[int, list[str]]:
    """Genera la caché de todas las imágenes del índice (incremental: omite las ya procesadas)."""
    size = size or config.IMAGE_SIZE
    with open(index_path or config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cache_dir(size).mkdir(parents=True, exist_ok=True)
    tasks = [(r["ruta"], r["id"], size) for r in rows if not cache_path(r["id"], size).exists()]

    from tqdm import tqdm  # import diferido

    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = [e for e in tqdm(pool.map(_process, tasks, chunksize=64), total=len(tasks),
                                  desc="Preprocesando") if e]
    return len(tasks), errors


def load_split(split: str, index_path: Path | None = None, size: tuple[int, int] | None = None,
               variant: str = "crop"):
    """Carga un split desde la caché: (X uint8 [n, alto, ancho, 3], filas del índice).

    Por defecto usa las frutas recortadas sobre blanco (`crop`); `seg` sin recortar, `img` originales.
    """
    from PIL import Image

    with open(index_path or config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    X = np.stack([np.asarray(Image.open(cache_path(r["id"], size, variant))) for r in rows]) if rows else \
        np.empty((0, *(size or config.IMAGE_SIZE), 3), dtype=np.uint8)
    return X, rows


def main() -> None:
    n, errors = preprocess_index()
    for e in errors:
        print(f"✗ {e}")
    print(f"{n - len(errors)} imágenes nuevas en {cache_dir()} · {len(errors)} error(es).")


if __name__ == "__main__":
    main()
