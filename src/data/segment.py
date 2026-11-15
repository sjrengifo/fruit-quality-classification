"""Segmentación de la fruta y reemplazo del fondo — Fase 3 de CRISP-DM.

Motivo: el fondo predice la clase (F1 solo-fondo 0,67; 0,91 en Healthy-Defective-Fruits, ver
src/evaluation/shortcut.py). Se separa la fruta con U²-Net (Qin et al., 2020) vía `rembg`, un modelo
preentrenado **solo para quitar fondos**; el clasificador de calidad se entrena desde cero.
El fondo se reemplaza por blanco, como en la recolección propia y el enunciado (fondo simple).

Pasos por imagen: prepare_image a SEG_INPUT px → máscara U²-Net (umbral 0,5) → se eliminan
componentes pequeños (< MIN_COMPONENT del área) → fondo blanco → reducción a IMAGE_SIZE.
Cachés: data/interim/seg<N>/<id>.png (imagen) y data/interim/mask<N>/<id>.png (máscara), más
data/interim/seg_stats.csv con el área de la máscara para control de calidad.

Uso:  python -m src.main segment
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src import config

SEG_MODEL = "u2net"
SEG_INPUT = (320, 320)       # resolución de entrada de U²-Net
MIN_COMPONENT = 0.01         # fracción mínima del área para conservar un componente
EMPTY_MASK = 0.02            # máscara con menos área que esto = segmentación fallida
BACKGROUND = 255             # blanco

_session = None


def _get_session():
    """Sesión de rembg por proceso (se crea una sola vez)."""
    global _session
    if _session is None:
        from rembg import new_session  # import diferido

        _session = new_session(SEG_MODEL, providers=["CPUExecutionProvider"])
    return _session


def fruit_mask(image: np.ndarray) -> np.ndarray:
    """Imagen RGB uint8 → máscara booleana de la(s) fruta(s), sin componentes pequeños."""
    import cv2
    from PIL import Image
    from rembg import remove

    alpha = np.asarray(remove(Image.fromarray(image), session=_get_session(), only_mask=True))
    mask = (alpha > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT * mask.size]
    return np.isin(labels, keep)


def remove_background(image: np.ndarray, mask: np.ndarray, value: int = BACKGROUND) -> np.ndarray:
    """Reemplaza por `value` los píxeles fuera de la máscara."""
    return np.where(mask[..., None], image, np.uint8(value))


def segment_image(source, size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Imagen (ruta, archivo o PIL) → (imagen sin fondo, máscara, área relativa) a `size`.

    Si la segmentación falla (máscara casi vacía) se devuelve la imagen sin modificar y máscara llena.
    """
    from PIL import Image

    from src.data.preprocess import prepare_image

    size = size or config.IMAGE_SIZE
    big = prepare_image(source, SEG_INPUT)
    mask = fruit_mask(big)
    area = float(mask.mean())
    if area < EMPTY_MASK:
        mask = np.ones_like(mask)
    out = Image.fromarray(remove_background(big, mask)).resize((size[1], size[0]), Image.Resampling.LANCZOS)
    small_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize((size[1], size[0]), Image.Resampling.NEAREST)
    return np.asarray(out), np.asarray(small_mask) > 127, area


def _process(task: tuple[str, str, tuple[int, int]]) -> tuple[str, float | None, str | None]:
    from PIL import Image

    from src.data.preprocess import cache_path

    ruta, image_id, size = task
    try:
        image, mask, area = segment_image(config.DATA_DIR / ruta, size)
        Image.fromarray(image).save(cache_path(image_id, size, "seg"))
        Image.fromarray(mask.astype(np.uint8) * 255).save(cache_path(image_id, size, "mask"))
        return image_id, area, None
    except Exception as exc:  # una imagen corrupta no debe detener todo el lote
        return image_id, None, f"{ruta}: {exc}"


SEG_WORKERS = 2  # cada proceso carga U²-Net (~1 GB de RAM); más procesos saturan equipos de 16 GB


def _init_worker(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)  # rembg lo usa para los hilos de onnxruntime


def segment_index(size: tuple[int, int] | None = None, workers: int | None = None) -> tuple[int, list[str]]:
    """Segmenta todas las imágenes del índice (incremental: omite las ya segmentadas)."""
    from tqdm import tqdm

    from src.data.preprocess import cache_dir, cache_path

    size = size or config.IMAGE_SIZE
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for variant in ("seg", "mask"):
        cache_dir(size, variant).mkdir(parents=True, exist_ok=True)
    # la máscara se escribe al final: si falta, la imagen no terminó (p. ej. proceso interrumpido)
    tasks = [(r["ruta"], r["id"], size) for r in rows if not cache_path(r["id"], size, "mask").exists()]

    stats_path = config.INTERIM_DIR / "seg_stats.csv"
    new_file = not stats_path.exists() or stats_path.stat().st_size == 0
    workers = workers or SEG_WORKERS
    threads = max(1, (os.cpu_count() or 2) // workers)
    errors = []
    with open(stats_path, "a", newline="", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(threads,)) as pool:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "area"])
        for image_id, area, error in tqdm(pool.map(_process, tasks, chunksize=8), total=len(tasks), desc="Segmentando"):
            if error:
                errors.append(error)
            else:
                writer.writerow([image_id, f"{area:.4f}"])
    return len(tasks), errors


def estimate_diameter_px(mask: np.ndarray) -> float:
    """Diámetro (px) del círculo mínimo que envuelve el componente más grande de la máscara."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    (_, _), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return 2.0 * radius


def diameter_to_size_class(diameter_norm: float, thresholds=(0.33, 0.66)) -> str:
    """Discretiza el diámetro normalizado en {Pequeño, Mediano, Grande}."""
    t1, t2 = thresholds
    if diameter_norm < t1:
        return config.SIZE_CLASSES[0]
    if diameter_norm < t2:
        return config.SIZE_CLASSES[1]
    return config.SIZE_CLASSES[2]


def main() -> None:
    n, errors = segment_index()
    for e in errors:
        print(f"✗ {e}")
    print(f"{n - len(errors)} imágenes segmentadas · {len(errors)} error(es).")


if __name__ == "__main__":
    main()
