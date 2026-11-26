"""Segmentación de la fruta y reemplazo del fondo — Fase 3 de CRISP-DM.

Motivo: el fondo predice la clase (F1 solo-fondo 0,67; 0,91 en Healthy-Defective-Fruits, ver
src/evaluation/shortcut.py). Se separa la fruta con U²-Net (Qin et al., 2020) vía `rembg`, un modelo
preentrenado **solo para quitar fondos**; el clasificador de calidad se entrena desde cero.
El fondo se reemplaza por blanco, como en la recolección propia y el enunciado (fondo simple).

Pasos por imagen: prepare_image a SEG_INPUT px → máscara U²-Net (umbral 0,5) → se eliminan
componentes pequeños (< MIN_COMPONENT del área) → fondo blanco → reducción a IMAGE_SIZE.
Cachés: data/interim/seg<N>/<id>.png (imagen) y data/interim/mask<N>/<id>.png (máscara), más
data/interim/seg_stats.csv con el área de la máscara para control de calidad.

Después se **normaliza el encuadre** (`crop_fruit`): se recorta la caja de la máscara y se reescala,
para que el zoom no prediga la clase (en HDF las `Buena` llenan el cuadro y las `Podrida` no).
Cachés: data/interim/crop<N>/ y cropmask<N>/. La máscara sin recortar se conserva para el tamaño.

Uso:  python -m src.main segment   (segmenta)
      python -m src.main crop      (recorta; rápido, no usa la red)
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
CROP_MARGIN = 0.05           # margen alrededor de la fruta al recortar (fracción del lado)

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


def crop_fruit(image: np.ndarray, mask: np.ndarray, size: tuple[int, int] | None = None,
               margin: float = CROP_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """Recorta la fruta (caja cuadrada de la máscara + margen) sobre fondo blanco y la lleva a `size`.

    `mask` puede venir a menor resolución que `image`. Con máscara llena (segmentación fallida) se
    devuelve la imagen completa.
    """
    from PIL import Image

    size = size or config.IMAGE_SIZE
    h, w = image.shape[:2]
    if mask.shape != (h, w):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        mask = np.ones((h, w), dtype=bool)
        ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    ox, oy = (side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2
    canvas = np.full((side, side, 3), BACKGROUND, dtype=np.uint8)
    canvas_mask = np.zeros((side, side), dtype=bool)
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = remove_background(image, mask)[y0:y1, x0:x1]
    canvas_mask[oy:oy + y1 - y0, ox:ox + x1 - x0] = mask[y0:y1, x0:x1]
    out = Image.fromarray(canvas).resize((size[1], size[0]), Image.Resampling.LANCZOS)
    out_mask = Image.fromarray(canvas_mask.astype(np.uint8) * 255).resize((size[1], size[0]), Image.Resampling.NEAREST)
    return np.asarray(out), np.asarray(out_mask) > 127


def _crop_process(task: tuple[str, str, tuple[int, int]]) -> str | None:
    from PIL import Image

    from src.data.preprocess import cache_path, prepare_image

    ruta, image_id, size = task
    try:
        mask = np.asarray(Image.open(cache_path(image_id, size, "mask"))) > 127
        image, crop_mask = crop_fruit(prepare_image(config.DATA_DIR / ruta, SEG_INPUT), mask, size)
        Image.fromarray(image).save(cache_path(image_id, size, "crop"))
        Image.fromarray(crop_mask.astype(np.uint8) * 255).save(cache_path(image_id, size, "cropmask"))
    except Exception as exc:
        return f"{ruta}: {exc}"
    return None


def segment_and_crop(source, size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Una imagen nueva → (fruta recortada sobre blanco, máscara recortada, área relativa).

    Mismo camino que `segment` + `crop` sobre el índice (incluida la máscara reducida a `size` antes
    de recortar), para que la app y la evaluación vean exactamente lo que vio el modelo al entrenar.
    """
    from src.data.preprocess import prepare_image

    big = prepare_image(source, SEG_INPUT)
    return crop_with_mask(big, fruit_mask(big), size)


def crop_with_mask(big: np.ndarray, mask: np.ndarray, size: tuple[int, int] | None = None):
    """Segunda mitad de `segment_and_crop` con una máscara ya calculada (a SEG_INPUT), p. ej. la de
    solo la fruta cuando la foto trae una moneda para el tamaño."""
    from PIL import Image

    size = size or config.IMAGE_SIZE
    area = float(mask.mean())
    if area < EMPTY_MASK:
        mask = np.ones_like(mask)
    small_mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                            .resize((size[1], size[0]), Image.Resampling.NEAREST)) > 127
    image, crop_mask = crop_fruit(big, small_mask, size)
    return image, crop_mask, area


def crop_index(size: tuple[int, int] | None = None, workers: int | None = None) -> tuple[int, list[str]]:
    """Recorta todas las imágenes ya segmentadas del índice (incremental)."""
    from tqdm import tqdm

    from src.data.preprocess import cache_dir, cache_path

    size = size or config.IMAGE_SIZE
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for variant in ("crop", "cropmask"):
        cache_dir(size, variant).mkdir(parents=True, exist_ok=True)
    tasks = [(r["ruta"], r["id"], size) for r in rows
             if cache_path(r["id"], size, "mask").exists() and not cache_path(r["id"], size, "cropmask").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = [e for e in tqdm(pool.map(_crop_process, tasks, chunksize=64), total=len(tasks), desc="Recortando") if e]
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
    """Segmenta con U²-Net las imágenes del índice que faltan en la caché e informa los errores."""
    n, errors = segment_index()
    for e in errors:
        print(f"✗ {e}")
    print(f"{n - len(errors)} imágenes segmentadas · {len(errors)} error(es).")


def crop_main() -> None:
    """Recorta cada fruta segmentada a la caja de su máscara (caché `crop128`) e informa los errores."""
    n, errors = crop_index()
    for e in errors:
        print(f"✗ {e}")
    print(f"{n - len(errors)} imágenes recortadas · {len(errors)} error(es).")


if __name__ == "__main__":
    main()
