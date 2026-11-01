"""Gestión de la recolección propia de imágenes (data/external/propias/).

Fase 3 de CRISP-DM. Protocolo de captura y anotación en data/external/propias/README.md.

Uso:
    python -m src.data.own_images redimensionar   # reduce las imágenes nuevas a OWN_MAX_SIDE
    python -m src.data.own_images validar         # verifica anotaciones.csv contra las imágenes
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src import config


def list_images(images_dir: Path | None = None) -> list[Path]:
    """Imágenes de la recolección propia, ordenadas por nombre."""
    images_dir = images_dir or config.OWN_IMAGES_DIR
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in config.OWN_IMAGE_EXTS)


def resize_images(images_dir: Path | None = None, max_side: int | None = None) -> int:
    """Reduce in situ las imágenes cuyo lado mayor supere `max_side`.

    Aplica la orientación EXIF (fotos de celular) y guarda en JPEG calidad 90.
    Devuelve el número de imágenes modificadas.
    """
    from PIL import Image, ImageOps  # import diferido

    max_side = max_side or config.OWN_MAX_SIDE
    changed = 0
    for path in list_images(images_dir):
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if max(img.size) <= max_side:
                continue
            img.thumbnail((max_side, max_side))
            img.convert("RGB").save(path, quality=90)
        changed += 1
    return changed


def validate_annotations(
    annotations: Path | None = None, images_dir: Path | None = None
) -> list[str]:
    """Verifica que anotaciones.csv sea consistente con las imágenes.

    Devuelve la lista de errores encontrados (vacía si todo está bien).
    """
    annotations = annotations or config.OWN_ANNOTATIONS
    images_dir = images_dir or config.OWN_IMAGES_DIR
    errors: list[str] = []

    with open(annotations, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != config.OWN_ANNOTATION_COLUMNS:
            return [f"Columnas {reader.fieldnames} != {config.OWN_ANNOTATION_COLUMNS}"]
        rows = list(reader)

    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):  # línea 1 = encabezado
        name = row["archivo"]
        if name in seen:
            errors.append(f"L{i}: '{name}' anotado más de una vez")
        seen.add(name)
        if not (images_dir / name).is_file():
            errors.append(f"L{i}: no existe la imagen '{name}'")
        if row["fruta"] not in config.FRUITS:
            errors.append(f"L{i}: fruta '{row['fruta']}' no está en {config.FRUITS}")
        if row["calidad"] not in config.QUALITY_CLASSES:
            errors.append(f"L{i}: calidad '{row['calidad']}' no está en {config.QUALITY_CLASSES}")
        if row["tamano"] not in config.SIZE_CLASSES:
            errors.append(f"L{i}: tamaño '{row['tamano']}' no está en {config.SIZE_CLASSES}")
        if row["diametro_cm"]:
            try:
                float(row["diametro_cm"])
            except ValueError:
                errors.append(f"L{i}: diametro_cm '{row['diametro_cm']}' no es numérico")

    for path in list_images(images_dir):
        if path.name not in seen:
            errors.append(f"Imagen sin anotar: '{path.name}'")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Recolección propia de imágenes")
    parser.add_argument("accion", choices=["redimensionar", "validar"])
    args = parser.parse_args()

    if args.accion == "redimensionar":
        n = resize_images()
        print(f"{n} imagen(es) redimensionada(s) a lado mayor <= {config.OWN_MAX_SIDE} px.")
    else:
        errors = validate_annotations()
        for e in errors:
            print(f"✗ {e}")
        n = len(list_images())
        print(f"{n} imagen(es), {len(errors)} error(es).")
        raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
