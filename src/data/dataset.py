"""Índice unificado del dataset: une todas las fuentes con las 3 clases del proyecto.

Fase 3 de CRISP-DM. Fuentes:
- Kaggle `Good`                → Buena
- Kaggle `Bad` re-anotado      → Defectuosa / Podrida / Buena (solo la muestra ya anotada)
- Healthy-Defective-Fruits     → según config.HDF_LABELS (solo imágenes reales)
- Recolección propia           → anotaciones.csv (fuente "propias": reservada como prueba externa)

Escribe data/processed/index.csv con una fila por imagen: id (hash estable de la ruta), ruta
(relativa a data/), fuente, fruta, calidad, ancho, alto, formato, grupo y split (ver split.py).
Ancho/alto/formato permiten verificar que ninguna clase se distinga por propiedades del archivo en
lugar del contenido.

Se excluyen las imágenes en **escala de grises** (copias aumentadas: 1.188 granadas `Good` de Kaggle):
no permiten juzgar la calidad y solo aparecen en `Buena`, por lo que serían un atajo. Las excluidas
se registran en data/processed/excluidas.csv con su motivo.

Uso:  python -m src.main index
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from src import config

EXCLUDED = config.PROCESSED_DIR / "excluidas.csv"
GRAY_SATURATION = 8  # saturación HSV media (0-255) por debajo de la cual la imagen es gris
INDEX_COLUMNS = ["id", "ruta", "fuente", "fruta", "calidad", "ancho", "alto", "formato", "grupo", "split"]


def _images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in config.OWN_IMAGE_EXTS)


def kaggle_good_rows() -> list[dict]:
    good_dir = config.RAW_DIR / "Good Quality_Fruits"
    return [
        {"ruta": p, "fuente": "kaggle", "fruta": d.name.split("_")[0], "calidad": "Buena"}
        for d in sorted(good_dir.iterdir()) if d.is_dir() for p in _images(d)
    ]


def kaggle_bad_rows() -> list[dict]:
    from src.data import relabel

    labels = relabel.consolidated_labels(relabel.load_annotations())
    return [
        {"ruta": config.RAW_DIR / archivo, "fuente": "kaggle",
         "fruta": Path(archivo).parent.name.split("_")[0], "calidad": label}
        for archivo, label in sorted(labels.items()) if label != "Descartar"
    ]


def hdf_rows() -> list[dict]:
    rows = []
    for fruit, sub in config.HDF_REAL_DIRS.items():
        base = config.HDF_DIR / sub
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            rows += [{"ruta": p, "fuente": "hdf", "fruta": fruit, "calidad": config.HDF_LABELS[d.name]}
                     for p in _images(d)]
    return rows


def own_rows() -> list[dict]:
    with open(config.OWN_ANNOTATIONS, newline="", encoding="utf-8") as f:
        return [{"ruta": config.OWN_IMAGES_DIR / r["archivo"], "fuente": "propias",
                 "fruta": r["fruta"], "calidad": r["calidad"]} for r in csv.DictReader(f)]


def is_grayscale(path: Path) -> bool:
    """True si la imagen no tiene color (saturación media muy baja); decodifica reducida."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        img.draft("RGB", (64, 64))
        img = img.convert("RGB")
        img.thumbnail((64, 64))
        return np.asarray(img.convert("HSV"))[..., 1].mean() < GRAY_SATURATION


def build_index() -> tuple[list[dict], list[dict]]:
    """Une todas las fuentes y agrega metadatos, grupo y split. Devuelve (incluidas, excluidas)."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image  # import diferido

    from src.data.split import assign_split, group_key

    rows = kaggle_good_rows() + kaggle_bad_rows() + hdf_rows() + own_rows()
    with ThreadPoolExecutor() as pool:
        gray = list(pool.map(is_grayscale, [r["ruta"] for r in rows]))
    excluded = [{"ruta": Path(r["ruta"]).relative_to(config.DATA_DIR).as_posix(), "motivo": "escala de grises"}
                for r, g in zip(rows, gray) if g]
    rows = [r for r, g in zip(rows, gray) if not g]
    for row in rows:
        with Image.open(row["ruta"]) as img:
            row["ancho"], row["alto"] = img.size
            row["formato"] = img.format
        row["ruta"] = Path(row["ruta"]).relative_to(config.DATA_DIR).as_posix()
        row["id"] = hashlib.sha1(row["ruta"].encode()).hexdigest()[:16]
        row["grupo"] = group_key(row["ruta"], row["fuente"])
        row["split"] = assign_split(row["grupo"], row["fuente"])
    return rows, excluded


def write_index(rows: list[dict], path: Path | None = None) -> Path:
    path = path or config.DATASET_INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    from collections import Counter

    rows, excluded = build_index()
    path = write_index(rows)
    with open(EXCLUDED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ruta", "motivo"])
        writer.writeheader()
        writer.writerows(excluded)
    print(f"{len(rows)} imágenes → {path} · {len(excluded)} excluidas → {EXCLUDED.name}")
    counts = Counter((r["fuente"], r["calidad"]) for r in rows)
    for quality in config.QUALITY_CLASSES:
        detail = ", ".join(f"{src}={counts[(src, quality)]}" for src in ("kaggle", "hdf", "propias"))
        print(f"  {quality:11s} {sum(counts[(s, quality)] for s in ('kaggle', 'hdf', 'propias')):6d}  ({detail})")

    from src.data.split import SPLITS

    by_split = Counter((r["split"], r["calidad"]) for r in rows)
    print(f"\n  {'split':8s}" + "".join(f"{q:>12s}" for q in config.QUALITY_CLASSES))
    for split in SPLITS:
        print(f"  {split:8s}" + "".join(f"{by_split[(split, q)]:12d}" for q in config.QUALITY_CLASSES))
    print(f"  grupos: {len({r['grupo'] for r in rows})}")


if __name__ == "__main__":
    main()
