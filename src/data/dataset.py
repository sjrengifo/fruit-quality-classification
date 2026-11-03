"""Índice unificado del dataset: une todas las fuentes con las 3 clases del proyecto.

Fase 3 de CRISP-DM. Fuentes:
- Kaggle `Good`                → Buena
- Kaggle `Bad` re-anotado      → Defectuosa / Podrida / Buena (solo la muestra ya anotada)
- Healthy-Defective-Fruits     → según config.HDF_LABELS (solo imágenes reales)
- Recolección propia           → anotaciones.csv (fuente "propias": reservada como prueba externa)

Escribe data/processed/index.csv con una fila por imagen: ruta (relativa a data/), fuente, fruta,
calidad, ancho, alto, formato. Ancho/alto/formato permiten verificar que ninguna clase se distinga
por propiedades del archivo en lugar del contenido.

Uso:  python -m src.main index
"""
from __future__ import annotations

import csv
from pathlib import Path

from src import config

INDEX_COLUMNS = ["ruta", "fuente", "fruta", "calidad", "ancho", "alto", "formato"]


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


def build_index() -> list[dict]:
    """Une todas las fuentes y agrega ancho/alto/formato (lee solo el encabezado de cada imagen)."""
    from PIL import Image  # import diferido

    rows = kaggle_good_rows() + kaggle_bad_rows() + hdf_rows() + own_rows()
    for row in rows:
        with Image.open(row["ruta"]) as img:
            row["ancho"], row["alto"] = img.size
            row["formato"] = img.format
        row["ruta"] = Path(row["ruta"]).relative_to(config.DATA_DIR).as_posix()
    return rows


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

    rows = build_index()
    path = write_index(rows)
    print(f"{len(rows)} imágenes → {path}")
    counts = Counter((r["fuente"], r["calidad"]) for r in rows)
    for quality in config.QUALITY_CLASSES:
        detail = ", ".join(f"{src}={counts[(src, quality)]}" for src in ("kaggle", "hdf", "propias"))
        print(f"  {quality:11s} {sum(counts[(s, quality)] for s in ('kaggle', 'hdf', 'propias')):6d}  ({detail})")


if __name__ == "__main__":
    main()
