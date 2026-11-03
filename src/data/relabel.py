"""Re-anotación de la carpeta `Bad` de Kaggle en Defectuosa / Podrida.

Fase 3 de CRISP-DM. `Bad` mezcla frutas con defectos leves y frutas podridas; se re-anota a mano
una muestra fija por fruta (criterios en docs/criterios_anotacion.md). La interfaz está en
src/app/anotar.py; aquí vive la lógica, sin dependencias de Streamlit.

Cada anotador escribe su propio CSV (data/anotaciones/bad_<autor>.csv) para evitar conflictos de
git. Las primeras AGREEMENT_PER_FRUIT imágenes de cada fruta las anotan todos y sirven para medir
el acuerdo entre anotadores (kappa de Cohen).

Uso:  python -m src.data.relabel    # resumen de avance y acuerdo
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter
from datetime import date
from itertools import combinations
from pathlib import Path

from src import config

COLUMNS = ["archivo", "fruta", "etiqueta", "autor", "fecha"]
BAD_DIR = config.RAW_DIR / "Bad Quality_Fruits"


def build_sample(bad_dir: Path | None = None, per_fruit: int | None = None) -> list[dict]:
    """Muestra reproducible de `Bad`: lista de {archivo, fruta, acuerdo}.

    `archivo` es relativo a data/raw/ (formato POSIX) para que sea estable entre equipos.
    """
    bad_dir = bad_dir or BAD_DIR
    per_fruit = per_fruit or config.RELABEL_PER_FRUIT
    rng = random.Random(config.RANDOM_SEED)
    sample = []
    for fruit_dir in sorted(p for p in bad_dir.iterdir() if p.is_dir()):
        fruit = fruit_dir.name.split("_")[0]
        files = sorted(p for p in fruit_dir.iterdir() if p.suffix.lower() in config.OWN_IMAGE_EXTS)
        for i, path in enumerate(rng.sample(files, min(per_fruit, len(files)))):
            sample.append({
                "archivo": path.relative_to(bad_dir.parent).as_posix(),
                "fruta": fruit,
                "acuerdo": i < config.AGREEMENT_PER_FRUIT,
            })
    return sample


def author_slug(name: str) -> str:
    """'Samuel José' -> 'samueljose' (nombre de archivo seguro)."""
    import unicodedata

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_name.lower())


def annotations_path(author: str, annotations_dir: Path | None = None) -> Path:
    return (annotations_dir or config.ANNOTATIONS_DIR) / f"bad_{author_slug(author)}.csv"


def load_annotations(annotations_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """{autor: {archivo: etiqueta}} con todos los CSV; la última fila por archivo prevalece."""
    annotations_dir = annotations_dir or config.ANNOTATIONS_DIR
    result: dict[str, dict[str, str]] = {}
    for path in sorted(annotations_dir.glob("bad_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                result.setdefault(row["autor"], {})[row["archivo"]] = row["etiqueta"]
    return result


def save_label(author: str, item: dict, label: str, annotations_dir: Path | None = None) -> None:
    """Agrega una anotación al CSV del autor."""
    if label not in config.RELABEL_OPTIONS:
        raise ValueError(f"Etiqueta inválida: {label}")
    path = annotations_path(author, annotations_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({
            "archivo": item["archivo"], "fruta": item["fruta"], "etiqueta": label,
            "autor": author_slug(author), "fecha": date.today().isoformat(),
        })


def undo_last(author: str, annotations_dir: Path | None = None) -> str | None:
    """Elimina la última anotación del autor; devuelve el archivo deshecho."""
    path = annotations_path(author, annotations_dir)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) <= 1:
        return None
    path.write_text("".join(lines[:-1]), encoding="utf-8")
    return lines[-1].split(",")[0]


def pending(sample: list[dict], annotations: dict[str, dict[str, str]], author: str,
            fruit: str | None = None) -> list[dict]:
    """Imágenes que le faltan al autor: las de acuerdo que no ha hecho + las que nadie ha hecho."""
    author = author_slug(author)
    mine = annotations.get(author, {})
    done_by_anyone = {a for labels in annotations.values() for a in labels}
    return [
        item for item in sample
        if (fruit is None or item["fruta"] == fruit)
        and item["archivo"] not in mine
        and (item["acuerdo"] or item["archivo"] not in done_by_anyone)
    ]


def agreement(sample: list[dict], annotations: dict[str, dict[str, str]]) -> dict[tuple[str, str], tuple[float, int]]:
    """Kappa de Cohen por par de anotadores sobre las imágenes de acuerdo: {(a, b): (kappa, n)}."""
    from sklearn.metrics import cohen_kappa_score  # import diferido

    shared = [item["archivo"] for item in sample if item["acuerdo"]]
    result = {}
    for a, b in combinations(sorted(annotations), 2):
        common = [f for f in shared if f in annotations[a] and f in annotations[b]]
        if len(common) >= 2:
            ya = [annotations[a][f] for f in common]
            yb = [annotations[b][f] for f in common]
            result[(a, b)] = (cohen_kappa_score(ya, yb), len(common))
    return result


def consolidated_labels(annotations: dict[str, dict[str, str]]) -> dict[str, str]:
    """Etiqueta final por imagen: voto mayoritario; empate -> 'Descartar'."""
    votes: dict[str, Counter] = {}
    for labels in annotations.values():
        for archivo, label in labels.items():
            votes.setdefault(archivo, Counter())[label] += 1
    final = {}
    for archivo, counter in votes.items():
        (top, n), *rest = counter.most_common()
        final[archivo] = "Descartar" if rest and rest[0][1] == n else top
    return final


def main() -> None:
    sample = build_sample()
    annotations = load_annotations()
    final = consolidated_labels(annotations)
    print(f"Muestra: {len(sample)} imágenes · anotadas: {len(final)}")
    for fruit in sorted({i['fruta'] for i in sample}):
        files = {i["archivo"] for i in sample if i["fruta"] == fruit}
        counts = Counter(final[f] for f in files if f in final)
        print(f"  {fruit:12s} {sum(counts.values()):4d}/{len(files)}  {dict(counts)}")
    for (a, b), (kappa, n) in agreement(sample, annotations).items():
        print(f"Acuerdo {a} vs {b}: kappa = {kappa:.2f} (n = {n})")


if __name__ == "__main__":
    main()
