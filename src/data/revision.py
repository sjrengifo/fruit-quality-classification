"""Revisión de etiquetas (tercera iteración de CRISP-DM): una nueva mirada a los datos.

Las fotos propias mostraron que el modelo confunde Podrida con Defectuosa (20 de 30 podridas) y la
revisión encontró imágenes con varias frutas. Este módulo NO modifica `index.csv` ni las
anotaciones anteriores: escribe un índice aparte (`config.DATASET_INDEX_V3`) para entrenar modelos
nuevos desde cero y comparar.

1. `objetos`: cuenta los objetos grandes de cada máscara de U²-Net; con 2 o más la imagen tiene
   varias frutas y se excluye (son casi todas de Kaggle `Buena`: el modelo aprendía "varias frutas
   → Buena", algo que nunca pasa en las fotos propias).
2. `hojas`: hojas de contacto numeradas, a ciegas (sin la etiqueta original), una celda por grupo
   de Defectuosa/Podrida y de `Buena` en HDF/FruQ. Los grupos son la misma fruta, así que se revisa
   un representativo (o un mosaico 2×2 si el grupo es grande).
3. Las etiquetas revisadas se escriben, con los criterios de docs/criterios_anotacion.md, en
   `data/anotaciones/revision_v3.txt`: una línea por hoja, `<hoja> <letras>` con B=Buena,
   D=Defectuosa, P=Podrida, X=Descartar, una letra por celda.
4. `indice`: aplica la revisión por grupo y escribe el índice nuevo. Los splits no cambian (son
   por grupo y se calculan con hash).

Uso:  python -m src.data.revision objetos|hojas|indice|resumen
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict

from src import config

REVISION_DIR = config.INTERIM_DIR / "revision"
OBJECTS_CSV = REVISION_DIR / "objetos.csv"
SHEETS_CSV = REVISION_DIR / "hojas.csv"          # hoja, celda, grupo
LABELS_TXT = config.ROOT_DIR / "data" / "anotaciones" / "revision_v3.txt"
LETTERS = {"B": "Buena", "D": "Defectuosa", "P": "Podrida", "X": "Descartar"}
MIN_OBJECT = 0.15        # un objeto cuenta como fruta si mide ≥ 15 % del más grande
CELLS, COLS, CELL_PX = 30, 6, 192


def _index() -> list[dict]:
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["fuente"] != "propias"]


def count_objects(mask) -> int:
    """Objetos grandes (componentes conexas) en una máscara binaria."""
    import numpy as np
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    if n == 0:
        return 0
    areas = np.bincount(lab.ravel())[1:]
    return int((areas >= MIN_OBJECT * areas.max()).sum())


def write_objects() -> Counter:
    import numpy as np
    from PIL import Image

    REVISION_DIR.mkdir(parents=True, exist_ok=True)
    rows = _index()
    counts = Counter()
    with open(OBJECTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "objetos"])
        for r in rows:
            n = count_objects(np.asarray(Image.open(config.INTERIM_DIR / "mask128" / f"{r['id']}.png")) > 127)
            w.writerow([r["id"], n])
            counts[(r["fuente"], r["calidad"], min(n, 2))] += 1
    return counts


def _objects() -> dict[str, int]:
    with open(OBJECTS_CSV, newline="", encoding="utf-8") as f:
        return {r["id"]: int(r["objetos"]) for r in csv.DictReader(f)}


def groups_to_review() -> list[tuple[str, str, list[dict]]]:
    """(fruta, grupo, filas de una sola fruta) a revisar, en orden reproducible y sin la etiqueta."""
    objects = _objects()
    groups = defaultdict(list)
    for r in _index():
        if objects[r["id"]] != 1:
            continue
        if r["calidad"] != "Buena" or r["fuente"] in ("hdf", "fruq"):
            groups[r["grupo"]].append(r)
    out = [(rows[0]["fruta"], g, rows) for g, rows in groups.items()]
    random.Random(config.RANDOM_SEED).shuffle(out)       # mezcla fuentes y clases dentro de cada fruta
    return sorted(out, key=lambda t: t[0])


def _cell(rows: list[dict]):
    from PIL import Image

    def load(r, px):
        im = Image.open(config.ROOT_DIR / "data" / r["ruta"]).convert("RGB")
        im.thumbnail((px, px))
        return im

    rows = sorted(rows, key=lambda r: r["ruta"])
    if len(rows) <= 5:
        return load(rows[len(rows) // 2], CELL_PX)
    out = Image.new("RGB", (CELL_PX, CELL_PX), "white")
    half = CELL_PX // 2
    for k in range(4):
        out.paste(load(rows[k * (len(rows) - 1) // 3], half), ((k % 2) * half, (k // 2) * half))
    return out


def write_sheets() -> int:
    from PIL import Image, ImageDraw, ImageFont

    out_dir = REVISION_DIR / "hojas"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=22)
    groups = groups_to_review()
    by_fruit = defaultdict(list)
    for fruit, g, rows in groups:
        by_fruit[fruit].append((g, rows))
    n_sheet = 0
    with open(SHEETS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hoja", "celda", "grupo"])
        for fruit, items in by_fruit.items():
            for start in range(0, len(items), CELLS):
                chunk = items[start:start + CELLS]
                name = f"h{n_sheet:03d}"
                n_rows = (len(chunk) + COLS - 1) // COLS
                sheet = Image.new("RGB", (COLS * (CELL_PX + 4), n_rows * (CELL_PX + 4) + 30), (60, 60, 60))
                d = ImageDraw.Draw(sheet)
                d.text((4, 2), f"{name} · {fruit}", fill="white", font=font)
                for i, (g, rows) in enumerate(chunk):
                    x, y = (i % COLS) * (CELL_PX + 4), (i // COLS) * (CELL_PX + 4) + 30
                    im = _cell(rows)
                    sheet.paste(im, (x + (CELL_PX - im.width) // 2, y + (CELL_PX - im.height) // 2))
                    d.rectangle((x, y, x + 34, y + 24), fill="black")
                    d.text((x + 3, y + 1), str(i), fill="yellow", font=font)
                    w.writerow([name, i, g])
                sheet.save(out_dir / f"{name}.jpg", quality=88)
                n_sheet += 1
    return n_sheet


def read_labels() -> dict[str, str]:
    """grupo → calidad revisada (incluye `Descartar`)."""
    with open(SHEETS_CSV, newline="", encoding="utf-8") as f:
        cells = {(r["hoja"], int(r["celda"])): r["grupo"] for r in csv.DictReader(f)}
    labels = {}
    for line in LABELS_TXT.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sheet, letters = line.split()[:2]
        for i, c in enumerate(letters):
            labels[cells[(sheet, i)]] = LETTERS[c]
    return labels


def build_index_v3() -> Counter:
    """Índice nuevo: sin imágenes de varias frutas y con las etiquetas revisadas por grupo."""
    objects = _objects()
    labels = read_labels()
    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    out, changes = [], Counter()
    for r in rows:
        if r["fuente"] != "propias":
            if objects[r["id"]] != 1:
                changes["varias frutas / sin fruta"] += 1
                continue
            new = labels.get(r["grupo"], r["calidad"])
            changes[(r["calidad"], new)] += 1
            if new == "Descartar":
                continue
            r = {**r, "calidad": new}
        out.append(r)
    with open(config.DATASET_INDEX_V3, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    return changes


def summary() -> None:
    """Acuerdo entre la etiqueta original y la revisada, por fuente (grupos)."""
    from sklearn.metrics import cohen_kappa_score

    labels = read_labels()
    first = {}
    for r in _index():
        first.setdefault(r["grupo"], r)
    for source in ("hdf", "kaggle", "fruq"):
        pairs = [(first[g]["calidad"], c) for g, c in labels.items() if first[g]["fuente"] == source]
        kept = [(a, b) for a, b in pairs if b != "Descartar"]
        print(f"\n{source}: {len(pairs)} grupos revisados · {len(pairs) - len(kept)} descartados · "
              f"kappa original↔revisada {cohen_kappa_score(*zip(*kept)):.2f}")
        for (a, b), n in sorted(Counter(pairs).items()):
            print(f"  {a:<11} → {b:<11} {n}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "resumen"
    if cmd == "objetos":
        for k, n in sorted(write_objects().items()):
            print(k, n)
    elif cmd == "hojas":
        print(f"{write_sheets()} hojas en {REVISION_DIR / 'hojas'}")
    elif cmd == "indice":
        for k, n in sorted(build_index_v3().items(), key=str):
            print(k, n)
        print(f"→ {config.DATASET_INDEX_V3}")
    else:
        summary()


if __name__ == "__main__":
    main()
