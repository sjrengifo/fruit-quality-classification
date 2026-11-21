"""Paquete para entrenar la CNN en Google Colab (GPU) — Fase 4 de CRISP-DM.

Arma un zip con el código (`src/`) y lo mínimo para entrenar: el índice y las imágenes recortadas
de **train y val**. Las de `test` y las fotos propias (`externa`) no se incluyen: así es imposible
usarlas por error antes de la evaluación final (Fase 5). La segmentación ya está hecha, así que en
Colab no se instala rembg ni se repite nada pesado.

Uso:  python -m src.training.colab   → experiments/colab/fruit_colab.zip
Luego se sigue notebooks/colab_entrenar_cnn.ipynb.
"""
from __future__ import annotations

import csv
import subprocess
import zipfile
from datetime import datetime

from src import config

BUNDLE = config.EXPERIMENTS_DIR / "colab" / "fruit_colab.zip"
SPLITS = ("train", "val")


def git_version() -> str:
    """Commit actual (y si hay cambios sin commit), para saber con qué código se entrenó."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT_DIR,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], cwd=config.ROOT_DIR,
                               capture_output=True, text=True).stdout.strip()
        return commit + (" (con cambios sin commit en src/)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def build_bundle() -> tuple[int, int]:
    """Escribe BUNDLE. Devuelve (n.º de imágenes, tamaño en MB)."""
    from src.data.preprocess import cache_path

    with open(config.DATASET_INDEX, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in SPLITS]
    missing = [r["id"] for r in rows if not cache_path(r["id"], variant="crop").exists()]
    if missing:
        raise FileNotFoundError(f"Faltan {len(missing)} imágenes recortadas: correr `python -m src.main crop`.")

    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE, "w") as z:
        for path in sorted(config.ROOT_DIR.glob("src/**/*.py")):
            z.write(path, path.relative_to(config.ROOT_DIR).as_posix(), zipfile.ZIP_DEFLATED)
        z.write(config.DATASET_INDEX, config.DATASET_INDEX.relative_to(config.ROOT_DIR).as_posix(),
                zipfile.ZIP_DEFLATED)
        for r in rows:  # PNG ya está comprimido: se guarda sin volver a comprimir (más rápido)
            path = cache_path(r["id"], variant="crop")
            z.write(path, path.relative_to(config.ROOT_DIR).as_posix(), zipfile.ZIP_STORED)
        z.writestr("VERSION.txt", f"commit: {git_version()}\ncreado: {datetime.now():%Y-%m-%d %H:%M}\n"
                                  f"splits: {', '.join(SPLITS)} ({len(rows)} imágenes)\n")
    return len(rows), round(BUNDLE.stat().st_size / 1e6)


def main() -> None:
    n, mb = build_bundle()
    print(f"{BUNDLE} · {n} imágenes · {mb} MB · commit {git_version()}")
    print("Siguiente paso: subirlo a Google Drive (carpeta fruit-quality/) y abrir notebooks/colab_entrenar_cnn.ipynb")


if __name__ == "__main__":
    main()
