"""Descarga del dataset base desde Kaggle hacia data/raw/.

Requiere tener configurado el token de Kaggle (ver docs/instalacion.md).
Uso:  python -m src.data.download
"""
from __future__ import annotations

from src import config


def download_dataset() -> None:
    """Descarga y descomprime el dataset Fruit Quality Classification en data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Import diferido: la librería kaggle valida credenciales al importarse.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Descargando '{config.KAGGLE_DATASET}' en {config.RAW_DIR} ...")
    api.dataset_download_files(
        config.KAGGLE_DATASET, path=str(config.RAW_DIR), unzip=True
    )
    print("Descarga completada.")


def download_healthy_defective() -> None:
    """Descarga solo las imágenes reales de Healthy-Defective-Fruits en data/external/.

    Usa un clon parcial (sparse checkout) para no bajar las imágenes sintéticas (~50 % del repo).
    """
    import subprocess

    if config.HDF_DIR.exists():
        print(f"Ya existe {config.HDF_DIR}; se omite.")
        return
    print(f"Descargando Healthy-Defective-Fruits (solo reales) en {config.HDF_DIR} ...")
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    config.HDF_REPO, str(config.HDF_DIR)], check=True)
    subprocess.run(["git", "-C", str(config.HDF_DIR), "sparse-checkout", "set",
                    *config.HDF_REAL_DIRS.values()], check=True)
    print("Descarga completada.")


if __name__ == "__main__":
    download_dataset()
    download_healthy_defective()
