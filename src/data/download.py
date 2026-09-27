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


if __name__ == "__main__":
    download_dataset()
