"""Orquestación del entrenamiento de modelos (ML y CNN).

Fase 4 de CRISP-DM. Guarda artefactos en experiments/.
Uso:  python -m src.training.train --model cnn
"""
from __future__ import annotations

import argparse

from src import config


def train_cnn():
    """Entrena la CNN y guarda el checkpoint del mejor modelo."""
    raise NotImplementedError("Implementar en Fase 4: cargar datos, fit, callbacks.")


def train_classic():
    """Entrena y ajusta los modelos de ML tradicional (GridSearchCV)."""
    raise NotImplementedError("Implementar en Fase 4: features, tune(), persistencia.")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos")
    parser.add_argument("--model", choices=["cnn", "classic"], default="cnn")
    args = parser.parse_args()

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.model == "cnn":
        train_cnn()
    else:
        train_classic()


if __name__ == "__main__":
    main()
