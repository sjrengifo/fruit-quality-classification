"""Punto de entrada / orquestación del pipeline.

Uso:  python -m src.main <etapa>
Etapas: download | index | train | evaluate
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline del proyecto de calidad de frutas")
    parser.add_argument(
        "stage",
        choices=["download", "index", "train", "evaluate"],
        help="Etapa del pipeline a ejecutar",
    )
    args = parser.parse_args()

    if args.stage == "download":
        from src.data.download import download_dataset, download_healthy_defective

        download_dataset()
        download_healthy_defective()
    elif args.stage == "index":
        from src.data.dataset import main as index_main

        index_main()
    elif args.stage == "train":
        from src.training.train import main as train_main

        train_main()
    elif args.stage == "evaluate":
        raise NotImplementedError("Implementar en Fase 5.")


if __name__ == "__main__":
    main()
