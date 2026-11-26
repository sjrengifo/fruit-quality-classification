"""Figuras del informe IEEE → docs/informe/latex/figuras/ (PDF vectorial).

- `preparacion.pdf`: imagen preprocesada → fruta segmentada y recortada (lo que ven los modelos), y la
  medición de tamaño con la moneda sobre una foto propia.
- Las figuras SVG de `assets/` que usa el informe se convierten a PDF con `rsvg-convert` (librsvg).

Uso:  python -m src.evaluation.figuras_informe
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

from src import config

OUT = config.ROOT_DIR / "docs" / "informe" / "latex" / "figuras"
SVGS = {
    "flujo_crisp_dm": config.ROOT_DIR / "assets" / "informe" / "flujo_crisp_dm.svg",
    "matrices_confusion": config.ROOT_DIR / "assets" / "evaluacion" / "matrices_confusion_final.svg",
    "arquitectura": config.ROOT_DIR / "assets" / "informe" / "arquitectura.svg",
}
# (id del índice, rótulo): fruta en la mano o en el árbol, y esquinas negras de rotación (atajos de fondo)
EXAMPLES = [("a6b899dd149c4eeb", "HDF · Defectuosa"), ("6e396999887e9b42", "HDF · Podrida"),
            ("6e7127d2df78448f", "Kaggle · Podrida")]
SIZE_PHOTO = "manzana1.jpg"


def preparation_figure() -> None:
    """Figura de preparación: original frente a segmentada de tres ejemplos y la medición de tamaño con moneda."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.app.app import measurement_overlay
    from src.data.preprocess import cache_path
    from src.features.size import measure

    fig, axes = plt.subplots(1, 7, figsize=(7.16, 1.35),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1.25], "wspace": 0.06})
    for k, (image_id, label) in enumerate(EXAMPLES):
        for j, variant in enumerate(("img", "crop")):
            ax = axes[2 * k + j]
            ax.imshow(Image.open(cache_path(image_id, variant=variant)))
            ax.set_title(f"{label}\n{'original' if variant == 'img' else 'segmentada'}", fontsize=6)
    path = config.SIZE_IMAGES_DIR / SIZE_PHOTO
    est = measure(path)
    overlay = measurement_overlay(path, est)
    rows = np.nonzero(overlay.std(axis=(1, 2)) > 2)[0]          # quita el relleno replicado arriba y abajo
    axes[6].imshow(overlay[rows.min():rows.max() + 1])
    axes[6].set_title(f"Foto propia + moneda\n{est.diametro_cm:.1f} cm (regla: 7,7)".replace(".", ","), fontsize=6)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    fig.savefig(OUT / "preparacion.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)


def convert_svgs() -> None:
    """Convierte a PDF vectorial (rsvg-convert) las figuras SVG de `assets/` que usa el informe."""
    if shutil.which("rsvg-convert") is None:
        raise SystemExit("Falta rsvg-convert (librsvg): brew install librsvg")
    for name, svg in SVGS.items():
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(OUT / f"{name}.pdf"), str(svg)], check=True)


def main() -> None:
    """Regenera todas las figuras del informe en `docs/informe/latex/figuras/`."""
    OUT.mkdir(parents=True, exist_ok=True)
    convert_svgs()
    preparation_figure()
    print(f"Figuras en {OUT}")


if __name__ == "__main__":
    main()
