"""Interfaz de despliegue (Streamlit) — Fase 6 de CRISP-DM.

Permite cargar una imagen o capturarla con la cámara y muestra la predicción de
calidad y tamaño de la fruta.

Uso:  streamlit run src/app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run` solo agrega src/app/ al sys.path; se añade la raíz para poder importar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Clasificación de Calidad de Frutas", page_icon="🍎")
    st.title("🍎 Clasificación de Calidad de Frutas")
    st.caption(
        "Cargue o capture la imagen de una fruta individual sobre fondo simple."
    )

    source = st.radio("Fuente de la imagen", ["Subir archivo", "Cámara"], horizontal=True)
    if source == "Subir archivo":
        file = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("Captura")

    if file is not None:
        st.image(file, caption="Entrada", use_container_width=True)
        # TODO(Fase 6): cargar el mejor modelo desde experiments/checkpoints/,
        # preprocesar la imagen y ejecutar la predicción.
        st.info(
            "Predicción pendiente de implementar. "
            f"Clases de calidad: {config.QUALITY_CLASSES} · "
            f"Tamaños: {config.SIZE_CLASSES}"
        )


if __name__ == "__main__":
    main()
