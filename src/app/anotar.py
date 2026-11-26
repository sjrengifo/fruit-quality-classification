"""Herramienta de re-anotación de `Bad` (Kaggle) en Defectuosa / Podrida — Fase 3 de CRISP-DM.

Uso:  streamlit run src/app/anotar.py
Criterios: docs/criterios_anotacion.md. Lógica en src/data/relabel.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run` solo agrega src/app/ al sys.path; se añade la raíz para poder importar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.data import relabel  # noqa: E402

HELP = {
    "Defectuosa": "Golpe, mancha, sarna o raspón superficial; la pulpa se ve firme. Aún consumible.",
    "Podrida": "Tejido blando, hundido u oscuro en profundidad, moho, líquido o piel rota con pudrición.",
    "Buena": "No se ve ningún defecto (error de etiqueta en Kaggle).",
    "Descartar": "Varias frutas, imagen borrosa, fruta irreconocible o no se puede decidir.",
}


@st.cache_data
def sample() -> list[dict]:
    """Muestra fija de `Bad` a re-anotar (ver `relabel.build_sample`), cacheada por Streamlit."""
    return relabel.build_sample()


def main() -> None:
    """Interfaz de anotación: muestra cada imagen de la muestra y guarda la etiqueta en el CSV del anotador."""
    st.set_page_config(page_title="Re-anotación de Bad", page_icon="🏷️", layout="wide")
    st.title("🏷️ Re-anotación de `Bad`")

    with st.sidebar:
        author = st.text_input("Tu nombre", key="author")
        fruits = sorted({i["fruta"] for i in sample()})
        fruit = st.selectbox("Fruta", fruits)
        st.markdown("**Criterios** (ver `docs/criterios_anotacion.md`)")
        for label, text in HELP.items():
            st.markdown(f"- **{label}:** {text}")
        st.caption("Haz `git pull` antes de empezar y commit + push de tu CSV al terminar.")

    if not relabel.author_slug(author or ""):
        st.info("Escribe tu nombre en la barra lateral para empezar.")
        return

    annotations = relabel.load_annotations()
    todo = relabel.pending(sample(), annotations, author, fruit)
    total = sum(1 for i in sample() if i["fruta"] == fruit)
    mine = len(annotations.get(relabel.author_slug(author), {}))
    st.progress(1 - len(todo) / total, text=f"{fruit}: faltan {len(todo)} de {total} · tus anotaciones: {mine}")

    if st.button("↩️ Deshacer última"):
        undone = relabel.undo_last(author)
        st.toast(f"Deshecho: {undone}" if undone else "Nada que deshacer")
        st.rerun()

    if not todo:
        st.success(f"¡No quedan imágenes de {fruit}! Elige otra fruta.")
        return

    item = todo[0]
    left, right = st.columns([3, 2])
    left.image(str(config.RAW_DIR / item["archivo"]), width="stretch")
    with right:
        st.caption(item["archivo"] + (" · 🤝 imagen de acuerdo (la anotan todos)" if item["acuerdo"] else ""))
        for label in config.RELABEL_OPTIONS:
            if st.button(label, key=label, width="stretch", help=HELP[label],
                         type="primary" if label in ("Defectuosa", "Podrida") else "secondary"):
                relabel.save_label(author, item, label)
                st.rerun()


if __name__ == "__main__":
    main()
