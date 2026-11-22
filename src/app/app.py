"""Interfaz de despliegue (Streamlit) — Fase 6 de CRISP-DM.

Permite cargar una o varias imágenes o capturarlas con la cámara y muestra la calidad predicha
(`Buena` / `Defectuosa` / `Podrida`), las probabilidades, lo que ve el modelo (fruta recortada sobre
blanco) y el tiempo de inferencia. Usa el mismo preprocesamiento que el entrenamiento
(`src/models/predict.py`) y por defecto el modelo de transfer learning (`config.FINAL_TRANSFER`).

Uso:  streamlit run src/app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# `streamlit run` solo agrega src/app/ al sys.path; se añade la raíz para poder importar `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402

MODEL_LABELS = {config.FINAL_TRANSFER: "EfficientNetB0 (transfer learning)",
                config.FINAL_MODEL: "CNN desde cero", config.FINAL_CLASSIC: "SVM (ML tradicional)"}
CLASS_STYLE = {"Buena": ("🟢", "green"), "Defectuosa": ("🟠", "orange"), "Podrida": ("🔴", "red")}
LOW_CONFIDENCE = 0.6        # por debajo se avisa que la predicción es dudosa
MIN_AREA = 0.02             # = segment.EMPTY_MASK: por debajo no se encontró la fruta


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_predictor(model_name: str):
    from src.models.predict import Predictor

    return Predictor(model_name)


@st.cache_data
def model_metrics(model_name: str) -> dict[str, dict]:
    """Métricas guardadas por src/evaluation/evaluate.py (val, test, fotos propias)."""
    out = {}
    for split in ("val", "test", "externa"):
        path = config.RESULTS_DIR / f"eval_{model_name}_{split}.json"
        if path.exists():
            out[split] = json.loads(path.read_text(encoding="utf-8"))
    return out


def sidebar() -> str:
    st.sidebar.header("Modelo")
    available = [m for m in MODEL_LABELS
                 if any((config.CHECKPOINTS_DIR / f"{m}{ext}").exists() for ext in (".keras", ".joblib"))]
    if not available:
        st.sidebar.error(f"No hay modelos en `{config.CHECKPOINTS_DIR}`. Ver `docs/instalacion.md`.")
        st.stop()
    model_name = st.sidebar.selectbox("Clasificador", available, format_func=MODEL_LABELS.get)

    metrics = model_metrics(model_name)
    names = {"val": "Validación", "test": "Prueba", "externa": "Fotos del equipo"}
    if metrics:
        st.sidebar.caption("Desempeño medido (F1-macro · recall de Podrida)")
        for split, r in metrics.items():
            st.sidebar.markdown(f"**{names[split]}:** {r['f1_macro']:.2f} · {r['por_clase']['Podrida']['recall']:.2f}".replace(".", ","))
    st.sidebar.warning(
        "**Limitación conocida:** con fotos de fruta y cámaras distintas a las de los datasets de "
        "entrenamiento el desempeño baja mucho (F1 ≈ 0,5 con las fotos del equipo). Úselo como "
        "apoyo, no como decisión final.")
    st.sidebar.caption("Una fruta por foto, fondo simple y buena luz. El tamaño (Pequeño/Mediano/Grande) "
                       "aún no se estima: requiere una referencia de escala en la foto.")
    return model_name


def show_prediction(name: str, source, pred) -> None:
    icon, color = CLASS_STYLE[pred.calidad]
    st.subheader(name)
    col_in, col_crop, col_out = st.columns([1, 1, 1.3])
    col_in.image(source, caption="Foto", width="stretch")
    col_crop.image(pred.recorte, caption="Lo que ve el modelo", width="stretch")
    with col_out:
        st.markdown(f"### {icon} :{color}[{pred.calidad}]")
        if pred.area_mascara < MIN_AREA:
            st.error("No se encontró una fruta clara en la foto: la predicción no es confiable.")
        if pred.probabilidades:
            for cls, p in pred.probabilidades.items():
                st.progress(p, text=f"{cls}: {p:.0%}")
            if pred.confianza < LOW_CONFIDENCE:
                st.warning("Predicción dudosa (ninguna clase supera el 60 %). Revise la fruta manualmente.")
        else:
            st.caption("El SVM no entrega probabilidades.")
        total = sum(pred.segundos.values())
        st.caption(f"Tiempo: {total:.2f} s (segmentación {pred.segundos['segmentacion']:.2f} s, "
                   f"modelo {pred.segundos['modelo']:.2f} s)")


def main() -> None:
    st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="wide")
    st.title("🍎 Clasificación de calidad de frutas")
    st.caption("Cargue o capture la foto de **una fruta** sobre fondo simple. El sistema quita el fondo, "
               "recorta la fruta y la clasifica como **Buena**, **Defectuosa** (daño superficial) o "
               "**Podrida** (moho, tejido blando o hundido, daño oscuro extenso).")

    predictor = load_predictor(sidebar())

    source = st.radio("Fuente de la imagen", ["Subir archivos", "Cámara"], horizontal=True)
    if source == "Subir archivos":
        files = st.file_uploader("Imágenes", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    else:
        shot = st.camera_input("Captura")
        files = [shot] if shot is not None else []

    if not files:
        st.info("Esperando una imagen.")
        return

    results = []
    for f in files:
        with st.spinner(f"Analizando {f.name}…"):
            pred = predictor.predict(f)
        show_prediction(f.name, f, pred)
        results.append({"imagen": f.name, "calidad": pred.calidad,
                        "confianza": None if pred.confianza is None else round(pred.confianza, 3)})
        st.divider()

    if len(results) > 1:
        import pandas as pd

        st.subheader("Resumen del lote")
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True)
        st.bar_chart(df["calidad"].value_counts().reindex(config.QUALITY_CLASSES, fill_value=0))


if __name__ == "__main__":
    main()
