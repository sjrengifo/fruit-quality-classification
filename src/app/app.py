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
FRUIT_NAMES = {"Apple": "Manzana", "Orange": "Naranja", "Lime": "Limón", "Guava": "Guayaba",
               "Pomegranate": "Granada", "Mango": "Mango", "Banana": "Banano", "Otra": "Otra"}
COIN_NAMES = {"100_nueva": "$100 nueva", "200_nueva": "$200 nueva", "500_nueva": "$500 nueva",
              "1000_nueva": "$1.000 nueva", "500_vieja": "$500 antigua", "euro_1c": "1 céntimo de euro",
              "euro_10c": "10 céntimos de euro", "euro_20c": "20 céntimos de euro",
              "euro_50c": "50 céntimos de euro", "dime": "Dime (EE. UU.)", "nickel": "Nickel (EE. UU.)",
              "quarter": "Quarter (EE. UU.)"}


@st.cache_resource(show_spinner="Descargando los modelos (solo la primera vez)…")
def ensure_models() -> list[str]:
    """En la app desplegada los modelos no vienen en el repositorio: se bajan de la release de GitHub
    (src/app/modelos.py). En local, si ya están en experiments/, no hace nada."""
    from src.app.modelos import download_missing

    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:  # noqa: BLE001 — sin secrets.toml (uso local)
        token = None
    try:
        return download_missing(token)
    except Exception as e:  # noqa: BLE001 — sin red o sin permiso: la app avisa en la barra lateral
        st.sidebar.warning(f"No se pudieron descargar los modelos: {e}")
        return []


@st.cache_resource(show_spinner="Cargando el modelo…")
def load_predictor(model_name: str):
    """Carga el modelo una sola vez por sesión de Streamlit (las predicciones siguientes reusan el objeto)."""
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
    """Barra lateral: elegir el modelo entre los disponibles y mostrar sus métricas guardadas. Devuelve su nombre."""
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
    st.sidebar.caption("Una fruta por foto, fondo simple y buena luz.")
    return model_name


def size_options() -> dict | None:
    """Controles del tamaño: requiere una moneda conocida al lado de la fruta."""
    st.sidebar.header("Tamaño")
    if not st.sidebar.toggle("Medir tamaño (foto con moneda)", value=False):
        return None
    st.sidebar.caption("Ponga una moneda al lado de la fruta, sin tocarla, y tome la foto desde arriba "
                       "a unos 30–50 cm.")
    fruits = list(config.FRUIT_SIZE_CM)
    fruit = st.sidebar.selectbox("Fruta", fruits, index=fruits.index("Otra"), format_func=FRUIT_NAMES.get)
    coins = list(config.COIN_DIAMETERS_MM)
    coin = st.sidebar.selectbox("Moneda", coins, index=coins.index(config.DEFAULT_COIN),
                                format_func=lambda c: f"{COIN_NAMES[c]} ({config.COIN_DIAMETERS_MM[c]} mm)")
    return {"measure_size": True, "fruit": fruit, "coin": coin}


def measurement_overlay(source, size):
    """Foto con el contorno de la fruta (verde) y la moneda detectada (azul)."""
    import cv2
    import numpy as np

    from src.data.preprocess import prepare_image
    from src.features.size import WORK_SIDE

    image = prepare_image(source, (WORK_SIDE, WORK_SIDE)).copy()
    fruit = cv2.resize(size.mascara_fruta.astype(np.uint8), (WORK_SIDE, WORK_SIDE), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 200, 0), 4)
    if size.moneda is not None:
        x, y, r = size.moneda
        cv2.circle(image, (int(x), int(y)), int(r), (30, 90, 255), 4)
    return image


def show_prediction(name: str, source, pred) -> None:
    """Muestra una predicción: foto original, fruta recortada, clase con su confianza y, si se midió, el tamaño."""
    icon, color = CLASS_STYLE[pred.calidad]
    st.subheader(name)
    col_in, col_crop, col_out = st.columns([1, 1, 1.3])
    if pred.tamano is not None and pred.tamano.diametro_cm is not None:
        col_in.image(measurement_overlay(source, pred.tamano), caption="Foto · fruta (verde) y moneda (azul)",
                     width="stretch")
    else:
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
        if pred.tamano is not None:
            t = pred.tamano
            if t.diametro_cm is None:
                st.warning(t.aviso)
            else:
                st.markdown(f"**Tamaño: {t.clase}** · {t.diametro_cm:.1f} cm".replace(".", ","))
                if t.aviso:
                    st.caption(t.aviso)
        total = sum(pred.segundos.values())
        st.caption(f"Tiempo: {total:.2f} s (segmentación {pred.segundos['segmentacion']:.2f} s, "
                   f"modelo {pred.segundos['modelo']:.2f} s)")


def main() -> None:
    """Punto de entrada de la app: carga o captura de imágenes, predicción y resumen de resultados."""
    st.set_page_config(page_title="Calidad de frutas", page_icon="🍎", layout="wide")
    st.title("🍎 Clasificación de calidad de frutas")
    st.caption("Cargue o capture la foto de **una fruta** sobre fondo simple. El sistema quita el fondo, "
               "recorta la fruta y la clasifica como **Buena**, **Defectuosa** (daño superficial) o "
               "**Podrida** (moho, tejido blando o hundido, daño oscuro extenso).")

    ensure_models()
    predictor = load_predictor(sidebar())
    size_opts = size_options() or {}

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
            pred = predictor.predict(f, **size_opts)
        show_prediction(f.name, f, pred)
        results.append({"imagen": f.name, "calidad": pred.calidad,
                        "confianza": None if pred.confianza is None else round(pred.confianza, 3),
                        **({"tamaño": pred.tamano.clase, "diámetro_cm": pred.tamano.diametro_cm} if pred.tamano else {})})
        st.divider()

    if len(results) > 1:
        import pandas as pd

        st.subheader("Resumen del lote")
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True)
        st.bar_chart(df["calidad"].value_counts().reindex(config.QUALITY_CLASSES, fill_value=0))


if __name__ == "__main__":
    main()
