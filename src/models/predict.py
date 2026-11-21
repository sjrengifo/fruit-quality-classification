"""Predicción para una imagen nueva — usada por la app (Fase 6).

Mismo camino que el entrenamiento: `segment_and_crop` (U²-Net + recorte, src/data/segment.py) y
luego la CNN (imagen recortada) o el modelo clásico (vector φ(x) de la fruta recortada).

Uso:  python -m src.models.predict ruta/a/foto.jpg [--model svm]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from src import config


@dataclass
class Prediction:
    calidad: str
    probabilidades: dict[str, float] | None   # None si el modelo no entrega probabilidades (SVM)
    recorte: np.ndarray                       # lo que ve el modelo: fruta recortada sobre blanco
    area_mascara: float                       # fracción de la foto ocupada por la fruta
    segundos: dict[str, float] = field(default_factory=dict)

    @property
    def confianza(self) -> float | None:
        return max(self.probabilidades.values()) if self.probabilidades else None


class Predictor:
    """Carga un modelo de experiments/checkpoints/ una vez y predice imágenes sueltas."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.FINAL_MODEL
        keras_path = config.CHECKPOINTS_DIR / f"{self.model_name}.keras"
        joblib_path = config.CHECKPOINTS_DIR / f"{self.model_name}.joblib"
        if keras_path.exists():
            import tensorflow as tf

            self.kind = "cnn"
            self.model = tf.keras.models.load_model(keras_path, compile=False)
        elif joblib_path.exists():
            import joblib

            self.kind = "clasico"
            self.model = joblib.load(joblib_path)
        else:
            raise FileNotFoundError(f"No hay modelo '{self.model_name}' en {config.CHECKPOINTS_DIR}")

    def predict(self, source) -> Prediction:
        """`source`: ruta, archivo subido o PIL.Image."""
        from src.data.segment import segment_and_crop

        t0 = time.perf_counter()
        image, mask, area = segment_and_crop(source)
        t1 = time.perf_counter()
        if self.kind == "cnn":
            prob = np.asarray(self.model(image[None], training=False))[0].astype(float)
            probs = dict(zip(config.QUALITY_CLASSES, prob))
            label = config.QUALITY_CLASSES[int(prob.argmax())]
        else:
            from src.features.extract_features import extract_all

            label = config.QUALITY_CLASSES[int(self.model.predict(extract_all(image, mask)[None])[0])]
            probs = None
        t2 = time.perf_counter()
        return Prediction(label, probs, image, area, {"segmentacion": t1 - t0, "modelo": t2 - t1})


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice la calidad de una foto de fruta")
    parser.add_argument("imagen")
    parser.add_argument("--model", default=None, help=f"por defecto config.FINAL_MODEL ({config.FINAL_MODEL})")
    args = parser.parse_args()
    p = Predictor(args.model).predict(args.imagen)
    probs = ", ".join(f"{k} {v:.2f}" for k, v in p.probabilidades.items()) if p.probabilidades else "—"
    print(f"{p.calidad} · probabilidades: {probs} · "
          f"{sum(p.segundos.values()):.2f} s (segmentación {p.segundos['segmentacion']:.2f} s)")


if __name__ == "__main__":
    main()
