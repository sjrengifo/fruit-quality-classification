"""CNN pequeña para clasificación de calidad (entrenada desde cero).

Fase 4 de CRISP-DM. Arquitectura de referencia en docs/arquitectura.md.
"""
from __future__ import annotations

from src import config


def build_cnn(input_shape=None, num_classes: int | None = None):
    """Construye una CNN simple (2-3 bloques conv + pooling + densas).

    Se usa import diferido de TensorFlow para no exigirlo al importar el paquete.
    """
    from tensorflow.keras import layers, models

    input_shape = input_shape or (*config.IMAGE_SIZE, 3)
    num_classes = num_classes or len(config.QUALITY_CLASSES)

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
