"""CNN pequeña para clasificación de calidad (entrenada desde cero).

Fase 4 de CRISP-DM. Entrada: fruta recortada sobre fondo blanco (uint8, 128×128×3); la
normalización a [0, 1] y el aumento de datos van **dentro del modelo**, así la app solo entrega la
imagen preprocesada. Bloques Conv-BN-ReLU-MaxPool + GlobalAveragePooling (pocos parámetros, menos
sobreajuste que Flatten) + densa con dropout.
"""
from __future__ import annotations

from src import config

# Configuración por defecto y rejilla explorada (ver docs/crisp-dm.md §4)
CNN_DEFAULT = {"filters": (32, 64, 128, 192), "dense": 128, "dropout": 0.4, "lr": 1e-3,
               "l2": 1e-4, "augment": True}
CNN_GRID = [
    {},                                              # base
    {"augment": False},                              # ¿aporta el aumento de datos?
    {"filters": (16, 32, 64, 96), "dense": 64},      # red más pequeña
    {"dropout": 0.6, "l2": 5e-4},                    # más regularización
    {"lr": 3e-4},                                    # tasa de aprendizaje menor
]


def build_cnn(input_shape=None, num_classes: int | None = None, **overrides):
    """Construye y compila la CNN. `overrides` reemplaza valores de CNN_DEFAULT.

    Se usa import diferido de TensorFlow para no exigirlo al importar el paquete.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    cfg = {**CNN_DEFAULT, **overrides}
    input_shape = input_shape or (*config.IMAGE_SIZE, 3)
    num_classes = num_classes or len(config.QUALITY_CLASSES)
    reg = regularizers.l2(cfg["l2"])

    stack = [layers.Input(shape=input_shape), layers.Rescaling(1.0 / 255)]
    if cfg["augment"]:  # solo actúa en entrenamiento; relleno blanco = mismo fondo que la segmentación
        stack += [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.25, fill_mode="constant", fill_value=1.0),
            layers.RandomZoom((-0.1, 0.15), fill_mode="constant", fill_value=1.0),
            layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=1.0),
            layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
            layers.RandomContrast(0.1),
        ]
    for n in cfg["filters"]:
        stack += [
            layers.Conv2D(n, 3, padding="same", use_bias=False, kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(2),
        ]
    stack += [
        layers.GlobalAveragePooling2D(),
        layers.Dense(cfg["dense"], activation="relu", kernel_regularizer=reg),
        layers.Dropout(cfg["dropout"]),
        layers.Dense(num_classes, activation="softmax"),
    ]
    model = models.Sequential(stack)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["lr"]),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )
    return model
