"""Transfer learning: red preentrenada en ImageNet ajustada a la calidad de frutas.

Fase 4 de CRISP-DM (segunda iteración, tras la evaluación de la Fase 5). La CNN desde cero sigue
siendo el modelo que pide el enunciado; este se agrega porque con las fotos propias el F1 cayó a
0,49 por **cambio de dominio** (docs/crisp-dm.md §5), y las características aprendidas con millones
de imágenes variadas suelen ser más robustas a cámaras y luces distintas.

Entrada igual que la CNN: fruta recortada sobre blanco, uint8 128×128×3 (EfficientNet incluye su
propia normalización). Entrenamiento en dos etapas (`train_transfer` en src/training/train.py):
1. Base congelada: solo se entrena la cabeza (pocas épocas, tasa alta).
2. Ajuste fino de toda la red con tasa baja y decaimiento coseno. La BatchNorm de la base queda en
   modo inferencia (`training=False`) para no destruir sus estadísticas con lotes pequeños.
"""
from __future__ import annotations

from src import config

TRANSFER_DEFAULT = {
    "modelo": "transfer",
    "backbone": "efficientnetb0",
    "augment": "none",        # "none" | "geom" (volteos y giros de 90°) | "color" (geom + luz/color suaves)
    "dropout": 0.3,
    "head_epochs": 3,
    "fine_epochs": 15,
    "lr_head": 1e-3,
    "lr_fine": 1e-4,
    "batch_size": 64,
    "weight_power": 1.0,
    "extra": False,           # agregar fuentes solo-entrenamiento (split "extra": FruQ)
    "holdout": None,          # fuente que se deja fuera del entrenamiento y se usa para validar
    "indice": None,           # "v3"/"v4": etiquetas revisadas (src/data/revision.py)
    "frutas": None,           # lista de frutas (train y val); None = todas
}

# Búsqueda principal: ¿qué aumento de datos? ¿aporta FruQ? → tl_busqueda_<i>
TL_GRID = [
    {},
    {"augment": "geom"},
    {"augment": "color"},
    {"extra": True},
    {"extra": True, "augment": "color"},
]

# Generalización entre fuentes (entrena sin una fuente y valida en sus imágenes de val): mide lo
# que la validación normal no ve — rendimiento con fotos de otro origen — sin tocar test ni las
# fotos propias. Incluye la CNN desde cero (`cnn_estable_1`) como referencia → tl_entre_fuentes_<i>
_CNN_FINAL = {"modelo": "cnn", "batch_size": 128, "schedule": "cosine", "augment": False}
TL_LSO = [
    {**_CNN_FINAL, "holdout": "hdf"},
    {**_CNN_FINAL, "holdout": "kaggle"},
    {"holdout": "hdf"},
    {"holdout": "kaggle"},
    {"holdout": "hdf", "augment": "color"},
    {"holdout": "kaggle", "augment": "color"},
    {"holdout": "hdf", "extra": True, "augment": "color"},
    {"holdout": "kaggle", "extra": True, "augment": "color"},
]
# Tercera iteración: los dos modelos finales (CNN `cnn_estable_1` y transfer `tl_busqueda_4`) con la
# misma configuración, entrenados desde cero con las etiquetas revisadas y sin imágenes de varias
# frutas (data/processed/index_v3.csv) → v3_<i>
V3 = [
    {**_CNN_FINAL, "indice": "v3"},
    {"augment": "color", "extra": True, "indice": "v3"},
]
# Cuarta iteración: solo manzana, banano y naranja (config.V4_FRUITS), transfer final. Dos
# configuraciones para separar efectos, fijadas antes de entrenar → v4_<i>:
#   0: etiquetas originales (solo cambia el alcance de frutas)
#   1: etiquetas v4 (revisión v3 + regla nueva del banano, sin imágenes de varias frutas)
_TL_FINAL = {"augment": "color", "extra": True}
V4 = [
    {**_TL_FINAL, "frutas": list(config.V4_FRUITS)},
    {**_TL_FINAL, "frutas": list(config.V4_FRUITS), "indice": "v4"},
]
# Sin "modelo" explícito una configuración es transfer learning (train_config usa la CNN por defecto)
SEARCHES = {search: [{"modelo": "transfer", **c} for c in grid]
            for search, grid in {"tl_busqueda": TL_GRID, "tl_entre_fuentes": TL_LSO, "v3": V3, "v4": V4}.items()}

BACKBONES = {"efficientnetb0": "EfficientNetB0", "mobilenetv3small": "MobileNetV3Small"}


def augmentation(kind: str):
    """Capas de aumento (solo actúan al entrenar). Relleno blanco = mismo fondo que el recorte."""
    from tensorflow.keras import layers

    if kind == "none":
        return []
    stack = [layers.RandomFlip("horizontal_and_vertical"),
             layers.RandomRotation(0.25, fill_mode="constant", fill_value=255.0)]
    if kind == "color":  # cámaras distintas: luz y balance de blancos, suaves para no borrar la señal de color
        stack += [layers.RandomBrightness(0.1, value_range=(0.0, 255.0)),
                  layers.RandomContrast(0.1)]
    return stack


def build_transfer(input_shape=None, num_classes: int | None = None, weights: str | None = "imagenet",
                   **overrides):
    """Devuelve (modelo, base). La base viene congelada; `train_transfer` la descongela en la etapa 2."""
    import tensorflow as tf
    from tensorflow.keras import layers

    cfg = {**TRANSFER_DEFAULT, **overrides}
    input_shape = input_shape or (*config.IMAGE_SIZE, 3)
    num_classes = num_classes or len(config.QUALITY_CLASSES)

    base = getattr(tf.keras.applications, BACKBONES[cfg["backbone"]])(
        include_top=False, weights=weights, input_shape=input_shape, pooling="avg")
    base.trainable = False

    inputs = layers.Input(shape=input_shape)  # valores 0–255 (el uint8 de la caché se convierte solo)
    x = inputs
    for layer in augmentation(cfg["augment"]):
        x = layer(x)
    x = base(x, training=False)  # BatchNorm de la base siempre en modo inferencia
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name=f"tl_{cfg['backbone']}"), base
