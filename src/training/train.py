"""Orquestación del entrenamiento de modelos (ML y CNN).

Fase 4 de CRISP-DM. Guarda artefactos en experiments/.
Uso:  python -m src.training.train --model cnn [--data-dir RUTA] [--epochs N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(data_dir: Path) -> tuple[list[str], np.ndarray]:
    """Recorre `data_dir/<clase>/` (clases de config.QUALITY_CLASSES).

    Devuelve las rutas de las imágenes y sus etiquetas como índices enteros.
    """
    paths, labels = [], []
    for idx, name in enumerate(config.QUALITY_CLASSES):
        class_dir = data_dir / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
                labels.append(idx)
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en {data_dir}")
    return paths, np.array(labels)


def split_dataset(paths, labels, seed: int | None = None) -> dict:
    """Split estratificado train/val/test según config.TEST_SIZE y config.VAL_SIZE."""
    from sklearn.model_selection import train_test_split

    seed = seed if seed is not None else config.RANDOM_SEED
    p_rest, p_test, y_rest, y_test = train_test_split(
        paths, labels, test_size=config.TEST_SIZE, stratify=labels, random_state=seed
    )
    p_train, p_val, y_train, y_val = train_test_split(
        p_rest, y_rest, test_size=config.VAL_SIZE, stratify=y_rest, random_state=seed
    )
    return {"train": (p_train, y_train), "val": (p_val, y_val), "test": (p_test, y_test)}


def make_dataset(paths, labels, batch_size: int, shuffle: bool = False):
    """Crea un tf.data.Dataset de (imagen normalizada, etiqueta one-hot)."""
    import tensorflow as tf

    num_classes = len(config.QUALITY_CLASSES)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, config.IMAGE_SIZE)
        # Mismo rango [0, 1] que preprocess.resize_and_normalize (usado en la app).
        image = image / 255.0
        return image, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels)))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=config.RANDOM_SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )


def train_cnn(data_dir=None, epochs: int | None = None, batch_size: int | None = None):
    """Entrena la CNN y guarda el checkpoint del mejor modelo.

    Artefactos:
      - experiments/checkpoints/cnn_best.keras  (mejor val_loss)
      - experiments/logs/cnn_history.csv        (curvas de aprendizaje)
      - experiments/results/cnn_split.csv       (partición usada, para reevaluar)
      - experiments/results/cnn_metrics.json    (métricas en test)
    """
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    from src.evaluation.evaluate import classification_report_dict, confusion
    from src.models.cnn import build_cnn
    from src.utils.helpers import set_seed

    set_seed()
    data_dir = Path(data_dir or config.PROCESSED_DIR)
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    splits = split_dataset(*list_images(data_dir))
    with open(config.RESULTS_DIR / "cnn_split.csv", "w", encoding="utf-8") as f:
        f.write("split,path,label\n")
        for name, (paths, labels) in splits.items():
            for path, label in zip(paths, labels):
                f.write(f"{name},{path},{config.QUALITY_CLASSES[label]}\n")

    train_ds = make_dataset(*splits["train"], batch_size, shuffle=True)
    val_ds = make_dataset(*splits["val"], batch_size)
    test_ds = make_dataset(*splits["test"], batch_size)

    # Pesos de clase para compensar el desbalanceo (docs/crisp-dm.md §3).
    y_train = splits["train"][1]
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    model = build_cnn()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.CHECKPOINTS_DIR / "cnn_best.keras", monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(config.LOGS_DIR / "cnn_history.csv"),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    y_test = splits["test"][1]
    y_pred = model.predict(test_ds).argmax(axis=1)
    metrics = {
        "data_dir": str(data_dir),
        "n_images": {name: len(paths) for name, (paths, _) in splits.items()},
        "epochs_run": len(history.history["loss"]),
        "params": int(model.count_params()),
        "class_weight": class_weight,
        "report": classification_report_dict(y_test, y_pred, labels=config.QUALITY_CLASSES),
        "confusion_matrix": confusion(y_test, y_pred).tolist(),
    }
    with open(config.RESULTS_DIR / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"F1-macro (test): {metrics['report']['macro avg']['f1-score']:.4f}")
    return model, history, metrics


def train_classic():
    """Entrena y ajusta los modelos de ML tradicional (GridSearchCV)."""
    raise NotImplementedError("Implementar en Fase 4: features, tune(), persistencia.")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos")
    parser.add_argument("--model", choices=["cnn", "classic"], default="cnn")
    parser.add_argument("--data-dir", default=None, help="Carpeta con subcarpetas por clase")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    for d in (config.CHECKPOINTS_DIR, config.RESULTS_DIR, config.LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if args.model == "cnn":
        train_cnn(data_dir=args.data_dir, epochs=args.epochs)
    else:
        train_classic()


if __name__ == "__main__":
    main()
