# Arquitectura de software y de modelos

## 1. Arquitectura de software

```
Usuario ──► App Streamlit (src/app/app.py)   https://fruit-quality-classification-2026.streamlit.app
                 │  carga o captura una foto (opcional: moneda para el tamaño)
                 ▼
        Predictor (src/models/predict.py)
                 │  prepare_image → U²-Net (segment.py) → recorte centrado 128×128
        ┌────────┴──────────────────────────┐
        ▼                                   ▼
  φ(x): 68 rasgos                     Fruta recortada sobre blanco
  (features/extract_features.py)            │
        ▼                         ┌─────────┴──────────┐
  SVM (models/classic_ml.py)      ▼                    ▼
                           CNN desde cero        EfficientNetB0 (transfer)
                           (models/cnn.py)       (models/transfer.py) ← modelo por defecto
        └───────────────┬─────────┴────────────────────┘
                        ▼
        Calidad + confianza ──► UI          Tamaño (features/size.py): moneda → escala → diámetro
```

Modelos usados por la app: `config.FINAL_TRANSFER` (EfficientNetB0, por defecto), `config.FINAL_MODEL`
(CNN desde cero) y `config.FINAL_CLASSIC` (SVM). En el despliegue los modelos se descargan de la
release `modelos-v1` (`src/app/modelos.py`), porque no se versionan en git.

### Flujo de datos (pipeline)

1. `src/data/download.py` — descarga de Kaggle y Healthy-Defective-Fruits.
2. `src/data/dataset.py` + `split.py` — índice unificado, grupos y splits estables.
3. `src/data/preprocess.py` — formato, orientación y tamaño unificados (128×128).
4. `src/data/segment.py` — máscara de la fruta con U²-Net, fondo blanco y recorte centrado.
5. `src/features/extract_features.py` — características de color/textura/forma (para ML).
6. `src/training/train.py` — entrenamiento y ajuste; artefactos → `experiments/`.
7. `src/evaluation/` — diagnóstico de atajos, ablación, métricas y análisis de error.
8. `src/app/app.py` — despliegue: `src/models/predict.py` (`Predictor`) aplica `segment_and_crop` (mismo camino que la caché de entrenamiento) y el modelo de `config.FINAL_TRANSFER` (EfficientNetB0; también se puede elegir la CNN o el SVM).

Constantes y rutas centralizadas en `src/config.py`.

## 2. Arquitectura de la CNN

```
Entrada uint8 (128×128×3), fruta recortada sobre blanco
 → Rescaling(1/255) → aumento de datos (solo en entrenamiento)
 → 4 × [Conv2D 3×3 (32, 64, 128, 192) → BatchNorm → ReLU → MaxPool 2×2]
 → GlobalAveragePooling → Dense(128) + ReLU + Dropout(0,4)
 → Dense(3) + Softmax        # 3 clases de calidad
```

341 mil parámetros. Hiperparámetros y configuraciones exploradas: `docs/crisp-dm.md` §4.2
(`CNN_DEFAULT` y `CNN_GRID` en `src/models/cnn.py`).


## 3. Transfer learning (modelo de la app)

```
Entrada (128×128×3), valores 0–255 → aumento de color y geométrico (solo entrenamiento)
 → EfficientNetB0 preentrenada en ImageNet (normalización propia, pooling promedio)
 → Dropout(0,3) → Dense(3) + Softmax
Etapa 1: 3 épocas solo la cabeza (base congelada, lr 1e-3)
Etapa 2: 15 épocas de ajuste fino de toda la red (coseno desde 1e-4)
```

4,05 M de parámetros; entrenada con FruQ como fuente solo de entrenamiento. Detalle en
`docs/crisp-dm.md` §4.5 (`TRANSFER_DEFAULT` en `src/models/transfer.py`).
