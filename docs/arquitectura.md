# Arquitectura de software y de modelos

## 1. Arquitectura de software

```
Usuario ──► App Streamlit (src/app/app.py)
                 │  carga/captura imagen
                 ▼
        Preprocesamiento + segmentación (src/data/preprocess.py, segment.py)
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  Extracción de       Fruta recortada
  features (ML)       sobre blanco (CNN)
        │                  │
        ▼                  ▼
  Modelo ML          Modelo CNN
  (src/models/       (src/models/cnn.py)
   classic_ml.py)
        └────────┬─────────┘
                 ▼
        Predicción: calidad + tamaño ──► UI
```

### Flujo de datos (pipeline)

1. `src/data/download.py` — descarga de Kaggle y Healthy-Defective-Fruits.
2. `src/data/dataset.py` + `split.py` — índice unificado, grupos y splits estables.
3. `src/data/preprocess.py` — formato, orientación y tamaño unificados (128×128).
4. `src/data/segment.py` — máscara de la fruta con U²-Net, fondo blanco y recorte centrado.
5. `src/features/extract_features.py` — características de color/textura/forma (para ML).
6. `src/training/train.py` — entrenamiento y ajuste; artefactos → `experiments/`.
7. `src/evaluation/` — diagnóstico de atajos, ablación, métricas y análisis de error.
8. `src/app/app.py` — despliegue (carga el mejor modelo + el mismo preprocesamiento).

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
