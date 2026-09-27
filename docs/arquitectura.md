# Arquitectura de software y de modelos

## 1. Arquitectura de software

```
Usuario ──► App Streamlit (src/app/app.py)
                 │  carga/captura imagen
                 ▼
        Preprocesamiento (src/data/preprocess.py)
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  Extracción de       Imagen normalizada
  features (ML)         (CNN)
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

1. `src/data/download.py` — descarga del dataset base (Kaggle).
2. `src/data/segment.py` — segmentación de frutas individuales (OpenCV).
3. `src/data/preprocess.py` — redimensionado, normalización, augmentation.
4. `src/features/extract_features.py` — características de color/textura/forma (para ML).
5. `src/training/train.py` — entrenamiento y tuning; artefactos → `experiments/`.
6. `src/evaluation/evaluate.py` — métricas y análisis de error.
7. `src/app/app.py` — despliegue (carga el mejor modelo + pipeline).

Constantes y rutas centralizadas en `src/config.py`.

## 2. Arquitectura de la CNN (propuesta inicial)

```
Input (128×128×3)
 → Conv2D(32,3×3) + ReLU → MaxPool(2×2)
 → Conv2D(64,3×3) + ReLU → MaxPool(2×2)
 → Conv2D(128,3×3)+ ReLU → MaxPool(2×2)
 → Flatten → Dense(128) + ReLU + Dropout(0.5)
 → Dense(3) + Softmax        # 3 clases de calidad
```

> Ajustar tamaño de entrada, número de filtros y regularización según resultados.
> Documentar la arquitectura definitiva y sus hiperparámetros tras la fase de modelado.
