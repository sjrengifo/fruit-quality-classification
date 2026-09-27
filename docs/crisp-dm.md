# Metodología CRISP-DM — Aplicación al proyecto

Este documento desarrolla explícitamente cada fase de **CRISP-DM** adaptada al proyecto de
clasificación de calidad de frutas. Se irá completando con evidencia de trabajo (no solo el diagrama).
El plan operativo y el checklist de rúbrica están en [`../ROADMAP.md`](../ROADMAP.md).

---

## 1. Comprensión del negocio

**Problema.** La clasificación manual de frutas por calidad/tamaño es lenta, subjetiva y propensa a
errores, generando pérdidas económicas y desperdicio.

**Objetivo de negocio.** Automatizar y estandarizar la clasificación de calidad para reducir errores y
pérdidas.

**Objetivo de analítica.** Dada una imagen de una fruta individual, predecir (a) su clase de calidad
`{Buena, Defectuosa, Podrida}` y (b) su tamaño `{Pequeño, Mediano, Grande}`.

**Criterios de éxito.** Superar la línea base y alcanzar un F1-macro objetivo (definir umbral).
Formulación matemática detallada en [`formulacion_matematica.md`](formulacion_matematica.md).

> _Por completar: impacto económico/social cuantificado, KPIs._

---

## 2. Comprensión de los datos

- Dataset base: **Fruit Quality Classification** (Kaggle) + carpeta `mixed_quality`.
- Análisis exploratorio (ver `notebooks/01_eda.ipynb`):
  - Distribución de clases y **desbalanceo**.
  - Calidad de imágenes (resolución, iluminación, fondo), variabilidad.
  - Casos ambiguos / múltiples frutas por imagen.

> _Por completar: hallazgos del EDA con figuras._

---

## 3. Preparación de los datos

- Segmentación de frutas individuales (OpenCV).
- Preprocesamiento: redimensionado, **normalización de píxeles**, augmentation.
- Extracción de características para ML (color HSV, textura GLCM/LBP, forma).
- Estimación de tamaño (diámetro en píxeles normalizados → bins).
- Manejo de **clases desbalanceadas** (class weights / SMOTE / augmentation).
- Splits estratificados train/val/test reproducibles.

> _Por completar: decisiones de preprocesamiento y su justificación._

---

## 4. Modelado

- Línea base (mayoría).
- ≥ 2 modelos ML tradicional con **tuning de hiperparámetros** (GridSearchCV / CV).
- 1 CNN pequeña desde cero (2–3 conv + pooling + densas); opcional transfer learning.
- Registro de experimentos en `experiments/`.

> _Por completar: configuraciones, hiperparámetros óptimos, alternativas descartadas._

---

## 5. Evaluación

- Métricas en test: accuracy, precision/recall/F1 (macro y por clase), matriz de confusión, ROC-AUC.
- Métricas de tamaño (MAE / accuracy por bin).
- Análisis de errores, diagnóstico de **sobreajuste**, comparación con línea base y literatura.

> _Por completar: tablas, gráficas e insights._

---

## 6. Despliegue

- Interfaz **Streamlit** (`src/app/app.py`): cargar/capturar imagen → predicción en tiempo real.
- Video de presentación (≤ 10 min).
- Opcional: línea de empaque con cámara + Raspberry Pi.

> _Por completar: capturas de la app, enlace al video._
