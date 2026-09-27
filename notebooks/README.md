# Notebooks

Experimentación y análisis exploratorio. Se sugiere la siguiente secuencia (crear como `.ipynb`):

| Notebook | Fase CRISP-DM | Contenido |
|----------|---------------|-----------|
| `01_eda.ipynb` | 2. Comprensión de los datos | Distribución de clases, desbalanceo, calidad/variabilidad de imágenes, ejemplos por clase |
| `02_preprocessing.ipynb` | 3. Preparación | Segmentación, normalización, extracción de características, splits, balanceo |
| `03_modeling_ml.ipynb` | 4. Modelado | ≥ 2 modelos ML tradicional + tuning (GridSearchCV) |
| `04_modeling_cnn.ipynb` | 4. Modelado | CNN desde cero (+ opcional transfer learning) |
| `05_evaluation.ipynb` | 5. Evaluación | Métricas, matriz de confusión, análisis de error, sobreajuste, comparación |

> Los notebooks deben apoyarse en el código reutilizable de `src/` en lugar de duplicar lógica.
