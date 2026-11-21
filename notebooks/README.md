# Notebooks

Trazabilidad del proyecto por fase de CRISP-DM. Cada notebook se escribe como `.py` (jupytext,
formato *percent*) y se guarda ejecutado como `.ipynb`. Los notebooks **no repiten el trabajo
pesado**: leen lo que producen los comandos de `src/` (índice, cachés, `experiments/`), así siempre
muestran el estado real del pipeline.

| Notebook | Fase CRISP-DM | Contenido | Requiere haber corrido |
|----------|---------------|-----------|------------------------|
| `01_eda` | 2. Comprensión de los datos | Distribución de clases, desbalanceo, propiedades de las imágenes, señal de color | `download` |
| `02_preprocessing` | 3. Preparación | Fuentes y 3 clases, re-anotación, excluidas, grupos y splits, segmentación y recorte, atajos, características, fotos propias | `index`, `preprocess`, `segment`, `crop`, `features`, `src.evaluation.shortcut` |
| `03_modeling_ml` | 4. Modelado | Líneas base, regresión logística, SVM, Random Forest; ajuste de hiperparámetros, ablación, generalización entre fuentes | `src.training.train --model classic`, `src.evaluation.ablation` |
| `04_modeling_cnn` | 4. Modelado | Arquitectura, búsqueda de configuraciones, curvas de entrenamiento, comparación con el SVM | `src.training.train --model cnn_search` |
| `05_evaluation` | 5. Evaluación | _pendiente_: test, fotos propias, intervalos de confianza, análisis de errores | — |

Regenerar un notebook después de editar su `.py`:

```bash
cd notebooks
jupytext --to ipynb 02_preprocessing.py
jupyter nbconvert --to notebook --execute --inplace 02_preprocessing.ipynb
```

> Los notebooks deben apoyarse en el código reutilizable de `src/` en lugar de duplicar lógica.
