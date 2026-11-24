# Notebooks

Trazabilidad del proyecto por fase de CRISP-DM. Cada notebook se escribe como `.py` (jupytext,
formato *percent*) y se guarda ejecutado como `.ipynb`. Los notebooks **no repiten el trabajo
pesado**: leen lo que producen los comandos de `src/` (índice, cachés, `experiments/`), así siempre
muestran el estado real del pipeline.

| Notebook | Fase CRISP-DM | Contenido | Requiere haber corrido |
|----------|---------------|-----------|------------------------|
| `01_eda` | 2. Comprensión de los datos | Distribución de clases, desbalanceo, propiedades de las imágenes, señal de color | `download` |
| `02_preprocessing` | 3. Preparación | Fuentes y 3 clases, re-anotación, excluidas, grupos y splits, segmentación y recorte, atajos, características, fotos propias; §8 tercera iteración: imágenes con varias frutas y revisión de etiquetas (`index_v3`); §9 cuarta iteración: 3 frutas, regla del banano y fotos nuevas | `index`, `preprocess`, `segment`, `crop`, `features`, `src.evaluation.shortcut` |
| `03_modeling_ml` | 4. Modelado | Líneas base, regresión logística, SVM, Random Forest; ajuste de hiperparámetros, ablación, generalización entre fuentes | `src.training.train --model classic`, `src.evaluation.ablation` |
| `04_modeling_cnn` | 4. Modelado | CNN desde cero: arquitectura, búsquedas, curvas, estabilización y selección; §7 transfer learning y generalización entre fuentes; §8–9 entrenamientos `v3` y `v4`; §10 tabla de **todos** los entrenamientos | `src.training.train --model cnn_search` y `cnn_stable` (Colab), `src.evaluation.evaluate` en val |
| `colab_entrenar_cnn` | 4. Modelado | Entrenar con GPU en Google Colab (no se ejecuta en local); historial de las 4 corridas (`cnn_*`, `tl_*`, `v3`, `v4`) | `src.training.colab` |
| `05_evaluation` | 5. Evaluación | KPIs en test y fotos propias con IC95, matrices de confusión, análisis de errores, sobreajuste, cambio de dominio, ROC-AUC; §6 segunda y tercera mirada a las fotos propias (transfer learning y modelos `v3`); §7 cuarta iteración con fotos nuevas | `src.evaluation.evaluate --split test/externa --final` (CNN y SVM) |

Regenerar un notebook después de editar su `.py`:

```bash
cd notebooks
jupytext --to ipynb 02_preprocessing.py
jupyter nbconvert --to notebook --execute --inplace 02_preprocessing.ipynb
```

> Los notebooks deben apoyarse en el código reutilizable de `src/` en lugar de duplicar lógica.
