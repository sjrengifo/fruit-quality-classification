# 🗺️ Roadmap de Desarrollo — CRISP-DM

Plan de trabajo del proyecto **Clasificación Automática de Calidad de Frutas**, organizado según las
seis fases de **CRISP-DM** (Cross-Industry Standard Process for Data Mining). Cada fase lista los
objetivos, tareas, entregables y su **mapeo con la rúbrica** para asegurar que se cumple todo lo pedido.

> Leyenda de estado: ⬜ Pendiente · 🟨 En progreso · ✅ Completado

---

## 🔁 Diagrama de flujo CRISP-DM (adaptado)

```
                 ┌───────────────────────────┐
                 │ 1. Comprensión del negocio│
                 │  ¿Por qué clasificar       │
                 │   calidad? Impacto.        │
                 └────────────┬──────────────┘
                              │
                 ┌────────────▼──────────────┐
        ┌───────►│ 2. Comprensión de los     │
        │        │    datos (EDA, desbalanceo)│
        │        └────────────┬──────────────┘
        │                     │
        │        ┌────────────▼──────────────┐
        │        │ 3. Preparación de datos   │
        │ (iterar│  segmentación, features,   │
        │  según │  balanceo, splits)         │
        │  hallaz└────────────┬──────────────┘
        │  gos)               │
        │        ┌────────────▼──────────────┐
        │        │ 4. Modelado (ML + CNN,     │
        └────────┤    tuning hiperparámetros) │
                 └────────────┬──────────────┘
                              │
                 ┌────────────▼──────────────┐
                 │ 5. Evaluación (métricas,   │
                 │  errores, vs. línea base)  │
                 └────────────┬──────────────┘
                    ¿objetivos cumplidos?
                        │ sí        │ no → volver a fase 2/3/4
                 ┌──────▼─────────────────────┐
                 │ 6. Despliegue (GUI Streamlit,│
                 │  video, informe final)      │
                 └─────────────────────────────┘
```

> El diagrama definitivo (vectorial, para el informe) se generará y guardará en `assets/`.

---

## Fase 0 — Configuración inicial del proyecto  ✅ / 🟨

**Objetivo:** dejar el repositorio listo para trabajar de forma reproducible.

- [x] Definir problema, clases de calidad (Buena / Defectuosa / Podrida) y tamaño (Peq/Med/Grande).
- [x] Reestructurar repositorio (`src/`, `data/`, `notebooks/`, `docs/`, `experiments/`, `tests/`).
- [x] `README.md`, `requirements.txt`, `environment.yml`, `.gitignore`.
- [x] Crear entorno virtual (`.venv`, Python 3.12) e instalar dependencias — verificado con `pytest` (3/3) y arranque de la app Streamlit.
- [x] Definir flujo de trabajo con git: ramas `main` (estable) y `dev` (desarrollo); trabajo en ramas `feature/*` con PR hacia `dev`.
- [x] Configurar acceso a Kaggle y descargar dataset base en `data/raw/` (~18.4k imágenes).

> **Nota sobre el dataset base:** trae **2 clases de calidad** (`Good` / `Bad`) para 6 frutas
> (Apple, Banana, Guava, Lime/Lemon, Orange, Pomegranate) + una carpeta `Mixed Quality` sin
> etiqueta de calidad. Nuestra **3ª clase (`Defectuosa`)** se construirá a partir de la recolección
> propia y de la anotación de `Mixed Quality` (Fase 3). Hay **fuerte desbalanceo**:
> `Pomegranate_Good` ≈ 5940 imágenes vs. ~1100 del resto (a tratar en Fases 2–3).

**Entregable:** repositorio inicializado y reproducible.

---

## Fase 1 — Comprensión del negocio  ✅

> **Rúbrica:** Criterio 3 (formulación), Criterio 1 (ética), Criterio 2 (impactos).

**Objetivo:** justificar por qué el problema importa y traducirlo a objetivos de analítica.

- [x] Redactar contexto: pérdidas económicas, desperdicio de alimentos, subjetividad manual → `docs/crisp-dm.md` §1.1.
- [x] Definir objetivos de negocio vs. objetivos de minería de datos (KPIs) → `docs/crisp-dm.md` §1.2–1.4.
- [x] **Formulación matemática del problema** (variables, clasificación multiclase, función de
      pérdida) → [`docs/formulacion_matematica.md`](docs/formulacion_matematica.md).
- [x] Argumentar por qué es un **problema complejo de ingeniería** → `docs/crisp-dm.md` §1.5.
- [x] Definir criterios de éxito y métricas objetivo (F1-macro ≥ 0.85; Recall Podrida ≥ 0.90).
- [x] Borrador inicial de **análisis ético e impactos** → [`docs/etica.md`](docs/etica.md).

> _Pendiente de equipo:_ cuantificar el impacto local con cifras citadas (IEEE) al redactar el informe.

**Entregables:** sección "Comprensión del negocio" en `docs/crisp-dm.md`, `formulacion_matematica.md`, borrador `etica.md`.

---

## Fase 2 — Comprensión de los datos  ✅

> **Rúbrica:** Criterio 6 (análisis), Criterio 8 (figuras).

**Objetivo:** conocer a fondo los datos disponibles y su calidad.

- [x] Descargar y explorar el dataset base (`src/data/download.py`).
- [x] EDA en `notebooks/01_eda.ipynb` (figuras en `assets/eda/`):
  - [x] Distribución de clases → **desbalanceo detectado** (ratio ≈ 5×, `Pomegranate_Good`).
  - [x] Resolución/dimensiones (heterogéneas) y formato (`.jpg`).
  - [x] Ejemplos por fruta y calidad (grids).
  - [x] Análisis de color/brillo por clase (señal separable).
  - [x] Análisis de la carpeta `Mixed Quality`.
- [x] Documentar hallazgos en `docs/crisp-dm.md` §2.
- [x] Definir estrategia de recolección propia (30–50 imágenes): fondo blanco, buena iluminación,
      una fruta por foto; se versionan en `data/external/propias/` (protocolo en su `README.md`).

**Entregables:** notebook de EDA con figuras, resumen en `docs/crisp-dm.md`.

---

## Fase 3 — Preparación de los datos  🟨

> **Rúbrica:** Criterio 5 (fundamentos matemáticos: normalización), Criterio 6.

**Objetivo:** obtener datasets limpios, balanceados y listos para modelar.

> **Decisión de datos:** un solo modelo para todas las frutas, 3 clases (Buena / Defectuosa /
> Podrida). `Bad` de Kaggle mezcla defectos leves y podredumbre → se **re-anota** a mano; se
> complementa con **Healthy-Defective-Fruits** (ESPOL, CC BY 4.0, manzana y mango con 3 niveles).
> La recolección propia queda como **conjunto de prueba externo**. Criterios en
> `docs/criterios_anotacion.md`.

- [x] Criterios de anotación de las 3 clases (`docs/criterios_anotacion.md`).
- [x] Descarga de Healthy-Defective-Fruits, solo imágenes reales (`python -m src.main download`).
- [x] Herramienta de re-anotación de `Bad` con medición de acuerdo (kappa) — `streamlit run src/app/anotar.py`.
- [x] Índice unificado de fuentes (`python -m src.main index` → `data/processed/index.csv`).
- [x] **Re-anotar la muestra de `Bad`** (1.200 imágenes): Podrida 842 · Defectuosa 211 · Buena 13 · Descartar 137.
- [ ] Anotación de las 60 imágenes de acuerdo por los otros integrantes (kappa) — _equipo_.

- [ ] **Recolección propia:** capturar y organizar 30–50 imágenes reales.
- [ ] **Anotación manual:** etiquetar calidad + tamaño en `data/external/propias/anotaciones.csv`
      (validar con `python -m src.data.own_images validar`). Se comparte con la clase vía el repo.
- [x] Diagnóstico de atajo por fondo (`python -m src.evaluation.shortcut`): F1 solo-fondo 0,67 (azar 0,33).
- [x] Exclusión de imágenes en escala de grises (1.292, solo `Buena` → atajo).
- [x] **Segmentación** con U²-Net (`python -m src.main segment`): fruta sobre fondo blanco + máscara.
- [x] Verificado con `python -m src.evaluation.shortcut`: F1 solo-fondo 0,68 → 0,20 (azar 0,32). Queda un atajo de encuadre en HDF (silueta 0,64).
- [x] Normalizar el encuadre (`python -m src.main crop`): silueta en HDF 0,64 → 0,37 (azar 0,31).
- [x] Agrupar casi-duplicados de HDF (la misma fruta repetida cientos de veces): F1 en HDF 0,99 → 0,79.
- [ ] `Mixed Quality`: separar frutas múltiples por componentes de la máscara.
- [x] **Preprocesamiento** (`src/data/preprocess.py`): formato/orientación unificados, 128×128 sin
      deformar, normalización [0, 1], caché incremental (`python -m src.main preprocess`).
- [x] Data augmentation dentro de la CNN (giros, rotación, zoom, brillo/contraste).
- [ ] **Estimación de tamaño**: diámetro en píxeles → normalización → bins (Peq/Med/Grande).
- [x] **Extracción de características** para ML tradicional (`src/features/`): histogramas de color
      (HSV), textura (GLCM/LBP), momentos de forma, área/perímetro.
- [x] **Manejo de desbalanceo**: pesos por muestra (1/tamaño del grupo, clases equilibradas); SMOTE descartado.
- [x] **Splits** train/val/test estables por hash y **agrupados** (copias aumentadas de Kaggle) — `src/data/split.py`.

**Entregables:** datos en `data/processed/`, scripts de preparación, dataset propio anotado.

---

## Fase 4 — Modelado  🟨

> **Rúbrica:** Criterio 4 (estrategia y justificación), Criterio 5 (fundamentos matemáticos).

**Objetivo:** entrenar y ajustar modelos (≥ 2 ML tradicional + 1 CNN).

- [x] **Línea base**: mayoría (F1-macro val 0,27) y azar estratificado (0,34).
- [x] **ML tradicional** (`src/models/classic_ml.py`, `python -m src.training.train --model classic`):
  - [x] Regresión logística (0,76), **SVM RBF (0,84)** y Random Forest (0,84) — F1-macro en validación.
  - [x] **Ajuste de hiperparámetros** con `GridSearchCV` y 5 pliegues estratificados por grupo.
  - [x] Ablación de características y prueba entre fuentes (`python -m src.evaluation.ablation`).
- [x] **CNN** (`src/models/cnn.py`, `python -m src.training.train --model cnn`): F1-macro val 0,85–0,89.
  - [x] Arquitectura pequeña desde cero (4 bloques conv + BN + pooling, GAP, densa; 341 mil parámetros).
  - [x] Búsqueda de 5 configuraciones (`--model cnn_search`).
  - [ ] Estabilizar el entrenamiento (el F1 de validación oscila ±0,1 entre épocas) y fijar la configuración final.
  - [ ] Opcional: transfer learning con capas congeladas (no previsto: el enunciado pide CNN desde cero).
  - [x] Callbacks (mejor época por F1-macro de validación, parada temprana, checkpoints en `experiments/checkpoints/`).
- [x] Registrar experimentos (`experiments/results/*.json`, `experiments/logs/`; figuras en `assets/modelado/`).
- [x] Justificar selección de modelos y **alternativas descartadas** (`docs/crisp-dm.md` §4).

**Entregables:** modelos entrenados, notebooks de modelado, logs de experimentos.

---

## Fase 5 — Evaluación  ⬜

> **Rúbrica:** Criterio 6 (análisis e interpretación), Criterio 7 (conclusiones).

**Objetivo:** medir desempeño real y analizar críticamente los resultados.

- [ ] Evaluar en **conjunto de prueba** (`src/evaluation/evaluate.py`).
- [ ] Métricas: accuracy, precision/recall/F1 (macro y por clase), matriz de confusión, ROC-AUC.
- [ ] Métricas de tamaño (MAE / accuracy por bin).
- [ ] **Análisis de errores** (casos mal clasificados, confusiones frecuentes).
- [ ] Diagnóstico de **sobreajuste** (train vs. val/test, curvas de aprendizaje).
- [ ] **Comparación**: modelos entre sí, vs. línea base y vs. literatura (NeurIPS/ICML/ICLR).
- [ ] Extraer *insights* conectados con el problema.

**Entregables:** tablas y figuras de resultados, sección de análisis en el informe.

---

## Fase 6 — Despliegue  ⬜

> **Rúbrica:** Criterio 8 (calidad del entregable). Enunciado: GUI + video.

**Objetivo:** poner el modelo a funcionar para un usuario final.

- [ ] **Interfaz Streamlit** (`src/app/app.py`): cargar imagen o **capturar desde cámara** y mostrar
      predicción (clase de calidad + tamaño) en tiempo real.
- [ ] Cargar el mejor modelo entrenado y pipeline de preprocesamiento.
- [ ] Guía de uso en `docs/instalacion.md` / `docs/arquitectura.md`.
- [ ] *(Opcional)* simulación de línea de empaque con cámara + Raspberry Pi.
- [ ] **Video de presentación** (≤ 10 min): contexto, técnicas, resultados, logros.

**Entregables:** app funcional, video, instrucciones de despliegue.

---

## Fase 7 — Informe final y cierre  ⬜

> **Rúbrica:** Criterios 1, 2, 7 y 8. Enunciado §7 (estructura del informe).

**Objetivo:** consolidar toda la evidencia en el informe (máx. 7 págs., IEEE).

- [ ] Redactar `docs/informe/` siguiendo la estructura: Título, Abstract, Introducción,
      Fundamentos teóricos, Metodología, Resultados, Análisis, Conclusiones, Referencias (IEEE).
- [ ] **Tabla de dilemas éticos** y su relación con el código de ética (PI1).
- [ ] **Matriz de impactos** social/económico/ambiental/global + mitigaciones (PI2).
- [ ] **Justificación matemática** completa del modelado (PI3).
- [ ] Diagramas de calidad **vectorial** (CRISP-DM, arquitectura, resultados).
- [ ] Verificar referencias en formato **IEEE** y documentación del código.
- [ ] Revisión final contra la rúbrica (checklist abajo).

---

## ✅ Checklist de cumplimiento (rúbrica + enunciado)

| # | Requisito | Fuente | Dónde se cubre |
|---|-----------|--------|----------------|
| 1 | Metodología CRISP-DM documentada por fase | Enunciado §4 | `docs/crisp-dm.md`, este roadmap |
| 2 | Diagrama de flujo CRISP-DM personalizado | Enunciado §4 | `assets/`, `docs/crisp-dm.md` |
| 3 | ≥ 2 modelos ML + 1 CNN, con tuning | Enunciado §3 | Fase 4 |
| 4 | Salidas: clase de calidad (≥3) + tamaño | Enunciado §1.3 | Fases 3–4 |
| 5 | Recolección propia de 30–50 imágenes | Enunciado §2.2 | Fase 3 |
| 6 | Anotación manual + compartir con la clase | Enunciado §2.3 | Fase 3 |
| 7 | Segmentación de frutas en `mixed_quality` | Enunciado §2.1 | Fase 3 |
| 8 | GUI de despliegue (Streamlit) | Enunciado §1.4 | Fase 6 |
| 9 | Video ≤ 10 min | Enunciado §6 | Fase 6 |
| 10 | Dilemas éticos + código de ética (PI1) | Competencias / Rúbrica 1 | `docs/etica.md` |
| 11 | Matriz de impactos (PI2) | Competencias / Rúbrica 2 | `docs/etica.md` |
| 12 | Formulación matemática (PI3) | Competencias / Rúbrica 3 | `docs/formulacion_matematica.md` |
| 13 | Estrategia de solución justificada | Rúbrica 4 | Fase 4 / informe |
| 14 | Fundamentos matemáticos aplicados | Rúbrica 5 | Fases 3–5 |
| 15 | Análisis e interpretación de resultados | Rúbrica 6 | Fase 5 |
| 16 | Conclusiones técnicas coherentes | Rúbrica 7 | Fase 7 |
| 17 | Informe IEEE ≤ 7 págs, figuras vectoriales | Enunciado §7 / Rúbrica 8 | `docs/informe/` |
| 18 | Referencias IEEE + código documentado | Enunciado §6 / Rúbrica 8 | Todo el repo |

---

## 📅 Cronograma sugerido

| Semana | Fases | Hito |
|--------|-------|------|
| 1 | 0–1 | Setup + comprensión del negocio + formulación matemática |
| 2 | 2 | EDA completo + estrategia de recolección |
| 3 | 3 | Recolección, anotación, segmentación y features |
| 4 | 4 | Entrenamiento ML + CNN con tuning |
| 5 | 5 | Evaluación y análisis de resultados |
| 6 | 6 | Despliegue (Streamlit) + grabación del video |
| 7 | 7 | Informe final + revisión contra rúbrica |
