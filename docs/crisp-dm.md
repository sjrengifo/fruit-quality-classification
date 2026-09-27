# Metodología CRISP-DM — Aplicación al proyecto

Este documento desarrolla explícitamente cada fase de **CRISP-DM** adaptada al proyecto de
clasificación de calidad de frutas. Se irá completando con evidencia de trabajo (no solo el diagrama).
El plan operativo y el checklist de rúbrica están en [`../ROADMAP.md`](../ROADMAP.md).

---

## 1. Comprensión del negocio

### 1.1 Contexto y justificación

En mercados y agroindustrias la clasificación de frutas frescas según su **calidad** (madurez,
defectos visibles) y **tamaño** se hace mayoritariamente de forma **manual**: un operario inspecciona
cada producto y lo separa. Este proceso es **lento, subjetivo (varía entre operarios y a lo largo de la
jornada) y propenso a errores**, lo que provoca:

- **Pérdidas económicas** por producto mal clasificado (fruta buena descartada o fruta dañada que
  llega al consumidor y se devuelve).
- **Desperdicio de alimentos:** según la FAO, alrededor del **14 % de los alimentos se pierde entre
  la cosecha y la venta minorista**, y el PNUMA estima que cerca del **17 % adicional se desperdicia**
  en venta y consumo — las frutas y verduras están entre los productos más afectados por su
  perecibilidad. *(Cifras a citar en formato IEEE en el informe final.)*
- **Falta de estandarización**, que dificulta fijar precios justos y cumplir requisitos de calidad.

En el contexto local (plazas de mercado y supermercados de Colombia), un sistema **asistivo** de
clasificación por visión por computadora puede aumentar la consistencia, reducir mermas y liberar
tiempo del personal para tareas de mayor valor.

### 1.2 Objetivos de negocio

- Aumentar la **consistencia y objetividad** de la clasificación de calidad.
- **Reducir mermas** (fruta buena descartada) y el **riesgo de dejar pasar fruta dañada**.
- Ofrecer una herramienta **accesible y de bajo costo** (cámara web/celular + software).

### 1.3 Objetivos de analítica (traducción del negocio a datos)

Dada una imagen de una fruta individual sobre fondo simple, el sistema debe:

1. **Clasificar la calidad** en 3 categorías definidas por el grupo: `Buena`, `Defectuosa`
   (golpes/manchas leves, apto con reservas) y `Podrida` (no apto).
2. **Estimar el tamaño**: `Pequeño`, `Mediano`, `Grande` (a partir del diámetro en píxeles normalizados).

> **Origen de las clases (decisión de proyecto — opción A).** El dataset base de Kaggle aporta solo
> **2 niveles** de calidad (`Good`/`Bad`). Mapeamos `Good → Buena` y `Bad → Podrida`, y construimos
> la clase intermedia **`Defectuosa`** mediante (i) la **recolección propia** de 30–50 imágenes y
> (ii) la **anotación de la carpeta `Mixed Quality`** (que viene sin etiqueta de calidad). Así se
> cumple el requisito del enunciado de "3 o más categorías definidas por grupo" y se le da valor a la
> recolección propia.

### 1.4 Criterios de éxito y KPIs

| KPI | Meta objetivo | Justificación |
|-----|---------------|---------------|
| **F1-macro** (calidad) en test | **≥ 0.85** | Métrica principal; robusta ante el desbalanceo de clases |
| **Recall de `Podrida`** | **≥ 0.90** | El error más costoso es *dejar pasar fruta podrida*; se penaliza el falso negativo |
| **Accuracy de tamaño** (Peq/Med/Grande) | **≥ 0.80** | Salida secundaria del sistema |
| **Mejora sobre línea base** (clasificador de mayoría) | Significativa (≫ baseline) | Demuestra aprendizaje real |
| **Latencia de inferencia** | **< 1 s / imagen** en CPU | Requisito de uso "en tiempo real" en la GUI |

> Los umbrales son metas iniciales; se revisarán tras el EDA (Fase 2) según el desbalanceo real.

### 1.5 ¿Por qué es un problema complejo de ingeniería?

- **Alta dimensionalidad** de la entrada (imágenes de ~128×128×3 ≈ 49k valores).
- **Alta variabilidad:** iluminación, fondo, orientación, oclusión y **múltiples especies** de fruta
  (manzana, banano, guayaba, lima/limón, naranja, granada) con apariencias muy distintas.
- **Clases desbalanceadas** (p. ej. `Pomegranate_Good` ≈ 5940 vs. ~1100 en otras).
- **Fronteras de decisión no lineales** entre calidades (la diferencia Buena/Defectuosa es sutil).
- **Restricción de despliegue:** inferencia rápida en hardware modesto.

La formulación matemática completa (variables, softmax, entropía cruzada, métricas) está en
[`formulacion_matematica.md`](formulacion_matematica.md) (PI3).

### 1.6 Ética e impactos (borrador — se profundiza en Fases 6–7)

Los dilemas éticos (privacidad en la recolección, sesgos entre especies, impacto laboral de la
automatización) y la matriz de impactos social/económico/ambiental/global se documentan en
[`etica.md`](etica.md) (PI1/PI2).

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
