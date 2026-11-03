# 🍎 Clasificación Automática de Calidad de Frutas por Visión por Computadora

Proyecto Final — **Algoritmos y Programación III**, Semestre 2026-2
Facultad de Ingeniería, Diseño y Ciencias Aplicadas · Departamento de Computación y Sistemas Inteligentes · Ingeniería de Sistemas.

#### -- Estado del proyecto: `Activo` 🟢

---

## 📌 Descripción general

En mercados y agroindustrias, la clasificación manual de productos frescos (tomates, manzanas,
papas, etc.) según su tamaño, madurez o defectos visibles es un proceso **lento, subjetivo y
propenso a errores**, lo que genera pérdidas económicas, desperdicio de alimentos y falta de
estandarización.

Este proyecto desarrolla un **sistema automático de clasificación de calidad basado en visión por
computadora** que, a partir de una imagen de una fruta individual sobre fondo simple, entrega:

1. **Clase de calidad** — 3 categorías: `Buena`, `Defectuosa` (golpes/manchas) y `Podrida`.
2. **Estimación de tamaño** — `Pequeño`, `Mediano` o `Grande` (a partir del diámetro en píxeles normalizados).

El sistema se entrena combinando el dataset público **Fruit Quality Classification** con imágenes
recolectadas y anotadas por el equipo, y se evalúa con métricas rigurosas. El desarrollo sigue la
metodología **CRISP-DM** documentada fase por fase (ver [`ROADMAP.md`](ROADMAP.md) y
[`docs/crisp-dm.md`](docs/crisp-dm.md)).

---

## 👥 Integrantes

| Nombre | Rol |
|--------|-----|
| **Juan Manuel Ramírez** | Desarrollo / Modelado |
| **Juan Sebastián Poveda** | Desarrollo / Datos |
| **Samuel José Rengifo** | Desarrollo / Despliegue |

> Curso: Algoritmos y Programación III — Semestre 2026-2.

---

## 🎯 Objetivos

- **General:** construir y evaluar un sistema de clasificación de calidad de frutas por visión por
  computadora que sea reproducible, riguroso y desplegable en una interfaz sencilla.
- **Específicos:**
  - Comprender el problema y su impacto económico/social (Comprensión del negocio).
  - Realizar análisis exploratorio del dataset y detectar desbalanceo/variabilidad.
  - Recolectar y anotar **30–50 imágenes propias** que enriquezcan el dataset base.
  - Entrenar y comparar **≥ 2 modelos de ML tradicional + 1 de Deep Learning (CNN)**.
  - Ajustar hiperparámetros con validación cruzada / búsqueda en rejilla.
  - Analizar resultados (métricas, sobreajuste, comparación con línea base y literatura).
  - Desplegar una GUI (Streamlit) que clasifique en tiempo real desde cámara/imagen.
  - Documentar los aspectos éticos e impactos (social, económico, ambiental, global).

---

## 🧪 Métodos y modelos

Según el enunciado se emplean **al menos dos modelos de ML tradicional y uno de Deep Learning**:

- **ML tradicional** (sobre características extraídas manualmente — color, textura, forma, tamaño):
  Regresión Logística, KNN, SVM, Árboles de Decisión, Random Forest, XGBoost, Bagging.
  Ajuste de hiperparámetros con `GridSearchCV` / validación cruzada.
- **Deep Learning:** CNN pequeña entrenada desde cero (2–3 capas conv + pooling + densas).
  Opcionalmente transfer learning con capas congeladas.
- **MLP:** opcional, sobre características extraídas manualmente.

---

## 🛠️ Tecnologías

`Python 3.11` · `scikit-learn` · `XGBoost` · `TensorFlow/Keras` · `OpenCV` ·
`scikit-image` · `pandas` · `NumPy` · `Matplotlib` / `Seaborn` · `Streamlit` · `Kaggle API`.

---

## 📂 Estructura del repositorio

```
fruit-quality-classification/
├── README.md                  # Este archivo
├── ROADMAP.md                 # Plan de desarrollo por fases (CRISP-DM)
├── requirements.txt           # Dependencias (pip)
├── environment.yml            # Dependencias (conda)
├── LICENSE
│
├── docs/                      # Documentación y entregables
│   ├── crisp-dm.md            # Fases CRISP-DM + diagrama de flujo
│   ├── arquitectura.md        # Arquitectura de software y de modelos
│   ├── instalacion.md         # Guía de instalación y datos
│   ├── etica.md               # Dilemas éticos (PI1) + matriz de impactos (PI2)
│   ├── formulacion_matematica.md  # Formulación matemática del problema (PI3)
│   ├── informe/               # Informe final (estructura IEEE, máx. 7 págs.)
│   └── ...                     # Enunciado, rúbrica, competencias (PDF/XLSX)
│
├── src/                       # Código fuente
│   ├── config.py              # Rutas, constantes, clases, hiperparámetros
│   ├── data/                  # Descarga, segmentación y preprocesamiento
│   ├── features/              # Extracción de características (color/textura/forma)
│   ├── models/                # Definición de modelos ML y CNN
│   ├── training/              # Entrenamiento y tuning de hiperparámetros
│   ├── evaluation/            # Métricas, matrices de confusión, análisis de error
│   ├── utils/                 # Utilidades comunes
│   ├── app/                   # Interfaz Streamlit (despliegue)
│   └── main.py                # Punto de entrada / orquestación
│
├── notebooks/                 # EDA y experimentación (Jupyter)
├── data/                      # Datos (ignorados por git; ver instalacion.md)
│   ├── raw/  interim/  processed/  external/
├── experiments/               # logs / checkpoints / results
├── assets/                    # Imágenes y figuras para docs/informe
└── tests/                     # Pruebas
```

---

## 🚀 Puesta en marcha

```bash
# 1. Clonar
git clone <URL-del-repo> && cd fruit-quality-classification

# 2. Entorno (opción pip)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
#    (opción conda)
# conda env create -f environment.yml && conda activate fruit-quality

# 3. Descargar dataset base (requiere kaggle.json configurado)
python -m src.data.download

# 4. Ejecutar la app de despliegue
streamlit run src/app/app.py
```

Guía detallada de instalación y obtención de datos: [`docs/instalacion.md`](docs/instalacion.md).

---

## 📊 Datos

- **Dataset base:** [Fruit Quality Classification (Kaggle)](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification).
  Se usa además la carpeta `mixed_quality`, segmentando individualmente cada fruta.
- **Recolección propia:** 30–50 imágenes por equipo (mercados, supermercados, hogar) cubriendo
  distintos estados de madurez, tamaños y defectos. Se anotan con clase de calidad y tamaño, y se
  comparten con el resto de la clase para construir un dataset más grande y diverso.

  Se guardan en [`data/external/propias/`](data/external/propias/) (fondo blanco, buena iluminación).

> ⚠️ Los datos **no se versionan** en git (ver `.gitignore`), salvo la recolección propia. Cualquier dato o código de terceros se
> referencia de forma explícita para evitar fraude académico.

---

## 📈 Métricas de evaluación

Accuracy, Precision/Recall/F1 (macro y por clase), matriz de confusión y curvas ROC-AUC para la
clasificación de calidad; MAE/accuracy por bin para la estimación de tamaño. Comparación siempre
contra una **línea base** (clasificador trivial / mayoría). Detalle en [`docs/crisp-dm.md`](docs/crisp-dm.md).

---

## 📦 Entregables

- [ ] Repositorio organizado por fases CRISP-DM.
- [ ] Informe final (máx. 7 págs., formato IEEE) — ver [`docs/informe/`](docs/informe/).
- [ ] Interfaz de despliegue funcional (Streamlit).
- [ ] Video de presentación (≤ 10 min).
- [ ] Análisis ético y de impactos.

Cronograma y checklist completos en [`ROADMAP.md`](ROADMAP.md).

---

## 📄 Licencia

Ver [`LICENSE`](LICENSE).
