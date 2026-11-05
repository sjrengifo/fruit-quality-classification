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

Análisis exploratorio completo en [`../notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb);
figuras en `assets/eda/`.

### 2.1 Inventario

- **19.526 imágenes** en formato **`.jpg`** (100 %), de **6 frutas**: manzana, banano, guayaba,
  lima/limón, naranja y granada.
- Estructura en 3 grupos: `Good Quality_Fruits` (**11.664**, 60 %), `Bad Quality_Fruits`
  (**6.788**, 35 %) y `Mixed Qualit_Fruits` (**1.074**, 5 %, sin etiqueta de calidad).

### 2.2 Distribución y desbalanceo (hallazgo crítico)

- El dataset base solo trae **2 niveles etiquetados** (`Good`/`Bad`). → la 3.ª clase `Defectuosa`
  se construye por anotación propia + `Mixed Quality` (**opción A**, ver §1.3).
- **Desbalanceo fuerte:** `Pomegranate_Good` ≈ **5.940** imágenes domina frente a ~1.100 del resto
  (**ratio ≈ 5×**). Además `Good` > `Bad` a nivel global.
- **Implicación:** usar `F1-macro` como métrica (no accuracy) y aplicar *class weights* /
  *augmentation* / submuestreo en la Fase 3. Ver figuras `dist_calidad.svg`, `dist_fruta_calidad.svg`.

### 2.3 Propiedades de las imágenes

- **Dimensiones heterogéneas:** la mayoría son 256×256 (y 192×256 / 256×192), pero hay imágenes de
  hasta miles de píxeles (p. ej. 3000×4000, 8000×6000). → **redimensionado y normalización
  obligatorios** en la Fase 3 (figura `dimensiones.svg`).

### 2.4 Señal de color

- El color/brillo promedio **difiere entre clases** (en la muestra, brillo `Bad` ≈ 163 vs. `Good`
  ≈ 138). → el color es una **característica prometedora** para los modelos de ML tradicional
  (figura `brillo_por_calidad.svg`).

### 2.5 `Mixed Quality`

- ~1.074 imágenes sin etiqueta de calidad; requieren **segmentación/anotación** manual y son
  candidatas a la clase `Defectuosa`.

### 2.6 Estrategia de recolección propia

- **30–50 imágenes** de las 6 frutas del dataset base, priorizando la clase `Defectuosa`.
- **Condiciones controladas:** fondo blanco liso, buena iluminación (luz difusa, sin flash directo),
  una fruta por foto, vista cenital y distancia fija (~30 cm) para que el tamaño en píxeles sea
  comparable.
- Se redimensionan a lado mayor ≤ 1024 px y se **versionan en el repositorio**
  (`data/external/propias/`), lo que sirve también como mecanismo para compartirlas con la clase.
- Anotación en `anotaciones.csv` (fruta, calidad, tamaño y diámetro medido en cm cuando sea posible).
  Protocolo completo en [`../data/external/propias/README.md`](../data/external/propias/README.md).

---

## 3. Preparación de los datos

### 3.1 Construcción de las 3 clases

El dataset base solo trae `Good`/`Bad`, y `Bad` mezcla defectos superficiales con podredumbre.
Para obtener `Buena / Defectuosa / Podrida` con **las 3 clases de las mismas fuentes** (evitando que
el modelo aprenda a reconocer el dataset de origen en lugar de la calidad):

| Clase | Kaggle (FruitNet) | Healthy-Defective-Fruits (reales) | Propias |
|-------|-------------------|-----------------------------------|---------|
| Buena | `Good` (11.664) | `fresh` (5.000) | prueba externa |
| Defectuosa | `Bad` re-anotado | `bruise`, `scab`, `black_spot` (3.000) | prueba externa |
| Podrida | `Bad` re-anotado | `rot` (2.000) | prueba externa |

- **Re-anotación de `Bad`:** muestra de 200 imágenes por fruta (semilla 42); 10 por fruta las
  anotan todos para medir el **acuerdo entre anotadores** (kappa de Cohen). Criterios en
  [`criterios_anotacion.md`](criterios_anotacion.md).
- **Riesgo de atajo detectado:** en Healthy-Defective-Fruits la mitad de `rot` está en **PNG** y el
  resto en JPEG; todas las imágenes de ese dataset son 256×256 mientras que Kaggle llega a 8000×6000.
  El preprocesamiento debe unificar formato y tamaño para que la clase no sea deducible del archivo.
- Las imágenes propias (fondo blanco) se reservan como **prueba externa** de generalización.

### 3.2 Copias aumentadas en Kaggle (hallazgo crítico)

Kaggle incluye **copias aumentadas** de una misma foto (`20190820_143522.jpg` + `_22896`, `_22897`, …)
y variantes numeradas (`IMG…_01.jpg`). Agrupando por carpeta + instante de captura:

| Fruta | Imágenes | Fotos únicas | Copias |
|-------|---------:|-------------:|-------:|
| Apple | 1.349 | 1.324 | 1,9 % |
| Banana | 1.301 | 1.265 | 2,8 % |
| Guava | 1.342 | 1.307 | 2,6 % |
| Lime | 1.263 | 966 | 23,5 % |
| Orange | 1.386 | 1.196 | 13,7 % |
| **Pomegranate** | **6.089** | **1.291** | **78,8 %** |

- El desbalanceo de `Pomegranate_Good` (≈ 5×) detectado en el EDA (§2.2) es **artificial**: en fotos
  únicas, las 6 frutas están balanceadas (~1.000–1.300 cada una).
- **Riesgo de fuga:** si una copia queda en entrenamiento y otra en prueba, las métricas se inflan.
  → los splits se hacen **por grupo** (`src/data/split.py`); ningún grupo cruza splits.
- **Para el modelado (Fase 4):** ponderar cada imagen por `1 / tamaño de su grupo` para que las
  fotos con muchas copias no dominen el entrenamiento.

### 3.3 Preprocesamiento y splits

- **Preprocesamiento único** (`prepare_image`, usado también por la app): orientación EXIF → RGB →
  lado mayor a 128 px sin deformar → relleno a cuadrado replicando el borde → caché PNG sin pérdida
  (`data/interim/img128/`). Elimina las diferencias de formato y resolución entre fuentes y clases.
- **Splits estables y agrupados:** cada grupo va a train/val/test según un hash de su clave
  (≈ 60/20/20). Al agregar imágenes, las existentes no cambian de split, así los resultados de
  distintas versiones del dataset son comparables. Las fotos propias van siempre a `externa`.

| Split | Buena | Defectuosa | Podrida |
|-------|------:|-----------:|--------:|
| train | 9.913 | 1.944 | 1.703 |
| val | 3.434 | 666 | 553 |
| test | 3.330 | 601 | 586 |

### 3.4 Atajo por fondo (hallazgo crítico)

La revisión visual (`assets/preparacion/muestras_preprocesadas.svg`) muestra que el **fondo está
correlacionado con la clase**: en Healthy-Defective-Fruits las `Buena` están recortadas sobre blanco,
las `Defectuosa` sobre fondo negro o en el árbol y varias `Podrida` tienen esquinas negras de rotación;
en Kaggle, `Good` tiene fondos variados y casi todo `Bad` está sobre la misma mesa gris.

Diagnóstico (`python -m src.evaluation.shortcut`): regresión logística que **solo ve el anillo
exterior de 10 px** de la imagen (fondo), F1-macro en validación:

| Subconjunto | F1 al azar | F1 solo-fondo |
|-------------|-----------:|--------------:|
| Todas | 0,33 | **0,67** |
| Kaggle | 0,33 | 0,50 |
| Healthy-Defective-Fruits | 0,32 | **0,91** |

→ Sin corregirlo, el modelo puede aprender el fondo en lugar de la calidad y fallar con fotos reales
(fondo blanco). **Mitigación:** segmentar la fruta y reemplazar el fondo por uno uniforme antes de
extraer características y entrenar; repetir este diagnóstico después (debe acercarse al azar).

### 3.5 Pasos pendientes

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
