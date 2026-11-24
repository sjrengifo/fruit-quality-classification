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
| train | 8.577 | 1.865 | 1.853 |
| val | 3.109 | 846 | 535 |
| test | 3.712 | 487 | 454 |

8.647 grupos. Los porcentajes no son exactamente 60/20/20 porque se reparten **grupos**, no imágenes,
y algunos grupos de HDF son muy grandes (§3.7).

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

### 3.5 Imágenes en escala de grises (excluidas)

1.292 imágenes no tienen color (saturación HSV media < 8): **1.188 granadas `Good` de Kaggle** (copias
aumentadas) y 104 de Healthy-Defective-Fruits (91 `fresh`, 13 `scab_defect`), casi todas en `Buena`. Sin color no se puede
juzgar la calidad y el modelo aprendería "gris = Buena" → se **excluyen** del índice
(`data/processed/excluidas.csv`). Quedan **21.438 imágenes**: Buena 15.398 · Defectuosa 3.198 ·
Podrida 2.842.

### 3.6 Segmentación (mitigación del atajo de fondo)

- **Método:** U²-Net (Qin et al., 2020, *Pattern Recognition* 106:107404) vía `rembg`, una red
  **preentrenada solo para separar el objeto principal del fondo**. Se eligió frente a GrabCut
  (OpenCV) porque las imágenes de Healthy-Defective-Fruits tienen fondos complejos (árbol, hojas,
  fondo negro) y manos; en una comparación visual U²-Net quitó incluso manos con guante, y la
  variante liviana `u2netp` dejaba restos de fondo. El clasificador de calidad se entrena **desde cero**.
- **Pipeline** (`src/data/segment.py`): imagen a 320 px → máscara (umbral 0,5) → se eliminan
  componentes < 1 % del área → fondo **blanco** (como la recolección propia) → 128×128. Se guardan la
  imagen sin fondo y la máscara (sirve para forma y tamaño). Si la máscara queda casi vacía (< 2 %) se
  conserva la imagen original.
- **Costo:** ~0,7 s/imagen con 2 procesos (21.438 imágenes en ≈ 4 h 20 min en un portátil de 16 GB;
  incremental, se puede interrumpir y retomar).
- **Resultado:** 0 errores; 39 imágenes (0,2 %) con máscara casi vacía conservan la original
  (19 `Defectuosa` de HDF, 18 `Buena` y 2 `Podrida` de Kaggle). Área mediana de la fruta: 17 % de la
  imagen (p5 = 5,5 %, p95 = 48 %). Muestras: `assets/preparacion/muestras_segmentadas.svg`.

**Verificación del atajo** (`python -m src.evaluation.shortcut`, F1-macro en validación):

| Prueba | Todas | Kaggle | HDF |
|--------|------:|-------:|----:|
| Al azar | 0,34 | 0,34 | 0,31 |
| Fondo, imagen original | **0,65** | 0,52 | **0,76** |
| Fondo, imagen segmentada | 0,18 | 0,36 | 0,33 |
| Silueta (área, caja, centroide, forma; sin color) | 0,48 | 0,33 | 0,45 |
| Silueta, fruta recortada | 0,47 | 0,32 | 0,37 |

(Cifras con los splits corregidos de §3.7; con los splits iniciales el fondo original daba 0,90 en
HDF y la silueta 0,64. La silueta en "Todas" refleja que la forma delata la fruta y cada fuente
tiene otra mezcla de frutas y clases.)

- **El atajo de fondo desapareció:** con fondo blanco el anillo exterior ya no predice la clase
  (todas las cifras al nivel del azar).
- **Quedaba un atajo de encuadre en HDF:** sin ver color, la silueta predecía la clase (0,64): las
  `Buena` llenan el cuadro (recortadas) y muchas `Podrida` están lejos y pequeñas.
- **Normalización del encuadre** (`crop_fruit`, `python -m src.main crop`): se recorta la caja
  cuadrada de la máscara (+5 % de margen) y se reescala a 128×128, de modo que todas las frutas
  quedan centradas y con el mismo zoom (`data/interim/crop128/`; los modelos entrenan con esta
  variante). La máscara sin recortar se conserva para estimar el tamaño. Con el recorte y los grupos
  corregidos (§3.7) la silueta en HDF baja a **0,37** (azar 0,31) y en Kaggle a 0,32 (azar 0,34).

### 3.7 Casi-duplicados en Healthy-Defective-Fruits (hallazgo crítico)

Los primeros modelos dieron F1 = 0,99 en HDF y 0,80 en Kaggle: demasiado bueno. Al revisar las
imágenes en orden, HDF **repite la misma fruta o la misma sesión de fotos decenas o cientos de
veces** (ráfagas girando la fruta y copias exactas): p. ej. las 2.500 manzanas `fresh` provienen de
unas 12 sesiones. Como cada imagen era su propio grupo, la misma fruta caía en train y en val.

- **Corrección** (`near_duplicate_groups`, `src/data/split.py`): dentro de cada carpeta de HDF se
  unen en un grupo las imágenes con histograma de color HSV casi idéntico (distancia L1 < 0,25,
  enlace simple), y el grupo completo va a un solo split. HDF pasa de 9.896 grupos a 1.298.
- **Efecto:** F1-macro de validación del SVM en HDF 0,99 → **0,79** (cifra honesta); global 0,96 → 0,84.
- **Limitación:** los grupos son *sesiones de captura* (pueden incluir varias frutas parecidas), por
  lo que el reparto por split es irregular (casi no hay manzanas `fresh` de HDF en val). En Kaggle
  los grupos siguen siendo por instante de captura; fotos distintas de una misma fruta pueden cruzar
  splits.

### 3.8 Extracción de características (ML tradicional)

`src/features/extract_features.py` (`python -m src.main features`) calcula φ(x) ∈ ℝ⁶⁸ usando **solo
los píxeles de la fruta** (máscara):

| Grupo | n | Contenido |
|-------|--:|-----------|
| Color | 38 | Histogramas H (16), S (8), V (8) + media y desviación por canal |
| Textura | 18 | Histograma LBP uniforme (P = 8, R = 1; 10) + GLCM (contraste, homogeneidad, energía, correlación a d = 1 y 3; promedio de 4 ángulos) |
| Forma | 12 | Área relativa, aspecto, circularidad, solidez, extensión, 7 momentos de Hu (log) |

**Desbalanceo:** peso por muestra = 1/tamaño del grupo (30 copias de una foto cuentan como una),
reescalado para que las tres clases pesen lo mismo (`sample_weights`, `src/training/train.py`). Se
descartó SMOTE: interpolar vectores de características de frutas distintas no produce frutas
plausibles y los pesos logran el mismo efecto sin datos sintéticos.

### 3.9 FruQ: fuente solo de entrenamiento (segunda iteración)

Tras la Fase 5 (F1 0,49 con las fotos propias por cambio de dominio, §5) se añadió variedad de
fruta y cámara sin tocar las 81 fotos de prueba. **FruQ** (Zenodo 7224690, CC BY 4.0) son
*time-lapses*: cada video es **una sola fruta** que pasa de fresca a podrida.

- Se usan solo las carpetas de fruta entera y sola (`config.FRUQ_FRUITS`: banano, kaki, papaya,
  durazno, pera), 1 de cada 3 cuadros. `download_fruq` descarga solo esos cuadros del zip de 4,6 GB
  con lecturas por rangos HTTP (con límite de velocidad: Zenodo responde 429).
- Mapeo `Fresh → Buena`, `Mild → Defectuosa`, `Rotten → Podrida`, revisado sobre los recortes.
  Descartados: **fresa** (Mild y Rotten iguales a 128 px), **durazno Mild** (el video hace zoom y la
  fruta sale del cuadro) y **banano Mild** (solo pecas de maduración: para nuestro criterio es Buena).
- Grupos = ventanas de 10 cuadros muestreados de un mismo video (cuadros vecinos son casi
  idénticos). Quedan **1.515 imágenes en 309 grupos**.
- Es **solo de entrenamiento** (split `extra`, `config.TRAIN_ONLY_SOURCES`): todos los cuadros son la
  misma fruta, así que en val/test medirían memoria del objeto, no generalización. El entrenamiento
  solo la usa con `extra=True`. Las 21.519 imágenes previas no cambiaron de split (verificado).

### 3.10 Pasos pendientes

- Estimación de tamaño (diámetro en píxeles normalizados → bins). _Hecho: §5.5._
- `Mixed Quality`: separar frutas múltiples por componentes de la máscara. _No se usa; el mismo
  conteo de componentes sirvió para excluir imágenes con varias frutas (§3.11)._

### 3.11 Tercera iteración: varias frutas y revisión de etiquetas

**Motivación.** En la segunda mirada a las fotos propias (§5.4) el transfer learning seguía
mandando 20 de 30 Podrida a Defectuosa. Hipótesis del equipo: parte del error viene de los
**datos**, no del modelo: (1) imágenes con varias frutas, que no son el caso de uso; (2) una
frontera Defectuosa/Podrida mal definida en las etiquetas, de modo que el modelo aprende como
Defectuosa fruta que nuestro criterio llama Podrida.

**Qué se hizo** (`src/data/revision.py`, `notebooks/02_preprocessing` §8). Nada de lo anterior se
modificó: el resultado es un índice aparte, `data/processed/index_v3.csv`.

1. **Varias frutas:** componentes conexas de la máscara de U²-Net que miden ≥ 15 % de la mayor.
   **1.816 imágenes** con 2 o más, casi todas de Kaggle `Buena` (1.712; Defectuosa/Podrida: 104). Era
   un atajo posible: "varias frutas → Buena". Muestra de 30: 30/30 correctas. No separa frutas que
   se tocan; esas se marcaron a mano en el paso 2.
2. **Revisión visual a ciegas** (sin etiqueta ni fuente) de **2.522 grupos**: todos los de
   Defectuosa y Podrida (Kaggle, HDF, FruQ) y los de `Buena` de HDF y FruQ. Una celda por grupo
   (representante, o mosaico 2×2 si tiene más de 5 imágenes); los 23 grupos de 40 o más imágenes
   se verificaron después con 24 imágenes cada uno (6 decisiones del mosaico se corrigieron).
   Criterio: `docs/criterios_anotacion.md`. Las etiquetas las asignó **un modelo de IA (Claude)** a
   pedido del equipo, por falta de tiempo para revisarlas a mano; se declara en el informe.
3. **Índice v3:** etiqueta revisada por grupo, sin varias frutas ni `Descartar`. Los splits no
   cambian (hash por grupo). Kaggle `Buena` (6.353 grupos) solo se filtró por varias frutas.

| Fuente | Grupos revisados | Descartados | κ original ↔ revisada | Cambio principal |
|---|---:|---:|---:|---|
| Kaggle (re-anotación de Samuel) | 917 | 207 | **0,29** | Podrida → Defectuosa (248 grupos) |
| HDF (carpetas del dataset) | 1.296 | 25 | 0,40 | Podrida → Defectuosa (323), sobre todo mango `rot_defect` con manchas pequeñas |
| FruQ (etapa del video) | 309 | 1 | 0,86 | Defectuosa → Buena (18) |

En imágenes: 945 Podrida → Defectuosa, 137 Defectuosa → Podrida, 287 → Buena, 311 → Descartar y
1.816 excluidas por varias frutas. El índice v3 tiene 20.826 imágenes (antes 22.953); Podrida en
`val` baja de 535 a 337 y en `test` de 454 a 204.

**Hallazgo.** La revisión hizo Podrida **más exigente**, lo contrario de lo que pedían los errores
con fotos propias. Y κ = 0,29 con la re-anotación de Kaggle (el documento de criterios pide ≥ 0,6)
muestra que **la frontera Defectuosa/Podrida es ambigua aun con el mismo criterio escrito**.
Resultado del reentrenamiento: §4.6 y §5.6.

### 3.12 Cuarta iteración: alcance de 3 frutas y fotos nuevas (plan fijado antes de entrenar)

Ideas del equipo tras §5.6 y decisión tomada sobre cada una:

| Idea | Decisión | Por qué |
|---|---|---|
| Quedarse con frutas donde el daño contrasta (manzana, banano, naranja) | **Sí** | Por fruta, el F1 con fotos propias ya era mayor en naranja (0,65) y banano (0,56) que en guayaba (0,22) o mango (0,29). Se reporta como decisión de **alcance**, no como mejora del modelo. |
| Criterio Defectuosa/Podrida "evidente" | **Sí, como regla; no quitando los casos dudosos** | Entrenar solo con ejemplos obvios deja al modelo sin ver la frontera, y en la práctica los casos dudosos siguen existiendo. Se ajustó la regla del banano (`docs/criterios_anotacion.md`). |
| Fotos nuevas del equipo (35, 24 nov.) | **Sí, como prueba limpia** | Frente a ~5.000 imágenes de entrenamiento no pesan; como prueba son lo único que ningún modelo ha visto (las 81 anteriores ya se miraron tres veces). |
| Buscar más fotos en internet | No | Las fotos de banco de imágenes (ya presentes en Kaggle `Bad`, con marcas de Alamy) muestran pudrición extrema: refuerzan el sesgo "podrido = muy podrido"; además requieren etiquetado, derechos y control de duplicados. |
| Generar imágenes con IA | No (trabajo futuro) | Texturas sintéticas = posible atajo; no sirven para evaluar; habría que revisar cada etiqueta. |

**Plan** (commit de este cambio, antes de entrenar; nada se elige después de ver las pruebas):
- Búsqueda `v4` (`src/models/transfer.py`), transfer final (`tl_busqueda_4`: color + FruQ) solo con
  las 3 frutas en train y val: **`v4_0`** con etiquetas originales (efecto del alcance) y **`v4_1`**
  con etiquetas v4 = revisión v3 + regla del banano, sin imágenes de varias frutas
  (`python -m src.data.revision indice4` → `index_v4.csv`).
- Evaluación (`python -m src.evaluation.iteracion4`) frente al modelo actual de la app:
  1. **Fotos nuevas** (`data/external/propias/nov24/`, 32 de las 3 frutas; primera mirada; con el
     camino de la app): Buena reconocidas como Buena y Podrida detectadas. Sin Defectuosas y con 6
     Podridas no se calcula F1 de 3 clases. Los 3 limones se muestran aparte.
  2. **Las 81 fotos anteriores restringidas a las 3 frutas** (62; cuarta mirada): F1-macro y
     diferencias pareadas frente al modelo actual.
- Antes de entrenar se revisó que la segmentación funcione con el fondo de las fotos nuevas (solo los
  recortes, sin predicciones). Las fotos nuevas sí se vieron con sus etiquetas al planear esta
  iteración; no se usaron para decidir ninguna etiqueta de entrenamiento.

---

## 4. Modelado

Protocolo común: se ajusta con `train`, se compara con `val`; `test` y las fotos propias (`externa`)
no se tocan hasta la Fase 5. Métrica: **F1-macro** (clases desbalanceadas). Todos los modelos usan
la fruta recortada sobre blanco y los pesos por muestra de §3.8.
Reproducir: `python -m src.training.train --model classic` / `--model cnn`.

### 4.1 Líneas base y ML tradicional

Ajuste con `GridSearchCV` (F1-macro) y 5 pliegues **estratificados por grupo**
(`StratifiedGroupKFold`): las copias de una misma foto nunca quedan en ambos lados de un pliegue.
El escalado (`StandardScaler`) va dentro del pipeline para no filtrar estadísticas de validación.

| Modelo | Rejilla | Mejor configuración | F1 CV | F1 train | F1 val | Exactitud val |
|--------|---------|---------------------|------:|---------:|-------:|--------------:|
| Mayoría | — | — | — | — | 0,27 | 0,69 |
| Azar estratificado | — | — | — | — | 0,34 | — |
| Regresión logística | C ∈ {0,01…100} | C = 1 | 0,76 ± 0,06 | 0,77 | 0,76 | 0,81 |
| **SVM (RBF)** | C ∈ {1, 10, 100}; γ ∈ {scale, 0,03, 0,003} | C = 100, γ = scale | **0,89 ± 0,01** | 1,00 | **0,84** | 0,88 |
| Random Forest (300 árboles) | profundidad ∈ {∞, 12, 20}; hojas ∈ {1, 3}; max_features ∈ {√, 0,3} | ∞, 1, √ | 0,86 ± 0,06 | 0,95 | 0,84 | 0,87 |

Métricas por clase en validación (precisión / recall):

| Modelo | Buena | Defectuosa | Podrida | F1 Kaggle | F1 HDF |
|--------|-------|------------|---------|----------:|-------:|
| Regresión logística | 0,97 / 0,80 | 0,57 / 0,79 | 0,63 / 0,88 | 0,62 | 0,70 |
| SVM | 0,96 / 0,89 | 0,67 / 0,79 | 0,83 / **0,92** | 0,79 | 0,79 |
| Random Forest | 0,93 / 0,91 | 0,68 / 0,71 | 0,87 / **0,93** | 0,80 | 0,79 |

- Los modelos no lineales superan con claridad a la referencia lineal (0,84 vs. 0,76) y triplican la
  línea base de mayoría (0,27): la frontera entre clases no es lineal en φ(x).
- **Recall de Podrida ≥ 0,90** (KPI de §1.4) se cumple en validación con SVM y Random Forest; el
  F1-macro (0,84) queda justo debajo de la meta de 0,85.
- La clase difícil es **Defectuosa** (se confunde sobre todo con Buena: 311 Buenas predichas como
  Defectuosa en el SVM), coherente con que es la categoría intermedia y la más subjetiva al anotar.
- SVM y Random Forest memorizan el entrenamiento (F1 train 1,00 / 0,95): hay sobreajuste, acotado
  por la validación cruzada por grupos.

**Ablación** (SVM, F1-macro en validación; `python -m src.evaluation.ablation`):

| Características | F1 val |
|-----------------|-------:|
| Solo color | 0,75 |
| Solo textura | 0,74 |
| Solo forma | 0,56 |
| Color + textura | 0,82 |
| Color + textura + forma | **0,84** |

Color y textura aportan información complementaria; la forma sola es débil pero suma 2 puntos.

**Generalización entre fuentes** (hallazgo crítico; SVM entrenado con una sola fuente):

| Entrena ↓ / valida → | Kaggle | HDF |
|----------------------|-------:|----:|
| Kaggle | 0,79 | 0,42 |
| HDF | **0,06** | 0,79 |

Un modelo entrenado con una fuente **no sirve en la otra**: cada dataset tiene su propio "aspecto"
(frutas, cámaras, iluminación, criterio de etiquetado) y las características de color/textura lo
capturan. El modelo conjunto funciona en ambas, pero esto anticipa que el desempeño con las **fotos
propias** puede ser bastante menor que en validación; por eso la prueba externa es la evaluación que
más importa en la Fase 5.

### 4.2 CNN desde cero

Arquitectura (`src/models/cnn.py`, 341 mil parámetros, entrenada desde cero):

```
Entrada uint8 128×128×3 → Rescaling(1/255) → [aumento de datos, solo al entrenar]
 → 4 × [Conv2D 3×3 (32, 64, 128, 192) → BatchNorm → ReLU → MaxPool 2×2]
 → GlobalAveragePooling → Dense(128, ReLU) → Dropout(0,4) → Dense(3, softmax)
```

- **Entrenamiento:** Adam, entropía cruzada con los pesos por muestra de §3.8, lotes de 32, máximo
  40 épocas; se conserva la época con mejor F1-macro en validación y se detiene tras 8 épocas sin
  mejorar (`experiments/checkpoints/`, historia en `experiments/logs/`).
- **Aumento de datos:** volteos, rotación ±90°, zoom, traslación (relleno blanco) y brillo/contraste
  ±10 % (suave, porque el color es la señal de calidad).
- GlobalAveragePooling en lugar de Flatten: 341 mil parámetros frente a ~4 millones.

Búsqueda de configuraciones (`--model cnn_search`, una variación a la vez sobre la base):

| Configuración | Épocas | F1 train | F1 val | F1 Kaggle | F1 HDF | Recall Podrida |
|---------------|-------:|---------:|-------:|----------:|-------:|---------------:|
| Base (lr 1e-3, dropout 0,4, L2 1e-4, con aumento) | 37 | 0,89 | 0,87 | 0,71 | 0,86 | 0,91 |
| **Sin aumento de datos** | 40 | 0,93 | **0,89** | 0,77 | 0,88 | 0,88 |
| Red pequeña (16-32-64-96, densa 64; 86 mil parámetros) | 37 | 0,79 | 0,85 | 0,72 | 0,83 | 0,93 |
| Más regularización (dropout 0,6, L2 5e-4) | 29 | 0,85 | 0,81 | 0,72 | 0,79 | 0,75 |
| Tasa de aprendizaje 3e-4 | 32 | 0,87 | 0,86 | 0,68 | 0,87 | 0,87 |

Figuras: `assets/modelado/` (`comparacion_modelos`, `cnn_curvas_validacion`, `matrices_confusion`;
se regeneran con `python -m src.evaluation.figuras_modelado`).

- La CNN supera a los modelos clásicos en validación (0,85–0,89 vs. 0,84) y a la meta de 0,85. La
  ventaja viene de HDF; **en Kaggle queda por debajo del SVM** (0,68–0,77 vs. 0,79).
- **Entrenamiento inestable:** el F1 de validación oscila ±0,1 entre épocas consecutivas en todas
  las configuraciones (figura `cnn_curvas_validacion`), también con tasa de aprendizaje menor. Como
  se elige la mejor de ~35 épocas ruidosas usando la propia validación, **el F1 de validación de la
  CNN es optimista** y las diferencias entre configuraciones (0,85–0,89) no son concluyentes; la
  cifra fiable saldrá del conjunto de prueba (Fase 5). Causa probable por verificar: pesos por
  muestra muy desiguales (1/tamaño de grupo) combinados con BatchNorm y lotes de 32.
- **Error más costoso:** la mejor CNN deja pasar 9 % de las `Podrida` como `Buena` (SVM: 4 %) y su
  recall de Podrida (0,88) no cumple el KPI de 0,90, que el SVM sí cumple (0,92).
- No hay evidencia de que el aumento de datos ayude en validación; se mantiene como opción porque su
  beneficio esperado es fuera de distribución (fotos propias), que la validación no mide.

**Estabilización** (`--model cnn_stable`, entrenada en Google Colab con GPU T4 y precisión mixta,
`notebooks/colab_entrenar_cnn`; ~5–10 min por configuración frente a ~1,5 h en CPU). Cambios sobre
la base: lotes de **128** (el gradiente con pesos por muestra muy desiguales es menos ruidoso) y
tasa de aprendizaje con **decaimiento coseno** hasta 0 en 40 épocas, sin parada temprana. Se reporta
además la media y desviación del F1 de validación en las **últimas 5 épocas**, que no depende de
elegir una época con la propia validación:

| Configuración (lote 128 + coseno) | Mejor época | F1 val | IC95 (grupos) | Últimas 5 (media ± desv.) | F1 Kaggle | F1 HDF | Recall Podrida |
|------------------------------------|------------:|-------:|:-------------:|:-------------------------:|----------:|-------:|---------------:|
| Con aumento de datos               | 36 | 0,88 | 0,82–0,91 | 0,870 ± 0,004 | 0,74 | 0,87 | 0,85 |
| **Sin aumento de datos**           | 30 | **0,91** | 0,85–0,95 | **0,890 ± 0,007** | **0,83** | **0,90** | **0,94** |
| Pesos 1/√grupo                     | 40 | 0,88 | 0,82–0,92 | 0,879 ± 0,002 | 0,72 | 0,88 | 0,88 |
| Pesos 1/√grupo, lr 2e-3            | 35 | 0,89 | 0,80–0,94 | 0,884 ± 0,002 | 0,74 | 0,88 | 0,91 |

Figura: `assets/modelado/cnn_curvas_estable`.

- **La inestabilidad era del optimizador, no del modelo:** con lote 128 y coseno, la desviación del
  F1 en las últimas épocas baja de ±0,1 a ±0,002–0,007, y la mejor época queda a ≤ 0,02 de la media
  final → la cifra de validación ya no depende de una época afortunada.
- **Sin aumento de datos es mejor por segunda vez** (también en la búsqueda anterior) y es la única
  configuración que supera al SVM **también en Kaggle** (0,83 vs. 0,79) y cumple el KPI de recall
  de Podrida (0,94). Frente a las otras tres configuraciones, la diferencia pareada en F1-macro
  (bootstrap por grupos, mismas imágenes) es positiva con IC95 por encima de 0 en las tres
  (+0,02 a +0,04). Hipótesis: rotaciones de ±90° y brillo/contraste alteran justo las señales de
  color y forma que distinguen la calidad, y el recorte (§3.6) ya normaliza encuadre y escala.
- Riesgo asumido: sin aumento, la red puede ser más frágil ante fotos distintas (cámara, luz). Eso
  lo mide la evaluación con las fotos propias (Fase 5), no la validación.
- Los pesos 1/√grupo no aportan; se mantienen los pesos 1/grupo de §3.8.

### 4.3 Selección del modelo final (antes de abrir `test`)

Se decide **solo con validación** y se registra en `src/config.py` (`FINAL_MODEL`, `FINAL_CLASSIC`)
antes de la evaluación final, para que la elección no pueda depender de `test`:

- **Modelo final: CNN `cnn_estable_1`** (lote 128, coseno, sin aumento de datos). Frente al SVM, en
  las mismas 4.490 imágenes de validación: **F1-macro +0,07, IC95 [+0,01; +0,12]** (diferencia
  real); recall de Podrida +0,02, IC95 [−0,01; +0,06] (empate). Además es mejor en ambas fuentes.
- **Comparación: SVM (RBF)**, el mejor modelo de ML tradicional. Se evalúa en `test` y fotos
  propias junto a la CNN, porque el enunciado pide comparar ambos enfoques; esa evaluación no cambia
  la elección.

### 4.4 Alternativas descartadas

- **SMOTE / sobremuestreo sintético:** ver §3.8.
- **Transfer learning en la primera iteración:** el enunciado pide una CNN desde cero; se agregó
  después, como segunda iteración (§4.5), con autorización del profesor.
- **GrabCut / umbral de color para segmentar:** falla con los fondos complejos de HDF (§3.6).
- **Partición aleatoria por imagen:** infla las métricas por copias y casi-duplicados (§3.2, §3.7).

### 4.5 Segunda iteración: transfer learning

Objetivo: reducir la caída con fruta y cámara reales (§5.2). **EfficientNetB0** preentrenada en
ImageNet (`src/models/transfer.py`, 4,05 M parámetros) sobre el mismo recorte 128×128: 3 épocas
solo de cabeza con la base congelada (tasa 1e-3) y 15 de ajuste fino de toda la red (coseno desde
1e-4; BatchNorm en modo inferencia). Entrenado en Colab (T4), ~10–15 min por configuración.

**Criterio de selección fijado antes de entrenar:** validación + **generalización entre fuentes**
(`tl_entre_fuentes`: entrenar sin una fuente y validar solo en sus imágenes de val). Así se mide el
cambio de dominio sin gastar test ni las fotos propias. Diferencias pareadas con IC95 por grupos.

| Modelo (`tl_busqueda_i`) | F1 val [IC95] | vs. CNN | Recall P |
|---|---|---|---:|
| CNN desde cero (`cnn_estable_1`) | 0,913 [0,85; 0,95] | — | 0,94 |
| TL sin aumento (0) | 0,904 [0,81; 0,96] | −0,01 [−0,07; +0,03] | 0,96 |
| TL + aumento geométrico (1) | 0,902 [0,82; 0,95] | −0,01 [−0,07; +0,03] | 0,93 |
| TL + aumento de color (2) | 0,909 [0,84; 0,96] | −0,00 [−0,06; +0,04] | 0,94 |
| TL + FruQ (3) | 0,915 [0,83; 0,97] | +0,00 [−0,03; +0,03] | 0,96 |
| **TL + color + FruQ (4)** | **0,912 [0,85; 0,96]** | −0,00 [−0,04; +0,03] | 0,95 |

| Entre fuentes (F1 en la fuente no vista) | valida en HDF | valida en Kaggle |
|---|---|---|
| CNN desde cero | 0,49 [0,32; 0,62] | 0,36 [0,34; 0,39] |
| TL | 0,58 (+0,09 [−0,12; +0,30]) | 0,42 (+0,05 [+0,02; +0,09]) |
| TL + color | 0,58 (+0,10 [−0,12; +0,29]) | 0,50 (**+0,13 [+0,09; +0,17]**) |
| TL + color + FruQ | 0,61 (+0,12 [−0,11; +0,30]) | 0,47 (**+0,11 [+0,08; +0,14]**) |

- **En las fuentes conocidas, empate:** todas las variantes quedan a ±0,01 de la CNN. El transfer
  learning reconoce mejor Defectuosa (recall 0,91–0,93 vs. 0,88).
- **Entre fuentes, mejora real pero insuficiente:** +0,11 a +0,13 al validar en Kaggle (IC sin
  cero); en HDF el IC es ancho (pocos grupos). Aun así el F1 en una fuente no vista sigue en
  0,5–0,6, y la media de las últimas épocas es mucho menor (0,16–0,18 en Kaggle): elegir la época con
  la propia fuente no vista es optimista para todos los modelos por igual.
- El **aumento de color** es lo que más aporta; **FruQ** no cambia el F1 pero sube el recall de
  Podrida en la fuente no vista (0,43 vs. 0,28 en HDF; 0,67 vs. 0,61 en Kaggle), justo la debilidad
  vista en las fotos propias.
- **Elegido: `tl_busqueda_4`** (color + FruQ) → `config.FINAL_TRANSFER`, fijado en el commit
  `8818bb0` antes de mirar test o las fotos propias. Pasa a ser el modelo de la app; la CNN desde
  cero sigue siendo el modelo exigido por el enunciado y se reporta igual.
- Nota de proceso: una primera corrida en Colab entrenó por error la CNN en lugar de EfficientNet
  (faltaba `"modelo": "transfer"` en la búsqueda); se corrigió y se descartó.

### 4.6 Tercera iteración: reentrenar desde cero con las etiquetas revisadas

Los dos modelos finales, con **la misma configuración** y entrenados desde cero con `index_v3.csv`
(§3.11) en Colab (T4): búsqueda `v3` de `src/models/transfer.py` (clave `"indice": "v3"`).
`v3_0` = CNN `cnn_estable_1`; `v3_1` = transfer `tl_busqueda_4`. No hubo selección: solo estas dos
configuraciones, fijadas antes de entrenar (commit `4bac7a8`).

| Modelo | Épocas (mejor) | F1 train | F1 val (etiquetas v3) | Media últimas 5 | Minutos |
|---|---|---:|---:|---:|---:|
| `v3_0` (CNN) | 40 (36) | 0,856 | 0,761 | 0,757 ± 0,003 | 3,7 |
| `v3_1` (transfer) | 18 (15) | 0,875 | 0,826 | 0,824 ± 0,002 | 13,5 |

El F1 de validación no es comparable con el de §4.2/§4.5 porque cambiaron las etiquetas. Comparación
justa sobre **las mismas 4.013 imágenes de val**, calificadas con cada juego de etiquetas
(`notebooks/05_evaluation` §6):

| Modelo | F1 con etiquetas originales | F1 con etiquetas v3 |
|---|---:|---:|
| CNN original (`cnn_estable_1`) | 0,910 | 0,800 |
| CNN v3 (`v3_0`) | 0,733 | 0,761 |
| Transfer original (`tl_busqueda_4`) | 0,912 | 0,802 |
| Transfer v3 (`v3_1`) | 0,753 | 0,825 |

Incluso calificada con sus propias etiquetas, la CNN v3 queda por debajo de la original: las
etiquetas revisadas no son más fáciles de aprender (coherente con κ = 0,29). El transfer v3 gana
+0,02 con las etiquetas v3 y pierde 0,16 con las originales.

---

## 5. Evaluación

Protocolo: los dos modelos de §4.3 (elegidos con validación, commit `d7b6928`) se evalúan **una
sola vez** en `test` (4.653 imágenes, 1.747 grupos) y en las 81 fotos propias
(`python -m src.evaluation.evaluate --split test|externa --final`). IC95 por bootstrap de grupos.
Detalle y figuras: `notebooks/05_evaluation`, `assets/evaluacion/`.

### 5.1 Resultados frente a los KPIs

| Modelo | Conjunto | F1-macro [IC95] | Exactitud | Recall Podrida [IC95] | F1 ≥ 0,85 | Recall P ≥ 0,90 |
|--------|----------|-----------------|----------:|-----------------------|:---------:|:---------------:|
| CNN | val | 0,91 [0,85; 0,95] | 0,94 | 0,94 [0,90; 0,97] | sí | sí |
| CNN | **test** | **0,80 [0,68; 0,93]** | 0,88 | **0,91 [0,86; 0,96]** | no | **sí** |
| CNN | **fotos propias** | **0,49 [0,37; 0,59]** | 0,49 | **0,27 [0,11; 0,44]** | no | no |
| SVM | val | 0,84 [0,73; 0,92] | 0,88 | 0,92 [0,87; 0,96] | no | sí |
| SVM | test | 0,80 [0,73; 0,87] | 0,90 | 0,81 [0,75; 0,88] | no | no |
| SVM | fotos propias | 0,46 [0,35; 0,56] | 0,47 | 0,57 [0,40; 0,74] | no | no |

Línea base de mayoría: F1 0,30 (test) y 0,17 (fotos propias). ROC-AUC de la CNN (uno contra el
resto): 0,96–0,97 en test, 0,59–0,81 en fotos propias.

### 5.2 Análisis

- **`test` (mismas fuentes, frutas nuevas):** 306 de los 369 errores Buena → Podrida de la CNN son
  **una sola sesión de HDF** (manzanas *Red Delicious* rojo oscuro, 386 fotos): el modelo asocia
  rojo muy oscuro con pudrición. Dando a cada grupo el mismo peso (cada fruta cuenta una vez; métrica
  decidida *después* de ver test, se reporta como diagnóstico), la CNN obtiene **F1 0,90** y recall
  de Podrida 0,92, igual que en validación; el SVM 0,85 y 0,83. El error más costoso
  (Podrida → Buena) es solo 3 % en la CNN.
- **Sobreajuste:** CNN train 0,91 ≈ val 0,91 (los pesos 1/grupo evitan memorizar sesiones
  repetidas); SVM train 1,00 ≫ val 0,84. La caída de la CNN no es sobreajuste sino **cambio de
  dominio**.
- **Fotos propias:** ambos modelos empujan casi todo a **Defectuosa**. La CNN clasifica 22 de 30
  Podrida como Defectuosa (11 de 14 manzanas con zonas marrones hundidas, 6 naranjas con moho, 3
  guayabas, 2 bananos) y 15 de 28 Buena como dañadas (10 son manzanas de una variedad rayada tipo Gala,
  rara en el entrenamiento). Los errores llegan con confianza > 90 %.
- **Causas:** (1) cambio de dominio — variedades, cámara y luz distintas, coherente con la prueba
  entre fuentes de §4.1 (F1 0,06–0,42); (2) **umbral de pudrición distinto**: las Podrida de los
  datasets públicos (sobre todo HDF) son más extremas que lo que nuestro criterio
  (`docs/criterios_anotacion.md`) llama Podrida.
- **Limitaciones:** 81 fotos, varias de la misma fruta (no anotado) → el IC95 real es más ancho; el
  tamaño no se evaluó (fotos sin escala).

### 5.3 Conclusión

Dentro de las fuentes conocidas la CNN desde cero cumple el KPI de recall de Podrida y, contando
cada fruta una vez, el de F1; **con fruta y cámara reales no es apta para uso** sin datos propios.
Para mejorarlo (requiere fotos nuevas, sin tocar las 81 actuales que son la prueba): fotografiar más
fruta propia y usar una parte para entrenar, alinear el criterio Defectuosa/Podrida en las
etiquetas de entrenamiento y añadir una salida "no sé" para fotos fuera de dominio.

### 5.4 Segunda mirada: transfer learning (§4.5)

Las fotos propias **ya se habían usado una vez** (§5.1); esta es una **segunda mirada**, con el
modelo fijado antes (commit `8818bb0`). Diferencias pareadas frente a la CNN con IC95 por grupos.

| Modelo | Conjunto | F1-macro [IC95] | Exactitud | Recall Podrida [IC95] | vs. CNN (F1) |
|--------|----------|-----------------|----------:|-----------------------|--------------|
| TL (`tl_busqueda_4`) | test | **0,85 [0,76; 0,94]** | 0,91 | 0,91 [0,88; 0,94] | +0,04 [−0,05; +0,11] |
| TL (`tl_busqueda_4`) | fotos propias | 0,47 [0,36; 0,57] | 0,47 | 0,30 [0,13; 0,47] | −0,02 [−0,16; +0,11] |

- **`test`:** los errores Buena → Podrida bajan de 369 a 72, casi todos de la carpeta de manzanas
  frescas de HDF (358 → 66): con el preentrenamiento el rojo oscuro se confunde mucho menos con
  pudrición. Ahora el error típico es Buena → Defectuosa (279). La mejora en F1 no alcanza a ser
  significativa.
- **Fotos propias: sin mejora.** Mismo patrón que la CNN: 20 de 30 Podrida → Defectuosa. La CNN y el
  TL coinciden en el 62 % de las fotos y **fallan juntos en el 35 %**: el problema no es la
  arquitectura sino los datos. Revisando las 21 Podrida mal clasificadas: 12 son **manzanas con
  manchas marrones moderadas** (fotos repetidas de pocas frutas), en la frontera entre Defectuosa y
  Podrida según nuestro criterio; 5 **naranjas con moho verde/blanco o tejido oscuro evidente** (4 →
  Defectuosa, 1 → Buena) sí son fallas claras; el resto son 2 bananos y 2 guayabas.
- **Conclusión:** el transfer learning mejora la generalización entre los datasets públicos y en
  test, pero no cierra la brecha con la fruta real. Con estos datos, la limitación es el **dominio y
  el umbral de pudrición** de las fuentes públicas, no la capacidad del modelo. Lo que sí ayudaría
  (fuera del alcance de tiempo): fotos propias para entrenar y ejemplos de moho en cítricos.

### 5.6 Tercera mirada: etiquetas revisadas (§3.11, §4.6)

Las fotos propias se usan por **tercera vez**, con los modelos fijados antes (commit `4bac7a8`). Solo
se evalúa en `externa`: `test` cambió de etiquetas en v3 y su cifra no sería comparable.

| Modelo | F1-macro [IC95] | Recall Podrida [IC95] | Podrida → Defectuosa |
|---|---|---|---:|
| CNN original (`cnn_estable_1`) | 0,49 [0,37; 0,59] | 0,27 [0,11; 0,44] | 22/30 |
| **CNN v3** (`v3_0`) | **0,38 [0,27; 0,49]** | 0,13 [0,03; 0,26] | 26/30 |
| Transfer original (`tl_busqueda_4`) | 0,47 [0,36; 0,57] | 0,30 [0,13; 0,47] | 20/30 |
| **Transfer v3** (`v3_1`) | **0,52 [0,40; 0,62]** | 0,27 [0,12; 0,43] | 21/30 |

Diferencias pareadas (original − v3, mismas 81 fotos, bootstrap por grupos): CNN **+0,11
[+0,003; +0,22]** en F1 (la v3 es peor, IC sin cero), +0,13 [−0,04; +0,32] en recall de Podrida;
transfer −0,045 [−0,15; +0,06] en F1 y +0,03 [−0,14; +0,21] en recall (sin diferencia demostrable).

**Conclusión: resultado negativo.** Revisar las etiquetas y quitar las imágenes con varias frutas
no mejoró la generalización a fruta real; la CNN empeoró y el recall de Podrida no subió. La causa
es la misma que se vio en §3.11: con el criterio escrito, la revisión hizo Podrida más exigente, y
el modelo aprendió con más fuerza que "podrido" es "muy podrido". Lo que deja la iteración es un
diagnóstico: **el cuello de botella es la definición operativa de Podrida** (κ = 0,29 entre
anotadores), no la arquitectura ni la cantidad de datos. Los modelos finales no cambian.

Para una cuarta iteración (fuera del tiempo disponible): consensuar la frontera con ejemplos
visuales anotados por todo el equipo, medir el κ antes de reetiquetar, y fotografiar fruta nueva
(otras frutas, no las 81 actuales) para entrenar y para una prueba limpia.

### 5.5 Tamaño

Validación de `src/features/size.py` (moneda de $1.000 como escala, corrección de perspectiva con la
focal del EXIF) con **11 frutas medidas con regla** (7 limones, 3 manzanas, 1 naranja; iPhone 15 Pro,
cámara a 20–29 cm). Resultados en `experiments/results/tamano_validar.csv`.

| Medida | Error medio absoluto | Sesgo | Clase de tamaño correcta |
|--------|---------------------:|------:|-------------------------:|
| Ancho máximo (círculo mínimo), 1.ª versión | 1,10 cm | +1,10 cm | 3/11 |
| **Diámetro ecuatorial** (lado corto del rectángulo mínimo), versión final | **0,57 cm** | +0,38 cm | 5/11 |

- **Escala verificada:** la moneda se encontró en las 11 fotos, y su escala coincide con la de la regla
  que aparece en cada foto (diferencia de 2–3 %). El error no viene de la moneda.
- **Cambio de medida tras ver los datos:** la 1.ª versión medía el ancho máximo, que incluye las puntas
  del limón y el lado del pedúnculo. Se cambió al **diámetro ecuatorial**, que es el calibre de las
  normas comerciales y lo que se mide con regla, y a un modelo exacto de esfera para la perspectiva.
  Como el cambio se decidió después de ver estas 11 fotos, el 0,57 cm es optimista.
- **Etiquetas ruidosas:** con una regla, el diámetro de una fruta redonda es difícil de medir. Hay pares
  incoherentes:
  - El limón 4 (3,8 cm) se ve más grande en la foto que el limón 7 (4,5 cm).
  - Las manzanas 1 y 3 se ven iguales y difieren 0,7 cm.
- **Clase:** 5/11 no alcanza el KPI (≥ 0,80). Los rangos de limón miden 1 cm de ancho, del mismo orden
  que el error (0,57 cm), y varias frutas están en el borde de un rango. Los umbrales son aproximados
  (calibres comerciales), no calibrados.
- **Limitaciones:** n = 11, solo 3 frutas distintas en tipo.

> _Pendiente: comparación con la literatura._

---

## 6. Despliegue

- Interfaz **Streamlit** (`src/app/app.py`): cargar/capturar imagen → predicción en tiempo real.
- Video de presentación (≤ 10 min).
- Opcional: línea de empaque con cámara + Raspberry Pi.

> _Por completar: capturas de la app, enlace al video._
