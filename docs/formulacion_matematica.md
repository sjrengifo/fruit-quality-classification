# Formulación matemática del problema (PI3)

Cubre la competencia **PI3** (solución de problemas complejos mediante fundamentos matemáticos),
evaluada en los Criterios 3, 4 y 5 de la rúbrica.

---

## 1. Formulación del problema

Sea una imagen de entrada $x \in \mathbb{R}^{H \times W \times 3}$ (fruta individual sobre fondo simple).
Se buscan dos funciones de decisión:

- **Clasificación de calidad:** $f_\theta : x \mapsto \hat{y} \in \{1,\dots,K\}$, con $K = 3$ clases
  $\{\text{Buena}, \text{Defectuosa}, \text{Podrida}\}$.
- **Estimación de tamaño:** $g : x \mapsto \hat{s} \in \{\text{Pequeño}, \text{Mediano}, \text{Grande}\}$,
  derivada del diámetro en píxeles normalizados.

Para el enfoque de ML tradicional se extrae un vector de características
$\phi(x) \in \mathbb{R}^d$ (color, textura, forma) y se aprende $f_\theta(\phi(x))$.

> **Origen de las etiquetas de calidad (opción A).** El dataset base aporta 2 niveles
> ($\text{Good}, \text{Bad}$); se mapea $\text{Good}\!\to\!\text{Buena}$, $\text{Bad}\!\to\!\text{Podrida}$
> y la clase intermedia $\text{Defectuosa}$ se obtiene por anotación propia (recolección + `Mixed Quality`).
> El problema resultante es de **clasificación multiclase con $K=3$**.

**Por qué es un problema complejo de ingeniería:** alta dimensionalidad de la entrada, variabilidad
(iluminación, fondo, oclusión, variedad de fruta), clases desbalanceadas, fronteras de decisión no
lineales y necesidad de generalización a datos nuevos e inferencia en tiempo real.

---

## 2. Modelos y funciones de pérdida

**Clasificación multiclase (probabilidades):** softmax
$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad k = 1,\dots,K.
$$

**Entropía cruzada categórica** (pérdida a minimizar):
$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\,\log p_{ik} \; + \; \lambda\,\lVert\theta\rVert_2^2,
$$
donde $y_{ik}$ es la codificación one-hot y $\lambda\lVert\theta\rVert_2^2$ el término de regularización.

**SVM (margen máximo):**
$$
\min_{w,b}\ \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i \quad \text{s.a.}\quad y_i(w^\top \phi(x_i)+b) \ge 1-\xi_i,\ \xi_i\ge 0.
$$

**Regresión logística / entropía cruzada, Random Forest (impureza de Gini), XGBoost (gradient
boosting)** se documentarán con su formulación al seleccionarlos.

---

## 3. Fundamentos matemáticos aplicados

- **Normalización de píxeles:** $x' = x/255$ o estandarización $x' = (x-\mu)/\sigma$ por canal.
- **Estimación de tamaño:** diámetro $D$ (px) del contorno segmentado, normalizado por una
  referencia $R$: $d = D/R$; discretización en bins por umbrales $\{t_1, t_2\}$.
- **Métricas:**
  $$
  \text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN},\quad
  F1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
  $$
  F1-macro promedia el F1 por clase (robusto ante desbalanceo).
- **Optimización:** descenso de gradiente (estocástico / Adam) para la CNN.
- **Validación cruzada k-fold** para estimar el error de generalización y comparar hiperparámetros.

---

## 4. Estrategia de solución (justificación)

Se comparan **≥ 2 modelos de ML tradicional + 1 CNN** para contrastar enfoques basados en
características diseñadas a mano vs. aprendizaje de representaciones. Alternativas y por qué se
descartan (o no) se argumentarán en el informe (ver Criterio 4).

**Criterio de éxito cuantitativo (definido en la Fase 1).** El modelo debe alcanzar
$F1\text{-macro} \ge 0.85$ en el conjunto de prueba, con $\text{Recall}(\text{Podrida}) \ge 0.90$
(minimizar el falso negativo más costoso: dejar pasar fruta podrida), superando de forma
significativa la línea base (clasificador de mayoría). Ver KPIs en [`crisp-dm.md`](crisp-dm.md) §1.4.

> _Por completar con hiperparámetros óptimos, resultados cuantitativos e interpretación
> (precisión, error, generalización, sobreajuste) en la fase de evaluación._
