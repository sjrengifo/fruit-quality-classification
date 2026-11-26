# Criterios de anotación de calidad

Definición operativa de las 3 clases de calidad del proyecto. Se aplica igual a la re-anotación de
`Bad` (Kaggle), a la recolección propia y a cualquier imagen que se agregue.

| Clase | Definición | Señales visuales |
|-------|------------|------------------|
| **Buena** | Sin daño visible; apta para venta al precio normal. | Piel íntegra, color propio de la fruta. **El grado de madurez no es un defecto**: un banano verde (biche) sin daño es Buena, y un banano amarillo **con pecas** de maduración también (regla de nov. 2026, ver abajo). |
| **Defectuosa** | Daño **superficial y localizado**: se quita la parte afectada y el resto se come sin problema. Menor valor comercial. | Piel firme; mancha pequeña o clara, raspón, cicatriz, sarna, golpe leve; banano con la cáscara **muy** manchada o con golpes. |
| **Podrida** | Descomposición del tejido; no apta para consumo. **No exige moho**: basta **cualquiera** de las señales. | Moho; zona **blanda o hundida** al tacto; mancha muy oscura que llega a la pulpa o cubre una parte grande (≈ ¼ de la fruta o más); piel rota con pudrición o líquido; banano mayormente negro. |

**Regla de decisión:** ante la duda entre Defectuosa y Podrida, preguntarse *¿alguien la comería
quitando la parte afectada?* Si sí → Defectuosa. Si la parte dañada es grande, está blanda o ya
afecta la pulpa (aunque no haya moho) → Podrida. Si no se puede decidir, o la imagen tiene varias
frutas, está borrosa o la fruta no se reconoce → **Descartar**.

> Con fotos propias se puede tocar la fruta: si se etiquetó Podrida por estar blanda y eso no se
> nota en la foto, anotarlo en `notas` ("blanda al tacto"). Son casos difíciles para el modelo, que
> solo ve la imagen, y sirven para el análisis de errores.
>
> *Historial:* la versión inicial (usada en la re-anotación de `Bad`) tenía la misma regla; esta
> versión (nov. 2026) solo precisa que Podrida no exige moho y que la madurez no es un defecto.

## Re-anotación de `Bad` (Kaggle)

El dataset base solo trae `Good`/`Bad`, y `Bad` mezcla defectos leves con podredumbre
(ver `assets/eda/muestras_bad.png`). Por eso se re-anota a mano una muestra de `Bad`:

- **Muestra:** 200 imágenes aleatorias por fruta (semilla 42) → 1.200 en total.
- **Acuerdo entre anotadores:** las primeras 10 de cada fruta (60 en total) las anotan **todos** los
  integrantes. Se reporta el kappa de Cohen por pareja; la etiqueta final es el voto mayoritario
  (empate → Descartar). Un kappa < 0,6 indica que hay que discutir y afinar estos criterios.
- **Reparto sugerido:** 2 frutas por integrante, para que nadie anote la misma imagen (salvo las de
  acuerdo).

```bash
git pull                              # traer las anotaciones de los demás
streamlit run src/app/anotar.py       # anotar
python -m src.data.relabel            # avance por fruta y kappa
git add data/anotaciones && git commit -m "Anotaciones de <nombre>" && git push
```

Cada integrante escribe en su propio archivo `data/anotaciones/bad_<nombre>.csv`, así no hay
conflictos de merge.

## Mapeo de Healthy-Defective-Fruits

Dataset complementario (Chuquimarca et al., ESPOL; CC BY 4.0). Solo imágenes **reales**:

| Carpeta original | Clase del proyecto |
|------------------|--------------------|
| `fresh` | Buena |
| `bruise_defect`, `scab_defect`, `black_spot_defect` | Defectuosa |
| `rot_defect` | Podrida |

## Revisión de etiquetas (tercera iteración, nov. 2026)

Revisión completa de Defectuosa/Podrida con estos mismos criterios, para reentrenar modelos desde
cero (`docs/crisp-dm.md` §3.11). No reemplaza la re-anotación anterior: produce un índice aparte
(`data/processed/index_v3.csv`).

- **Quién:** las etiquetas las asignó **un modelo de IA (Claude)** a pedido del equipo, por falta de
  tiempo para revisarlas a mano. Debe declararse así en el informe.
- **Unidad:** el grupo (misma fruta), a ciegas: hojas de contacto numeradas sin etiqueta ni fuente
  (`python -m src.data.revision hojas`). Los grupos de 40 o más imágenes se verificaron con 24
  imágenes cada uno.
- **Archivo:** `data/anotaciones/revision_v3.txt`, una línea por hoja: `<hoja> <letras>`, una letra
  por celda (B = Buena, D = Defectuosa, P = Podrida, X = Descartar). Se marcó Descartar además
  cuando había varias frutas que se tocan (el detector automático no las separa), foto en blanco y
  negro, borrosa o fruta tapada.
- **Acuerdo con la etiqueta anterior:** κ = 0,29 en Kaggle (con la re-anotación de Samuel), 0,40 en
  HDF, 0,86 en FruQ. En Kaggle está por debajo del 0,6 que este documento fija como umbral: **la
  frontera Defectuosa/Podrida necesita consenso del equipo con ejemplos** antes de volver a anotar.
- **Resultado:** reentrenar con estas etiquetas no mejoró el desempeño con las fotos propias
  (`docs/crisp-dm.md` §5.6).

```bash
python -m src.data.revision objetos   # varias frutas por máscara
python -m src.data.revision hojas     # hojas de contacto (data/interim/revision/hojas/)
python -m src.data.revision indice    # → data/processed/index_v3.csv
python -m src.data.revision resumen   # acuerdo por fuente
```

## Regla del banano y alcance de 3 frutas (cuarta iteración, 24 nov. 2026)

- **Banano:** las pecas de maduración **no son defecto** (antes eran Defectuosa). Defectuosa solo si
  la cáscara está muy manchada o con golpes; Podrida si hay zonas negras grandes o blandas, o si está
  mayormente negro. Al aplicarla a las hojas de banano de la revisión v3 solo cambian 3 grupos de
  FruQ (`data/anotaciones/revision_v4.txt`); los bananos de Kaggle `Bad` están muy manchados o
  negros y no cambian.
- **Alcance:** la cuarta iteración entrena y evalúa solo **manzana, banano y naranja**
  (`config.V4_FRUITS`), frutas en que el daño contrasta con el color de la piel. Es una decisión de
  alcance del producto, no una mejora del modelo: se reporta así.
