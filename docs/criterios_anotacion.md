# Criterios de anotación de calidad

Definición operativa de las 3 clases de calidad del proyecto. Se aplica igual a la re-anotación de
`Bad` (Kaggle), a la recolección propia y a cualquier imagen que se agregue.

| Clase | Definición | Señales visuales |
|-------|------------|------------------|
| **Buena** | Sin defectos visibles; apta para venta al precio normal. | Piel íntegra, color uniforme y propio de la fruta. |
| **Defectuosa** | Defecto **superficial**; aún consumible, pero de menor valor comercial. | Golpe o magulladura sin hundimiento profundo, manchas, sarna, raspones, cicatrices, sobremaduración leve (p. ej. pecas en el banano). |
| **Podrida** | Descomposición del tejido; no apta para consumo. | Zona blanda, hundida o muy oscura, moho, exudado/líquido, piel rota con pudrición, banano mayormente negro. |

**Regla de decisión:** ante la duda entre Defectuosa y Podrida, preguntarse *¿alguien la comería
quitando la parte afectada?* Si sí → Defectuosa; si no → Podrida. Si no se puede decidir, o la imagen
tiene varias frutas, está borrosa o la fruta no se reconoce → **Descartar**.

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
