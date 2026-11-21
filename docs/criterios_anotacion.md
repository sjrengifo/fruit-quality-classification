# Criterios de anotación de calidad

Definición operativa de las 3 clases de calidad del proyecto. Se aplica igual a la re-anotación de
`Bad` (Kaggle), a la recolección propia y a cualquier imagen que se agregue.

| Clase | Definición | Señales visuales |
|-------|------------|------------------|
| **Buena** | Sin daño visible; apta para venta al precio normal. | Piel íntegra, color propio de la fruta. **El grado de madurez no es un defecto**: un banano verde (biche) sin daño es Buena. |
| **Defectuosa** | Daño **superficial y localizado**: se quita la parte afectada y el resto se come sin problema. Menor valor comercial. | Piel firme; mancha pequeña o clara, raspón, cicatriz, sarna, golpe leve, sobremaduración leve (p. ej. pecas en el banano). |
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
