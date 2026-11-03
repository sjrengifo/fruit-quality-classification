# Recolección propia de imágenes

Imágenes tomadas por el equipo (meta: **30–50**). Se usan como **conjunto de prueba externo**:
fotos reales, en condiciones propias, que el modelo nunca ve al entrenar. Miden si generaliza.
A diferencia del resto de `data/`, **esta carpeta sí se versiona en git**.

## Protocolo de captura

| Aspecto | Criterio |
|---------|----------|
| Fondo | **Blanco** liso (cartulina, hoja o mesa blanca), sin texturas ni objetos |
| Iluminación | Buena luz, idealmente natural difusa; evitar sombras duras, reflejos y flash directo |
| Encuadre | **Una sola fruta** por foto, centrada y completa, vista cenital (desde arriba) |
| Distancia | Fija (~30 cm) para que el tamaño en píxeles sea comparable entre fotos |
| Frutas | Las del dataset: `Apple`, `Banana`, `Guava`, `Lime`, `Orange`, `Pomegranate`, `Mango` |
| Clases | Repartir parejo entre `Buena`, `Defectuosa` y `Podrida` (criterios en `docs/criterios_anotacion.md`) |
| Privacidad | Sin personas, rostros, marcas ni datos de establecimientos (ver `docs/etica.md`) |

Si es posible, medir el diámetro de la fruta con una regla y anotarlo en `diametro_cm`; sirve para
validar la estimación de tamaño de la Fase 3.

## Cómo agregar imágenes

1. Copiar las fotos a `imagenes/` con el nombre `<Fruta>_<Calidad>_<NNN>.jpg`,
   p. ej. `Apple_Defectuosa_001.jpg`.
2. Reducirlas para no inflar el repo (lado mayor ≤ 1024 px, corrige la rotación del celular):
   ```bash
   python -m src.data.own_images redimensionar
   ```
3. Agregar una fila por imagen en `anotaciones.csv`:

   | Columna | Valores |
   |---------|---------|
   | `archivo` | nombre del archivo en `imagenes/` |
   | `fruta` | `Apple` · `Banana` · `Guava` · `Lime` · `Orange` · `Pomegranate` · `Mango` |
   | `calidad` | `Buena` · `Defectuosa` · `Podrida` |
   | `tamano` | `Pequeño` · `Mediano` · `Grande` |
   | `diametro_cm` | número (opcional) |
   | `autor` | quién tomó la foto |
   | `fecha` | `AAAA-MM-DD` |
   | `notas` | opcional (tipo de defecto, condiciones, etc.) |

4. Validar antes de hacer commit:
   ```bash
   python -m src.data.own_images validar
   ```
5. Commit en una rama `feature/*` y PR hacia `dev`.

Para compartir con el resto de la clase basta con enlazar esta carpeta del repositorio.
