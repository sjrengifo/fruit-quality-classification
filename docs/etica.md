# Análisis ético e impactos

Cubre las competencias **PI1** (dilemas éticos y código de ética) y **PI2** (impactos globales,
sociales, ambientales y económicos), evaluadas en los Criterios 1 y 2 de la rúbrica.

---

## 1. Dilemas éticos y relación con el código de ética (PI1)

| # | Situación / decisión del proyecto | Dilema ético | Principio del código de ética | Medida adoptada |
|---|-----------------------------------|--------------|-------------------------------|-----------------|
| 1 | Recolección de imágenes en mercados/hogares | Privacidad; captura accidental de personas o datos | Respeto a la privacidad; consentimiento informado | Fotografiar solo la fruta sobre fondo simple; evitar personas/rostros; pedir permiso en establecimientos |
| 2 | Uso y publicación del dataset | Uso responsable de la información; propiedad de datos de terceros | Honestidad; atribución (evitar fraude) | Referenciar explícitamente el dataset de Kaggle y todo código de terceros |
| 3 | Sesgos del modelo | Sesgo por tipo de fruta, iluminación, origen; decisiones injustas | Responsabilidad social; equidad | Analizar desempeño por clase; balancear datos; documentar limitaciones |
| 4 | Impacto laboral de la automatización | Desplazamiento de trabajadores de clasificación manual | Bienestar público; responsabilidad social | Presentar el sistema como apoyo (asistencia), no reemplazo total; transparencia |
| 5 | Errores de clasificación | Falsos positivos/negativos con consecuencias económicas | Seguridad y bienestar; diligencia profesional | Reportar incertidumbre; mantener supervisión humana en casos límite |

> Completar/ajustar durante el desarrollo con las decisiones reales tomadas.

---

## 2. Matriz de impactos (PI2)

| Dimensión | Impactos potenciales | Riesgos | Estrategias de mitigación |
|-----------|----------------------|---------|---------------------------|
| **Social** | Reducción de desperdicio; estandarización; menos trabajo tedioso | Percepción de vigilancia; desplazamiento laboral | Uso asistivo; recapacitación; transparencia en el uso de datos |
| **Económica** | Menores pérdidas; mayor eficiencia en la cadena de suministro | Costos de hardware/implementación; dependencia tecnológica | Modelos ligeros; despliegue en hardware accesible (p. ej. Raspberry Pi) |
| **Ambiental** | Menos desperdicio de alimentos | Consumo energético del procesamiento en tiempo real; e-waste | Priorizar CNN pequeñas/eficientes; inferencia optimizada; hardware de bajo consumo |
| **Global** | Aplicable en distintos mercados y cultivos | Diferencias culturales/regulatorias; datos poco representativos | Datasets diversos; adaptación local; cumplimiento normativo por región |

> Se recomienda acompañar cada celda con argumentos técnicos y éticos en el informe final.

---

## 3. Medidas generales de comportamiento ético

- Atribución explícita de datos y código de terceros (formato IEEE).
- Reproducibilidad (semillas fijas, entornos documentados).
- Documentación de limitaciones y sesgos conocidos.
- Supervisión humana en decisiones con impacto económico.
