# Guía de instalación y datos

## 1. Requisitos

- Python 3.11
- (Opcional) Conda / Miniconda
- Cuenta de Kaggle con token API (`kaggle.json`)

## 2. Entorno

### Opción A — pip + venv
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Opción B — conda
```bash
conda env create -f environment.yml
conda activate fruit-quality
```

## 3. Configurar Kaggle

1. En kaggle.com → Account → *Create New API Token* → descarga `kaggle.json`.
2. Colócalo en `~/.kaggle/kaggle.json` (Linux/Mac) o `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac).

> `kaggle.json` está en `.gitignore`: **no** lo subas al repositorio.

## 4. Descargar el dataset base

```bash
python -m src.data.download
```

Descarga [Fruit Quality Classification](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
en `data/raw/`. Los datos **no se versionan** (ver `.gitignore`).

## 5. Datos propios

Las imágenes recolectadas por el equipo **sí se versionan**, en `data/external/propias/imagenes/`,
con sus etiquetas en `data/external/propias/anotaciones.csv`. El protocolo de captura, la convención de
nombres y los comandos (`python -m src.data.own_images redimensionar|validar`) están en
[`../data/external/propias/README.md`](../data/external/propias/README.md).

## 6. Ejecutar la aplicación

```bash
streamlit run src/app/app.py
```

La app necesita los modelos entrenados en `experiments/checkpoints/` (no se versionan):
`cnn_estable_1.keras` (CNN final, `config.FINAL_MODEL`) y `svm.joblib` (comparación). Si faltan,
entrenarlos (`python -m src.training.train --model classic` y la búsqueda `cnn_stable` en Colab) o
pedírselos al equipo. La primera foto tarda ~2 s (carga U²-Net); las siguientes ~0,4 s en CPU.

Uso: elegir el modelo en la barra lateral, subir una o varias fotos (o usar la cámara) y leer la
calidad, las probabilidades y "lo que ve el modelo" (fruta recortada sobre fondo blanco). Para una
sola foto desde la terminal: `python -m src.models.predict ruta/foto.jpg [--model svm]`.

## 7. Pruebas

```bash
pytest
```

## Despliegue en Streamlit Community Cloud

La app (`src/app/app.py`) se despliega gratis en <https://share.streamlit.io>. Los modelos no están en
git: se publican como archivos de la release `modelos-v1` de GitHub y la app los descarga al arrancar
(`src/app/modelos.py`). La nube usa `src/app/requirements.txt`, no el de la raíz.

1. **Publicar los modelos** (solo si cambian; ya está hecho): `python -m src.app.modelos publicar`.
2. **Token** (el repositorio es privado): en GitHub → *Settings → Developer settings → Fine-grained
   tokens*, crear un token con acceso **solo a este repositorio** y permiso **Contents: Read-only**.
3. En share.streamlit.io → *Create app*: repositorio `sjrengifo/fruit-quality-classification`, la rama a
   desplegar (`main` después del merge) y archivo `src/app/app.py`. En *Advanced settings*: Python
   **3.12** y en *Secrets*:

   ```toml
   GITHUB_TOKEN = "github_pat_..."
   ```

4. *Deploy*. La primera vez tarda varios minutos (instala TensorFlow, descarga los modelos y U²-Net).
   La app se duerme si nadie la usa por unos días; al abrirla se despierta sola.

La cámara funciona en el despliegue (es HTTPS), también desde el celular.
