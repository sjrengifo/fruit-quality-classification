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

## 7. Pruebas

```bash
pytest
```
