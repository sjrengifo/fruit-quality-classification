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

Coloca las 30–50 imágenes recolectadas y anotadas por el equipo en `data/external/` siguiendo la
convención de carpetas por clase de calidad. Ver [`../ROADMAP.md`](../ROADMAP.md), Fase 3.

## 6. Ejecutar la aplicación

```bash
streamlit run src/app/app.py
```

## 7. Pruebas

```bash
pytest
```
