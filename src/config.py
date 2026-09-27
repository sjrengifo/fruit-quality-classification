"""Configuración central del proyecto: rutas, clases e hiperparámetros por defecto.

Centralizar aquí evita rutas y "números mágicos" dispersos por el código.
"""
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"  # imágenes propias del equipo
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# ─── Definición del problema ──────────────────────────────
QUALITY_CLASSES = ["Buena", "Defectuosa", "Podrida"]      # 3 clases de calidad
SIZE_CLASSES = ["Pequeño", "Mediano", "Grande"]           # estimación de tamaño

# ─── Dataset base (Kaggle) ────────────────────────────────
KAGGLE_DATASET = "ryandpark/fruit-quality-classification"

# ─── Hiperparámetros por defecto ──────────────────────────
IMAGE_SIZE = (128, 128)     # entrada de la CNN
RANDOM_SEED = 42            # reproducibilidad
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 30
