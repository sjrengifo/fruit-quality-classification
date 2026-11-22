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
EXTERNAL_DIR = DATA_DIR / "external"
OWN_DIR = EXTERNAL_DIR / "propias"          # recolección propia (única carpeta de datos versionada)
OWN_IMAGES_DIR = OWN_DIR / "imagenes"
OWN_ANNOTATIONS = OWN_DIR / "anotaciones.csv"
HDF_DIR = EXTERNAL_DIR / "healthy_defective"  # dataset complementario (solo imágenes reales)
FRUQ_DIR = EXTERNAL_DIR / "fruq"            # FruQ (solo entrenamiento, ver FRUQ_* abajo)
ANNOTATIONS_DIR = DATA_DIR / "anotaciones"  # re-anotación de Bad (versionada)
DATASET_INDEX = PROCESSED_DIR / "index.csv"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

# ─── Definición del problema ──────────────────────────────
QUALITY_CLASSES = ["Buena", "Defectuosa", "Podrida"]      # 3 clases de calidad
SIZE_CLASSES = ["Pequeño", "Mediano", "Grande"]           # estimación de tamaño
# Mismos nombres que las carpetas del dataset base para poder fusionar ambas fuentes.
FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate", "Mango"]  # Mango: solo HDF

# ─── Recolección propia ───────────────────────────────────
OWN_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OWN_MAX_SIDE = 1024         # lado mayor (px) antes de subir al repo, para no inflarlo
OWN_ANNOTATION_COLUMNS = ["archivo", "fruta", "calidad", "tamano", "diametro_cm", "autor", "fecha", "notas"]

# ─── Estimación de tamaño (src/features/size.py) ─────────
SIZE_DIR = OWN_DIR / "tamano"
SIZE_IMAGES_DIR = SIZE_DIR / "imagenes"             # fruta + moneda, medida con regla
SIZE_ANNOTATIONS = SIZE_DIR / "anotaciones_tamano.csv"
SIZE_COINS_DIR = SIZE_DIR / "monedas"               # cada moneda junto a una regla (prueba de la medición)
# Diámetros oficiales (mm): Banco de la República, Banco Central Europeo, U.S. Mint
COIN_DIAMETERS_MM = {
    "100_nueva": 20.3, "200_nueva": 22.4, "500_nueva": 23.7, "1000_nueva": 26.7, "500_vieja": 23.5,
    "euro_1c": 16.25, "euro_10c": 19.75, "euro_20c": 22.25, "euro_50c": 24.25,
    "dime": 17.91, "nickel": 21.21, "quarter": 24.26,
}
DEFAULT_COIN = "1000_nueva"          # la más grande: la más fácil de detectar
SIZE_COIN_FILES = {                  # foto de data/external/propias/tamano/monedas/ → moneda
    "100pesos": "100_nueva", "200pesos": "200_nueva", "500pesos": "500_nueva", "1000pesos": "1000_nueva",
    "500centavoEuro": "500_vieja", "1centavoEuro": "euro_1c", "10centavoEuro": "euro_10c",
    "20centavoEuro": "euro_20c", "50centavoEuro": "euro_50c", "oneDimeDollar": "dime",
    "fiveCentsDollar": "nickel", "quarterDollar": "quarter",
}
# Umbrales (cm) del diámetro ecuatorial (calibre): < t1 Pequeño, < t2 Mediano, si no Grande.
# Aproximados a partir de calibres comerciales típicos; en banano y mango se usa el largo.
FRUIT_SIZE_CM = {
    "Apple": (6.5, 7.5), "Orange": (7.0, 8.5), "Lime": (4.5, 5.5), "Guava": (5.0, 7.0),
    "Pomegranate": (7.5, 9.5), "Mango": (10.0, 13.0), "Banana": (15.0, 20.0), "Otra": (5.0, 8.0),
}
SIZE_BY_LENGTH = ("Banana", "Mango")

# ─── Dataset base (Kaggle) ────────────────────────────────
KAGGLE_DATASET = "ryandpark/fruit-quality-classification"

# ─── Dataset complementario: Healthy-Defective-Fruits ────
# Chuquimarca et al. (ESPOL), CC BY 4.0. Se usan solo las imágenes reales, no las sintéticas.
HDF_REPO = "https://github.com/luischuquim/Healthy-Defective-Fruits.git"
HDF_REAL_DIRS = {"Apple": "apple_images/apple_real_images", "Mango": "mango_images/mango_real_images"}
# FruQ-multi (Abayomi-Alli y Damaševičius, Zenodo 10.5281/zenodo.7224690, CC BY 4.0): fotogramas de
# time-lapse; cada fruta es UN objeto que pasa de fresco a podrido. Se usan solo las frutas enteras y
# solas (sin partidas, racimos ni verduras), 1 de cada FRUQ_STEP fotogramas, y solo para entrenar.
FRUQ_URL = "https://zenodo.org/api/records/7224690/files/FruQ-multi.zip/content"
FRUQ_FRUITS = {"BananaDB": "Banana", "KakiQ": "Kaki", "PapayaQ": "Papaya", "PeachQ": "Peach", "PearQ": "Pear"}
FRUQ_LABELS = {"good": "Buena", "fresh": "Buena", "mild": "Defectuosa", "rotten": "Podrida"}
# Excluidos tras revisar los recortes: banano "mild" = pecas de maduración (en nuestro criterio es
# Buena); durazno "mild" = el video hace zoom y la fruta se sale del cuadro. La fresa se descartó
# entera: a 128 px "mild" y "rotten" se ven iguales.
FRUQ_SKIP = {("BananaDB", "mild"), ("PeachQ", "mild")}
FRUQ_STEP = 3
FRUQ_GROUP_FRAMES = 10              # fotogramas usados consecutivos = un grupo (mismo objeto, aspecto parecido)
TRAIN_ONLY_SOURCES = ("fruq",)      # split "extra": nunca en val/test; el entrenamiento la agrega a pedido
HDF_LABELS = {
    "fresh": "Buena",
    "bruise_defect": "Defectuosa",
    "scab_defect": "Defectuosa",
    "black_spot_defect": "Defectuosa",
    "rot_defect": "Podrida",
}

# ─── Re-anotación de Bad (Kaggle) en Defectuosa / Podrida ─
# Criterios en docs/criterios_anotacion.md.
RELABEL_OPTIONS = ["Defectuosa", "Podrida", "Buena", "Descartar"]
RELABEL_PER_FRUIT = 200     # muestra por fruta a re-anotar
AGREEMENT_PER_FRUIT = 10    # imágenes por fruta que anotan TODOS (acuerdo entre anotadores)

# ─── Hiperparámetros por defecto ──────────────────────────
IMAGE_SIZE = (128, 128)     # entrada de la CNN
RANDOM_SEED = 42            # reproducibilidad
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 40                 # máximo; se detiene antes si el F1 de validación no mejora

# ─── Modelos finales (elegidos solo con validación, antes de abrir test — docs/crisp-dm.md §4.3) ─
FINAL_MODEL = "cnn_estable_1"     # CNN desde cero: lote 128, coseno, sin aumento de datos
FINAL_CLASSIC = "svm"             # mejor modelo de ML tradicional, se reporta como comparación
# Segunda iteración (§4.5), elegido con validación y generalización entre fuentes, antes de la
# segunda mirada a test y fotos propias: EfficientNetB0 + aumento de color + FruQ. Es el modelo de la app.
FINAL_TRANSFER = "tl_busqueda_4"
