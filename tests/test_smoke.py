"""Pruebas de humo: verifican configuración y utilidades básicas sin dependencias pesadas."""
from src import config
from src.data.segment import diameter_to_size_class


def test_quality_classes_min_three():
    """El enunciado exige al menos 3 clases de calidad."""
    assert len(config.QUALITY_CLASSES) >= 3


def test_size_classes():
    assert config.SIZE_CLASSES == ["Pequeño", "Mediano", "Grande"]


def test_diameter_to_size_class():
    assert diameter_to_size_class(0.1) == "Pequeño"
    assert diameter_to_size_class(0.5) == "Mediano"
    assert diameter_to_size_class(0.9) == "Grande"


def test_count_objects():
    import numpy as np

    from src.data.revision import count_objects

    mask = np.zeros((40, 40), dtype=bool)
    mask[2:12, 2:12] = True                       # fruta 1
    assert count_objects(mask) == 1
    mask[20:30, 20:30] = True                     # fruta 2, mismo tamaño
    mask[35, 35] = True                           # ruido de la máscara: no cuenta
    assert count_objects(mask) == 2
    assert count_objects(np.zeros((5, 5), dtype=bool)) == 0
