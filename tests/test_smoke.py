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
