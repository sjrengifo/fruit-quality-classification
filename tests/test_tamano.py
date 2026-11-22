"""Estimación de tamaño (src/features/size.py) con imágenes sintéticas de medidas conocidas."""
import numpy as np
import pytest

from src import config

cv2 = pytest.importorskip("cv2")


def _scene(coin_r=40, fruit_r=150):
    """Fondo claro, 'fruta' naranja grande y 'moneda' gris con sombra difusa y relieve interior."""
    img = np.full((1024, 1024, 3), 235, np.uint8)
    cv2.circle(img, (400, 500), fruit_r, (230, 120, 20), -1)
    shadow = np.zeros_like(img)
    cv2.circle(shadow, (795, 505), coin_r + 4, (40, 40, 40), -1)
    img = cv2.subtract(img, cv2.GaussianBlur(shadow, (21, 21), 0))
    cv2.circle(img, (790, 500), coin_r, (150, 150, 140), -1)
    cv2.circle(img, (790, 500), int(coin_r * 0.6), (200, 190, 120), 2)   # núcleo / relieve
    return img


def test_moneda_se_encuentra_fuera_de_la_fruta_y_se_mide_sin_sombra():
    from src.features.size import coin_diameter_px, find_coin

    img = _scene()
    fruit = np.zeros(img.shape[:2], bool)
    cv2.circle(fruit.view(np.uint8), (400, 500), 160, 1, -1)
    x, y, r = find_coin(img, fruit)
    assert abs(x - 790) < 4 and abs(y - 500) < 4
    assert abs(coin_diameter_px(img, x, y, r) - 80) < 2.5     # ±3 %


def test_correccion_de_perspectiva():
    import math

    from src.features.size import perspective_correct

    # cámara muy lejos: sin cambio
    assert perspective_correct(90, 70, 1e9) == pytest.approx((90, 70), rel=1e-6)
    # esfera de 70 mm con la cámara a 300 mm: su contorno corta la mesa en a = H·tan(asin(r/(H − r)))
    seen = 2 * 300 * math.tan(math.asin(35 / 265))
    assert perspective_correct(seen, seen, 300)[1] == pytest.approx(70, rel=1e-6)
    # el largo, a la altura del centro (35 mm), se ve ampliado H/(H − 35)
    assert perspective_correct(100 * 300 / 265, seen, 300)[0] == pytest.approx(100, rel=1e-6)
    # objeto plano (grosor 0): sin cambio
    assert perspective_correct(50, 0, 300) == pytest.approx((50, 0))


def test_clases_de_tamano_por_fruta():
    from src.features.size import size_class

    lo, hi = config.FRUIT_SIZE_CM["Apple"]
    assert size_class(lo - 0.1, "Apple") == "Pequeño"
    assert size_class(hi - 0.1, "Apple") == "Mediano"
    assert size_class(hi + 0.1, "Apple") == "Grande"
    assert size_class(6.0, "Desconocida") == size_class(6.0, "Otra")
    assert set(config.SIZE_COIN_FILES.values()) <= set(config.COIN_DIAMETERS_MM)
