"""Estimación del tamaño de la fruta con una moneda como referencia de escala — Fases 5–6.

Ningún dataset trae tamaño y una foto sola no tiene escala (la misma fruta se ve grande de cerca y
pequeña de lejos), así que se pide una **moneda de diámetro conocido** al lado de la fruta:

1. Máscara de U²-Net (la misma de la calidad, `segment.fruit_mask`): la fruta es el componente más
   grande; la moneda, si U²-Net la incluye, se descarta.
2. Moneda: círculo de Hough fuera de la fruta; su borde se afina con rayos desde el centro (borde
   nítido más externo, sin la sombra) → escala px/mm sobre la mesa.
3. Medida: el **diámetro ecuatorial** (calibre comercial), que visto desde arriba es el lado corto
   del rectángulo mínimo del contorno —el largo incluiría las puntas del limón o el pedúnculo—;
   en banano y mango (`config.SIZE_BY_LENGTH`) se mide el largo.
4. **Corrección de perspectiva:** la moneda está sobre la mesa, pero el contorno de la fruta se ve
   desde más cerca y más grande (≈ 13 % con una fruta de 7 cm y la cámara a 27 cm). La altura de la
   cámara sale de la distancia focal (EXIF) y la escala de la moneda: H = f_px / (px/mm). Sin EXIF
   se asume `DEFAULT_CAMERA_HEIGHT_MM`.
5. Diámetro (cm) → Pequeño / Mediano / Grande con umbrales por fruta (`config.FRUIT_SIZE_CM`).

Uso:  python -m src.features.size preparar   (HEIC → JPEG ≤ 1024 px conservando el EXIF)
      python -m src.features.size monedas    (cada moneda junto a una regla: prueba de la medición)
      python -m src.features.size validar    (frutas medidas con regla: data/external/propias/tamano/)
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

import numpy as np

from src import config

WORK_SIDE = 1024                 # resolución de trabajo para medir (la máscara de U²-Net se amplía)
DEFAULT_CAMERA_HEIGHT_MM = 300   # sin EXIF (p. ej. cámara del navegador): se asume la foto a ~30 cm
_DIAG_35MM = 43.27               # diagonal del formato 35 mm (la focal equivalente se define con ella)


@dataclass
class SizeEstimate:
    diametro_cm: float | None        # diámetro ecuatorial (largo en SIZE_BY_LENGTH), corregido
    clase: str | None                # Pequeño / Mediano / Grande (None si no hay moneda)
    moneda: tuple[float, float, float] | None   # (x, y, radio) en px de la imagen de trabajo
    px_por_mm: float | None
    altura_camara_mm: float | None
    focal_exif: bool                 # False → altura supuesta (DEFAULT_CAMERA_HEIGHT_MM)
    mascara_fruta: np.ndarray        # máscara SEG_INPUT solo con la fruta (sin la moneda)
    aviso: str | None = None


def focal_px(source, diag_px: float) -> float | None:
    """Distancia focal en píxeles a partir de la focal equivalente a 35 mm del EXIF."""
    from PIL import Image

    try:
        img = source if isinstance(source, Image.Image) else Image.open(source)
        f35 = img.getexif().get_ifd(0x8769).get(0xA405)   # Exif IFD → FocalLengthIn35mmFilm
    except Exception:
        return None
    return float(f35) * diag_px / _DIAG_35MM if f35 else None


def _largest_component(mask: np.ndarray) -> np.ndarray:
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def find_coin(image: np.ndarray, fruit: np.ndarray | None = None) -> tuple[float, float, float] | None:
    """(x, y, radio) del círculo más fuerte que no esté sobre la fruta, o None."""
    import cv2

    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 5)
    side = min(gray.shape)
    fruit_r = np.sqrt(fruit.sum() / np.pi) if fruit is not None and fruit.any() else side
    for param2 in (50, 40, 30, 22):   # de más a menos exigente
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=side // 20, param1=100,
                                   param2=param2, minRadius=max(8, side // 100), maxRadius=side // 4)
        for x, y, r in (circles[0] if circles is not None else []):
            inside = fruit is not None and fruit[min(int(y), fruit.shape[0] - 1), min(int(x), fruit.shape[1] - 1)]
            if not inside and r < 0.8 * fruit_r:
                return float(x), float(y), float(r)
    return None


def coin_diameter_px(image: np.ndarray, x: float, y: float, r: float, n_rays: int = 180) -> float:
    """Diámetro de la moneda por rayos desde el centro de Hough: en cada rayo, el borde nítido más
    externo entre 0,8 r y 1,25 r (el borde de la sombra es difuso y queda por debajo del umbral; los
    bordes interiores —relieve, núcleo bimetálico— quedan más adentro). Mediana de los rayos."""
    import cv2

    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    radii = np.linspace(0.8 * r, 1.25 * r, int(0.45 * r * 4))           # paso de ~1/4 px
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    xs = (x + np.outer(np.cos(angles), radii)).astype(np.float32)
    ys = (y + np.outer(np.sin(angles), radii)).astype(np.float32)
    prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    grad = np.abs(np.gradient(prof, radii, axis=1))
    edges = []
    for g in grad:
        strong = np.nonzero(g >= 0.5 * g.max())[0]
        if g.max() > 0 and len(strong):
            edges.append(radii[strong[-1]])
    return 2 * float(np.median(edges)) if edges else 2 * r


def perspective_correct(length_mm: float, width_mm: float, height_mm: float) -> tuple[float, float]:
    """(largo, ancho) reales de una fruta apoyada en la mesa cuyo contorno se mide a la escala de la mesa.

    Ancho: la sección ecuatorial se modela como esfera de radio r con el centro a altura r; su contorno
    es el cono tangente desde la cámara, que corta la mesa a a = H·tan(asin(r/(H − r))). Despejando,
    con s = a/√(H² + a²): r = H·s/(1 + s). Largo: está a la altura del centro y se ve ampliado H/(H − r).
    """
    a = width_mm / 2
    s = a / np.hypot(height_mm, a)
    r = height_mm * s / (1 + s)
    return length_mm * (height_mm - r) / height_mm, 2 * r


def size_class(diameter_cm: float, fruit: str | None) -> str:
    from src.data.segment import diameter_to_size_class

    return diameter_to_size_class(diameter_cm, config.FRUIT_SIZE_CM.get(fruit, config.FRUIT_SIZE_CM["Otra"]))


def estimate_size(source, mask: np.ndarray, fruit: str | None = None,
                  coin: str = config.DEFAULT_COIN) -> SizeEstimate:
    """`mask`: máscara de U²-Net a SEG_INPUT (`segment.fruit_mask`) de la misma imagen."""
    import cv2
    from PIL import Image

    from src.data.preprocess import prepare_image

    fruit_small = _largest_component(mask)
    image = prepare_image(source, (WORK_SIDE, WORK_SIDE))   # mismo encuadre que SEG_INPUT, más resolución
    big = cv2.resize(fruit_small.astype(np.uint8), (WORK_SIDE, WORK_SIDE), interpolation=cv2.INTER_NEAREST) > 0
    found = find_coin(image, cv2.dilate(big.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    if found is None:
        return SizeEstimate(None, None, None, None, None, False, fruit_small,
                            "No se encontró la moneda: el tamaño necesita una moneda al lado de la fruta.")
    x, y, r = found
    px_mm = coin_diameter_px(image, x, y, r) / config.COIN_DIAMETERS_MM[coin]

    # escala de la foto original (la imagen de trabajo está reducida y con relleno)
    if hasattr(source, "seek"):
        source.seek(0)
    img = source if isinstance(source, Image.Image) else Image.open(source)
    w0, h0 = img.size
    scale = WORK_SIDE / max(w0, h0)
    f = focal_px(img, np.hypot(w0, h0))
    height = f * scale / px_mm if f else DEFAULT_CAMERA_HEIGHT_MM

    contours, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    width, length = sorted(cv2.minAreaRect(c)[1])          # px a la escala de la mesa
    length, width = perspective_correct(length / px_mm, width / px_mm, height)
    diameter = (length if fruit in config.SIZE_BY_LENGTH else width) / 10
    aviso = None if f else (f"La foto no trae la distancia focal (EXIF): se asume la cámara a "
                            f"{DEFAULT_CAMERA_HEIGHT_MM / 10:.0f} cm de la mesa.")
    return SizeEstimate(round(diameter, 2), size_class(diameter, fruit), found, px_mm, height, bool(f),
                        fruit_small, aviso)


def measure(path) -> SizeEstimate:
    """Atajo: segmenta y mide una foto (sin la fruta declarada)."""
    from src.data.preprocess import prepare_image
    from src.data.segment import SEG_INPUT, fruit_mask

    return estimate_size(path, fruit_mask(prepare_image(path, SEG_INPUT)))


def validate_coins() -> list[dict]:
    """Monedas fotografiadas junto a una regla: diámetro medido con la escala de la regla (marcas de
    1 mm, periodo por FFT) frente al oficial. Prueba la detección y la medición de la moneda."""
    import cv2

    from src.data.preprocess import prepare_image

    out = []
    for path in sorted(config.SIZE_COINS_DIR.glob("*.jpg")):
        image = prepare_image(path, (WORK_SIDE, WORK_SIDE))
        found = find_coin(image)
        if found is None:
            out.append({"moneda": path.stem, "detectada": False})
            continue
        x, y, r = found
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        best, period = 0.0, None
        for y0 in range(int(y + r), gray.shape[0] - 8, 4):    # la regla está debajo de la moneda
            prof = gray[y0:y0 + 8].mean(0)
            spec = np.abs(np.fft.rfft((prof - prof.mean()) * np.hanning(len(prof))))
            freq = np.fft.rfftfreq(len(prof))
            i = int(np.argmax(spec * ((freq > 1 / 40) & (freq < 1 / 6))))
            if spec[i] > best:
                a, b, c = spec[i - 1], spec[i], spec[i + 1]
                best, period = spec[i], 1 / ((i + 0.5 * (a - c) / (a - 2 * b + c)) * freq[1])
        mm = coin_diameter_px(image, x, y, r) / period
        official = config.COIN_DIAMETERS_MM[config.SIZE_COIN_FILES[path.stem]]
        out.append({"moneda": path.stem, "detectada": True, "medido_mm": round(mm, 2), "oficial_mm": official,
                    "razon": round(mm / official, 4)})
    return out


def validate_fruits() -> list[dict]:
    """Frutas con moneda medidas con regla (data/external/propias/tamano/anotaciones_tamano.csv)."""
    with open(config.SIZE_ANNOTATIONS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        from src.data.preprocess import prepare_image
        from src.data.segment import SEG_INPUT, fruit_mask

        path = config.SIZE_IMAGES_DIR / r["archivo"]
        est = estimate_size(path, fruit_mask(prepare_image(path, SEG_INPUT)), r["fruta"], r["moneda"])
        real = float(r["diametro_fruta_cm"])
        out.append({"archivo": r["archivo"], "fruta": r["fruta"], "real_cm": real, "estimado_cm": est.diametro_cm,
                    "error_cm": None if est.diametro_cm is None else round(est.diametro_cm - real, 2),
                    "clase_real": size_class(real, r["fruta"]), "clase_estimada": est.clase,
                    "altura_camara_cm": None if est.altura_camara_mm is None else round(est.altura_camara_mm / 10, 1),
                    "exif": est.focal_exif})
    return out


def prepare_photos() -> int:
    """Fotos del iPhone (HEIC) → JPEG y todas a ≤ OWN_MAX_SIDE px, **conservando el EXIF** (la focal
    hace falta para la corrección de perspectiva). HEIC se convierte con `sips` (macOS)."""
    import subprocess

    from src.data.own_images import resize_images

    for heic in [*config.SIZE_IMAGES_DIR.glob("*.HEIC"), *config.SIZE_IMAGES_DIR.glob("*.heic")]:
        subprocess.run(["sips", "-s", "format", "jpeg", str(heic), "--out", str(heic.with_suffix(".jpg"))],
                       check=True, capture_output=True)
        heic.unlink()
    return resize_images(config.SIZE_IMAGES_DIR)


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Validación de la estimación de tamaño")
    parser.add_argument("que", choices=["preparar", "monedas", "validar"])
    args = parser.parse_args()
    if args.que == "preparar":
        print(f"{prepare_photos()} foto(s) reducidas en {config.SIZE_IMAGES_DIR}")
        return
    if args.que == "monedas":
        df = pd.DataFrame(validate_coins())
        print(df.to_string(index=False))
        print(f"razón medido/oficial: media {df['razon'].mean():.3f} · desviación {df['razon'].std():.3f}")
    else:
        df = pd.DataFrame(validate_fruits())
        print(df.to_string(index=False))
        ok = df["estimado_cm"].notna()
        print(f"moneda encontrada en {ok.sum()}/{len(df)} · MAE {df.loc[ok, 'error_cm'].abs().mean():.2f} cm · "
              f"sesgo {df.loc[ok, 'error_cm'].mean():+.2f} cm · "
              f"clase correcta {(df.loc[ok, 'clase_real'] == df.loc[ok, 'clase_estimada']).mean():.0%}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / f"tamano_{args.que}.csv", index=False)


if __name__ == "__main__":
    main()
