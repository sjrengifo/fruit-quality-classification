"""Pruebas del preprocesamiento (preprocess.py) y de los splits agrupados y estables (split.py)."""
import numpy as np
from PIL import Image

from src import config
from src.data.preprocess import normalize, prepare_image
from src.data.split import assign_split, group_key


def test_prepare_image_is_square_and_rgb(tmp_path):
    path = tmp_path / "rect.png"
    Image.new("L", (300, 150), color=200).save(path)  # gris, no cuadrada
    out = prepare_image(path)
    assert out.shape == (*config.IMAGE_SIZE, 3) and out.dtype == np.uint8
    assert normalize(out).max() <= 1.0


def test_prepare_image_same_result_for_png_and_jpeg(tmp_path):
    """El formato de origen no debe cambiar el resultado más allá de la compresión."""
    img = Image.new("RGB", (256, 256), color=(180, 40, 40))
    img.save(tmp_path / "a.png")
    img.save(tmp_path / "a.jpg", quality=95)
    diff = np.abs(prepare_image(tmp_path / "a.png").astype(int) - prepare_image(tmp_path / "a.jpg").astype(int))
    assert diff.mean() < 2


def test_group_key_joins_augmented_copies_and_variants():
    base = "raw/Good Quality_Fruits/Pomegranate_Good/"
    keys = {group_key(base + n, "kaggle") for n in
            ["20190820_143522.jpg", "20190820_143522_22896.jpg", "20190820_143522_22897.jpg"]}
    assert len(keys) == 1
    guava = "raw/Good Quality_Fruits/Guava_Good/"
    assert group_key(guava + "IMG20200729172820.jpg", "kaggle") == group_key(guava + "IMG20200729172820_01.jpg", "kaggle")
    # mismo nombre en otra carpeta → grupo distinto
    assert group_key("raw/a/IMG_1234.JPG", "kaggle") != group_key("raw/b/IMG_1234.JPG", "kaggle")


def test_assign_split_is_stable_and_proportional():
    groups = [f"g{i}" for i in range(5000)]
    first = [assign_split(g, "kaggle") for g in groups]
    assert first == [assign_split(g, "kaggle") for g in groups]
    share = first.count("test") / len(first)
    assert abs(share - config.TEST_SIZE) < 0.03
    assert assign_split("cualquiera", "propias") == "externa"


def test_remove_background_and_diameter():
    from src.data.segment import estimate_diameter_px, remove_background

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:128, :128]
    mask = (yy - 64) ** 2 + (xx - 64) ** 2 <= 30 ** 2  # círculo de diámetro 60
    out = remove_background(image, mask)
    assert (out[~mask] == 255).all() and (out[mask] == 0).all()
    assert abs(estimate_diameter_px(mask) - 60) < 3


def test_fruq_solo_entrenamiento_y_grupos_por_ventana():
    from src.data.split import assign_split, group_key

    assert assign_split("cualquiera", "fruq") == "extra"
    ventana = config.FRUQ_STEP * config.FRUQ_GROUP_FRAMES
    a = group_key(f"external/fruq/PearQ/Good/Image{ventana + 1}.png", "fruq")
    b = group_key(f"external/fruq/PearQ/Good/Image{2 * ventana - 1}.png", "fruq")
    c = group_key(f"external/fruq/PearQ/Good/Image{2 * ventana}.png", "fruq")
    assert a == b != c
