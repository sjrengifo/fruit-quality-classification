"""Pruebas del recorte, las características, los grupos de casi-duplicados y los pesos por muestra."""
import numpy as np
from PIL import Image

from src import config
from src.data.segment import crop_fruit
from src.data.split import near_duplicate_groups
from src.features.extract_features import FEATURE_GROUPS, extract_all, group_columns
from src.training.train import sample_weights


def _fruit(size=320, center=(100, 220), radius=40):
    yy, xx = np.mgrid[:size, :size]
    mask = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
    image = np.where(mask[..., None], np.uint8([200, 40, 40]), np.uint8([10, 90, 10])).astype(np.uint8)
    return image, mask


def test_crop_fruit_centers_and_fills_frame():
    image, mask = _fruit()
    small_mask = np.asarray(Image.fromarray(mask).resize((128, 128), Image.Resampling.NEAREST))
    crop, crop_mask = crop_fruit(image, small_mask)  # máscara a menor resolución que la imagen
    assert crop.shape == (*config.IMAGE_SIZE, 3) and crop_mask.shape == config.IMAGE_SIZE
    ys, xs = np.nonzero(crop_mask)
    assert (xs.max() - xs.min()) / config.IMAGE_SIZE[1] > 0.8          # la fruta llena el cuadro
    assert abs(xs.mean() - 64) < 3 and abs(ys.mean() - 64) < 3          # y queda centrada
    assert (crop[0, 0] == 255).all()                                    # fondo blanco


def test_crop_fruit_with_empty_mask_keeps_image():
    image, _ = _fruit()
    crop, crop_mask = crop_fruit(image, np.zeros((128, 128), dtype=bool))
    assert crop_mask.mean() > 0.8 and crop.shape == (*config.IMAGE_SIZE, 3)


def test_features_have_fixed_length_and_ignore_background():
    image, mask = _fruit(128, (64, 64), 50)
    other = image.copy()
    other[~mask] = 255  # cambiar el fondo no debe cambiar el color ni la textura
    a, b = extract_all(image, mask), extract_all(other, mask)
    assert a.shape == (sum(FEATURE_GROUPS.values()),) and np.isfinite(a).all()
    cols = group_columns(["color", "textura"])
    assert np.allclose(a[cols], b[cols], atol=1e-5)
    assert len(group_columns(["forma"])) == FEATURE_GROUPS["forma"]


def test_near_duplicate_groups(tmp_path):
    colors = [(200, 30, 30), (202, 32, 30), (30, 30, 200)]  # dos rojos casi iguales y un azul
    paths = []
    for i, c in enumerate(colors):
        paths.append(tmp_path / f"{i}.png")
        Image.new("RGB", (64, 64), c).save(paths[-1])
    g = near_duplicate_groups(paths)
    assert g[0] == g[1] != g[2]


def test_sample_weights_balance_classes_and_groups():
    rows = ([{"grupo": "a", "calidad": "Buena"}] * 8 + [{"grupo": "b", "calidad": "Buena"}]
            + [{"grupo": "c", "calidad": "Podrida"}])
    w = sample_weights(rows)
    assert np.isclose(w.mean(), 1.0)
    assert np.isclose(w[:9].sum(), w[9:].sum())      # cada clase pesa lo mismo
    assert np.isclose(w[:8].sum(), w[8])             # 8 copias de un grupo = 1 foto


def test_bootstrap_ci_resamples_groups():
    from src.evaluation.evaluate import bootstrap_ci, f1_macro, recall_of

    y = np.array([0, 1, 2] * 10)
    groups = np.repeat(np.arange(10), 3)
    assert bootstrap_ci(y, y, groups, f1_macro, n_boot=200) == (1.0, 1.0)  # perfecto → [1, 1]
    wrong = y.copy()
    wrong[:6] = 0                                                             # 2 grupos con errores
    lo, hi = bootstrap_ci(y, wrong, groups, recall_of(2), n_boot=500)
    assert 0 <= lo < recall_of(2)(y, wrong) < hi <= 1


def test_paired_diff_ci():
    from src.evaluation.evaluate import f1_macro, paired_diff_ci

    y = np.array([0, 1, 2] * 10)
    groups = np.repeat(np.arange(10), 3)
    assert paired_diff_ci(y, y, y, groups, f1_macro, n_boot=100) == (0.0, 0.0, 0.0)  # mismo modelo
    worse = y.copy()
    worse[::3] = 1                                                                    # falla en todos los grupos
    diff, lo, hi = paired_diff_ci(y, y, worse, groups, f1_macro, n_boot=300)
    assert 0 < lo <= diff <= hi


def test_busquedas_tl_usan_el_modelo_indicado(tmp_path, monkeypatch):
    """Las configuraciones de transfer learning no pueden caer en la CNN por defecto, y al retomar
    una búsqueda se reentrena un resultado guardado de otro tipo de modelo."""
    import json

    from src.models import transfer
    from src.training import train

    for search in ("tl_busqueda", "tl_entre_fuentes"):
        for c in transfer.SEARCHES[search]:
            assert c["modelo"] in ("transfer", "cnn")
    assert all(c["modelo"] == "transfer" for c in transfer.SEARCHES["tl_busqueda"])

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(train, "all_searches", lambda: {"s": [{"modelo": "transfer"}, {"modelo": "cnn"}]})
    fake = {"epocas": 1, "parametros": 1, "train": {"f1_macro": 1.0}, "val": {"f1_macro": 1.0, "por_fuente": {}}}
    for i in (0, 1):  # ambos guardados como CNN (sin "modelo")
        (tmp_path / f"s_{i}.json").write_text(json.dumps({**fake, "configuracion": {}}))
    trained = []
    monkeypatch.setattr(train, "train_config", lambda name, o: trained.append(name) or {**fake, "configuracion": o})
    train.search_cnn("s")
    assert trained == ["s_0"]
