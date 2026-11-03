"""Pruebas de la validación de la recolección propia (data/external/propias/)."""
from src import config
from src.data.own_images import validate_annotations

HEADER = ",".join(config.OWN_ANNOTATION_COLUMNS)


def _setup(tmp_path, rows, images):
    images_dir = tmp_path / "imagenes"
    images_dir.mkdir()
    for name in images:
        (images_dir / name).write_bytes(b"")
    csv_path = tmp_path / "anotaciones.csv"
    csv_path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return csv_path, images_dir


def test_repo_annotations_are_valid():
    """El anotaciones.csv versionado debe ser siempre consistente con las imágenes."""
    assert validate_annotations() == []


def test_valid_row(tmp_path):
    csv_path, images_dir = _setup(
        tmp_path,
        ["Apple_Defectuosa_001.jpg,Apple,Defectuosa,Mediano,7.5,Samuel,2026-11-01,golpe"],
        ["Apple_Defectuosa_001.jpg"],
    )
    assert validate_annotations(csv_path, images_dir) == []


def test_detects_errors(tmp_path):
    csv_path, images_dir = _setup(
        tmp_path,
        [
            "falta.jpg,Fresa,Regular,Enorme,abc,Samuel,2026-11-01,",
        ],
        ["sin_anotar.jpg"],
    )
    errors = validate_annotations(csv_path, images_dir)
    assert len(errors) == 6  # imagen inexistente, fruta, calidad, tamaño, diámetro, sin anotar
