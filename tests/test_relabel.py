"""Pruebas de la re-anotación de `Bad` (src/data/relabel.py)."""
from src.data import relabel


def _bad_dir(tmp_path, fruits=("Apple", "Lime"), n=5):
    bad = tmp_path / "Bad Quality_Fruits"
    for fruit in fruits:
        d = bad / f"{fruit}_Bad"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img{i}.jpg").write_bytes(b"")
    return bad


def test_sample_is_reproducible(tmp_path):
    bad = _bad_dir(tmp_path)
    a = relabel.build_sample(bad, per_fruit=3)
    assert a == relabel.build_sample(bad, per_fruit=3)
    assert len(a) == 6
    assert a[0]["archivo"].startswith("Bad Quality_Fruits/Apple_Bad/")


def test_save_undo_and_load(tmp_path):
    item = {"archivo": "Bad Quality_Fruits/Apple_Bad/img0.jpg", "fruta": "Apple"}
    relabel.save_label("Samuel José", item, "Podrida", tmp_path)
    relabel.save_label("Samuel José", item, "Defectuosa", tmp_path)  # corrección: prevalece la última
    assert relabel.load_annotations(tmp_path) == {"samueljose": {item["archivo"]: "Defectuosa"}}
    assert relabel.undo_last("Samuel José", tmp_path) == item["archivo"]
    assert relabel.load_annotations(tmp_path) == {"samueljose": {item["archivo"]: "Podrida"}}


def test_pending_shares_agreement_items_only():
    sample = [
        {"archivo": "a", "fruta": "Apple", "acuerdo": True},
        {"archivo": "b", "fruta": "Apple", "acuerdo": False},
        {"archivo": "c", "fruta": "Apple", "acuerdo": False},
    ]
    annotations = {"juan": {"a": "Podrida", "b": "Podrida"}}
    # "a" es de acuerdo (Samuel también la anota); "b" ya la hizo Juan.
    assert [i["archivo"] for i in relabel.pending(sample, annotations, "Samuel")] == ["a", "c"]


def test_consolidated_majority_and_tie():
    annotations = {
        "a": {"x": "Podrida", "y": "Podrida"},
        "b": {"x": "Podrida", "y": "Defectuosa"},
        "c": {"x": "Defectuosa"},
    }
    assert relabel.consolidated_labels(annotations) == {"x": "Podrida", "y": "Descartar"}
