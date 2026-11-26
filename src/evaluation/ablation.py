"""Ablación de características y generalización entre fuentes — Fase 4/5 de CRISP-DM.

Con los hiperparámetros del mejor modelo clásico guardado:
- **ablación:** F1-macro en validación usando solo algunos grupos de características;
- **entre fuentes:** entrenar con una fuente (Kaggle o HDF) y validar en la otra. Mide si el modelo
  aprende calidad o el "aspecto" de cada dataset; anticipa el desempeño con fotos propias.

Uso:  python -m src.evaluation.ablation [--model svm]
"""
from __future__ import annotations

import argparse

import numpy as np

from src import config

GROUP_SETS = [["color"], ["textura"], ["forma"], ["color", "textura"], ["color", "textura", "forma"]]


def main() -> None:
    """Ablación de rasgos del SVM (color, textura, forma) y prueba de generalización entre fuentes."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import f1_score

    from src.features.extract_features import group_columns, load_features
    from src.training.train import labels, sample_weights, save_results

    parser = argparse.ArgumentParser(description="Ablación y generalización entre fuentes")
    parser.add_argument("--model", default="svm")
    name = parser.parse_args().model
    model = joblib.load(config.CHECKPOINTS_DIR / f"{name}.joblib")

    X_tr, rows_tr = load_features("train")
    X_va, rows_va = load_features("val")
    y_tr, y_va = labels(rows_tr), labels(rows_va)
    src_tr = np.array([r["fuente"] for r in rows_tr])
    src_va = np.array([r["fuente"] for r in rows_va])
    f1 = lambda y, p: float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0))  # noqa: E731
    results = {"ablacion": {}, "entre_fuentes": {}}

    print(f"Ablación ({name}) — F1-macro en validación")
    w = sample_weights(rows_tr)
    for groups in GROUP_SETS:
        cols = group_columns(groups)
        pred = clone(model).fit(X_tr[:, cols], y_tr, clf__sample_weight=w).predict(X_va[:, cols])
        results["ablacion"]["+".join(groups)] = f1(y_va, pred)
        print(f"  {'+'.join(groups):22s} {results['ablacion']['+'.join(groups)]:.3f}")

    print("Entre fuentes — F1-macro en validación")
    for source in sorted(set(src_tr)):
        m = src_tr == source
        clf = clone(model).fit(X_tr[m], y_tr[m], clf__sample_weight=sample_weights([r for r in rows_tr if r["fuente"] == source]))
        for target in sorted(set(src_va)):
            k = src_va == target
            results["entre_fuentes"][f"{source}→{target}"] = f1(y_va[k], clf.predict(X_va[k]))
            print(f"  entrena {source:7s} → valida {target:7s} {results['entre_fuentes'][f'{source}→{target}']:.3f}")
    save_results(f"ablacion_{name}", results)


if __name__ == "__main__":
    main()
