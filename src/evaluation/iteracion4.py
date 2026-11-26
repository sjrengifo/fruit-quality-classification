"""Evaluación de la cuarta iteración (solo manzana, banano y naranja) — Fase 5 de CRISP-DM.

Plan fijado antes de entrenar (`docs/crisp-dm.md` §3.12): se comparan el modelo actual de la app
(`config.FINAL_TRANSFER`) y los dos de la búsqueda `v4` en

1. **Fotos nuevas** (`data/external/propias/nov24/`, 35 fotos del 24 nov. 2026): **primera mirada**,
   la prueba principal. Se predicen con el mismo camino que la app (`Predictor`: U²-Net + recorte en
   vivo). No hay Defectuosas y solo 6 Podridas, así que no se calcula el F1 de 3 clases: se reporta
   cuántas Buena se reconocen como Buena y cuántas Podrida se detectan como dañadas o como Podrida.
   Los 3 limones quedan fuera del alcance de v4 y se muestran aparte.
2. **Las 81 fotos anteriores restringidas a las 3 frutas** (62 fotos): **cuarta mirada** a ese
   conjunto. F1-macro y diferencias pareadas frente al modelo actual sobre las mismas fotos.

Uso:  python -m src.evaluation.iteracion4      → experiments/results/iteracion4.json
"""
from __future__ import annotations

import csv
import json

import numpy as np

from src import config

MODELS = (config.FINAL_TRANSFER, "v4_0", "v4_1")
C = config.QUALITY_CLASSES


def nov24_rows() -> list[dict]:
    with open(config.NOV24_DIR / "anotaciones.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predict_nov24(model_name: str, rows: list[dict]) -> np.ndarray:
    from src.models.predict import Predictor

    predictor = Predictor(model_name)
    return np.array([C.index(predictor.predict(config.NOV24_DIR / "imagenes" / r["archivo"]).calidad) for r in rows])


def bootstrap(values: np.ndarray, n_boot: int = 2000, seed: int = config.RANDOM_SEED) -> list[float]:
    """IC95 de una proporción remuestreando fotos (cada foto, una fruta distinta salvo lo anotado)."""
    rng = np.random.default_rng(seed)
    stats = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return [float(x) for x in np.percentile(stats, [2.5, 97.5])]


def nov24_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    buena, podrida = C.index("Buena"), C.index("Podrida")
    ok_buena = (p[y == buena] == buena).astype(float)
    damaged = (p[y == podrida] != buena).astype(float)
    exact_podrida = (p[y == podrida] == podrida).astype(float)
    return {
        "fotos": int(len(y)),
        "buena_como_buena": f"{int(ok_buena.sum())}/{len(ok_buena)}",
        "recall_buena": float(ok_buena.mean()), "recall_buena_ic95": bootstrap(ok_buena),
        "podrida_detectada_como_danada": f"{int(damaged.sum())}/{len(damaged)}",
        "podrida_como_podrida": f"{int(exact_podrida.sum())}/{len(exact_podrida)}",
        "exactitud": float((y == p).mean()),
        "matriz_confusion": [[int(((y == i) & (p == j)).sum()) for j in range(len(C))] for i in range(len(C))],
    }


def main() -> None:
    from src.evaluation.evaluate import f1_macro, paired_diff_ci, predict, recall_of

    results = {"modelos": list(MODELS), "nov24": {}, "nov24_limon": {}, "externa_3_frutas": {}}

    rows = nov24_rows()
    y = np.array([C.index(r["calidad"]) for r in rows])
    in_scope = np.array([r["fruta"] in config.V4_FRUITS for r in rows])
    preds_new = {m: predict_nov24(m, rows) for m in MODELS}
    for m, p in preds_new.items():
        results["nov24"][m] = nov24_metrics(y[in_scope], p[in_scope])
        results["nov24_limon"][m] = [C[k] for k in p[~in_scope]]

    y_ex, _, rows_ex = predict(MODELS[0], "externa")
    keep = np.array([r["fruta"] in config.V4_FRUITS for r in rows_ex])
    groups = np.array([r["grupo"] for r in rows_ex])[keep]
    y_ex = y_ex[keep]
    preds_ex = {m: predict(m, "externa")[1][keep] for m in MODELS}
    podrida = C.index("Podrida")
    for m, p in preds_ex.items():
        r = {"fotos": int(keep.sum()), "f1_macro": f1_macro(y_ex, p), "recall_podrida": recall_of(podrida)(y_ex, p),
             "matriz_confusion": [[int(((y_ex == i) & (p == j)).sum()) for j in range(len(C))] for i in range(len(C))]}
        if m != MODELS[0]:
            r[f"dif_f1_vs_{MODELS[0]}"] = paired_diff_ci(y_ex, p, preds_ex[MODELS[0]], groups, f1_macro)
            r[f"dif_recall_podrida_vs_{MODELS[0]}"] = paired_diff_ci(y_ex, p, preds_ex[MODELS[0]], groups,
                                                                    recall_of(podrida))
        results["externa_3_frutas"][m] = r

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "iteracion4.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Fotos nuevas (primera mirada; {int(in_scope.sum())} fotos de {', '.join(config.V4_FRUITS)}):")
    for m, r in results["nov24"].items():
        print(f"  {m:14s} Buena→Buena {r['buena_como_buena']} (IC95 {r['recall_buena_ic95'][0]:.2f}–"
              f"{r['recall_buena_ic95'][1]:.2f}) · Podrida detectada como dañada {r['podrida_detectada_como_danada']}"
              f" · como Podrida {r['podrida_como_podrida']} · limones: {results['nov24_limon'][m]}")
    print(f"Fotos anteriores, solo 3 frutas (cuarta mirada; {int(keep.sum())} fotos):")
    for m, r in results["externa_3_frutas"].items():
        diff = r.get(f"dif_f1_vs_{MODELS[0]}")
        extra = f" · Δ F1 vs actual {diff[0]:+.3f} [{diff[1]:+.3f}; {diff[2]:+.3f}]" if diff else ""
        print(f"  {m:14s} F1 {r['f1_macro']:.3f} · recall Podrida {r['recall_podrida']:.2f}{extra}")


if __name__ == "__main__":
    main()
