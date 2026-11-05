# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University final project (Algoritmos y Programación III, 2026-2): fruit quality classification by computer vision, organized by **CRISP-DM** phases. Given an image of a single fruit on a simple background, predict quality (`Buena` / `Defectuosa` / `Podrida`) and size (`Pequeño` / `Mediano` / `Grande`). The assignment requires **≥2 traditional ML models + 1 CNN trained from scratch**, hyperparameter tuning, a Streamlit GUI, an IEEE-format report (≤7 pages) and an ethics/impact analysis.

All code, docstrings, comments, docs and commit messages are written in **Spanish** — keep it that way.

## Commands

All modules are run as packages from the repo root (`python -m src....`), since code imports `from src import config`.

```bash
source .venv/bin/activate                 # venv (Python 3.12); or: conda env create -f environment.yml
pip install -r requirements.txt

pytest                                    # pytest.ini sets pythonpath=. so `src` is importable
pytest tests/test_smoke.py::test_diameter_to_size_class   # single test

python -m src.main download               # Kaggle → data/raw/ (needs ~/.kaggle/kaggle.json) + Healthy-Defective-Fruits real images → data/external/
python -m src.main index                  # unified index of all sources → data/processed/index.csv
python -m src.main preprocess             # cache every indexed image at 128×128 → data/interim/img128/ (incremental)
streamlit run src/app/anotar.py           # tool to re-label Kaggle `Bad` into Defectuosa/Podrida
python -m src.data.relabel                # re-labeling progress + inter-annotator kappa
python -m src.main train                  # or: python -m src.training.train --model {cnn,classic}
streamlit run src/app/app.py              # deployment GUI (app.py prepends repo root to sys.path)

python -m src.data.own_images redimensionar   # shrink team photos to ≤1024 px before committing
python -m src.data.own_images validar         # check anotaciones.csv vs images (also run by pytest)
```

Notebooks are authored as **jupytext percent-format `.py` files** (`notebooks/01_eda.py`) and paired with an executed `.ipynb`. Edit the `.py`, then regenerate/execute the notebook (e.g. `jupytext --to ipynb notebooks/01_eda.py && jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`). Figures for the report are saved to `assets/<phase>/` as both SVG and PNG.

## Architecture

- `src/config.py` is the single source of truth for paths (`RAW_DIR`, `PROCESSED_DIR`, `OWN_DIR`/`OWN_IMAGES_DIR`/`OWN_ANNOTATIONS` for team-collected images, `CHECKPOINTS_DIR`, `RESULTS_DIR`), class lists, the Kaggle dataset id, and default hyperparameters (`IMAGE_SIZE=(128,128)`, `RANDOM_SEED=42`, splits). Don't hardcode these elsewhere.
- Pipeline (see `docs/arquitectura.md`): `data/download` → `data/segment` (OpenCV segmentation of individual fruits, diameter → size class) → `data/preprocess` (resize + [0,1] normalization) → two branches: `features/extract_features` (color HSV / texture GLCM-LBP / shape → vector φ(x)) feeding `models/classic_ml` (`build_models()` returns `{name: (estimator, param_grid)}`, `tune()` runs GridSearchCV with `f1_macro`), and `models/cnn` (Keras Sequential) → `training/train` (artifacts to `experiments/`) → `evaluation/evaluate` → `app/app.py`.
- Heavy dependencies (`cv2`, `tensorflow`, `sklearn`, `kaggle`) are **imported lazily inside functions** so `src` modules and the smoke tests import without them. Follow this pattern.
- Many functions are stubs raising `NotImplementedError("Implementar en Fase N")`; the phase number tells where they belong in `ROADMAP.md`.
- Notebooks should reuse code from `src/` rather than duplicate logic.

## Data strategy (decided in Phase 3 — see `docs/criterios_anotacion.md`, `docs/crisp-dm.md` §3.1)

- One fruit-agnostic model, 3 quality classes. The required Kaggle dataset (FruitNet, 6 fruits) only has `Good`/`Bad`, and `Bad` mixes mild defects with rot, so a fixed sample of `Bad` (200/fruit, seed 42) is **manually re-labeled** (`src/data/relabel.py`; one CSV per annotator in `data/anotaciones/` — versioned — to avoid merge conflicts; 10/fruit are labeled by everyone for Cohen's kappa; majority vote, ties → `Descartar`).
- Complemented by **Healthy-Defective-Fruits** (apple + mango, real images only; folder→class map in `config.HDF_LABELS`).
- All 3 classes must draw from the same sources, otherwise the model learns the source instead of quality. Known shortcut risk: half of HDF `rot` is PNG and HDF is all 256×256 → preprocessing must normalize format/size.
- Team photos (`data/external/propias/`) are an **external test set**, never used for training.
- **Everything downstream reads `data/processed/index.csv`** (id, ruta, fuente, fruta, calidad, grupo, split). Adding data = add a source loader in `src/data/dataset.py`, re-run `index` + `preprocess`; model/eval code must not special-case sources.
- Splits (`src/data/split.py`) are **hash-based per group**, so existing images never change split when data is added. Kaggle contains augmented copies (`<timestamp>_<n>.jpg`, 79 % of Pomegranate) → groups = folder + capture timestamp; never split by image. Weight samples by 1/group size when training.
- `prepare_image` in `src/data/preprocess.py` is the single preprocessing function — the app must use it too.
- Strong class imbalance (~5×) → use F1-macro and class weights / resampling (`imbalanced-learn` available); always compare against a majority-class baseline.
- Image sizes vary from 256×256 to 8000×6000 → resizing is mandatory.

## Repo conventions

- `data/*` and `experiments/*` contents are git-ignored (only `.gitkeep` tracked) **except `data/external/propias/`**: the team's own photos (white background, good lighting, one fruit per shot) and `anotaciones.csv` are versioned; protocol in its `README.md`. `tests/test_own_images.py` fails if the CSV and images get out of sync. Model files (`*.h5`, `*.keras`, `*.pkl`, `*.joblib`, …) and `kaggle.json` must never be committed.
- Git flow: `main` (stable) ← `dev` ← `feature/*` branches; PRs target `dev`.
- **No AI attribution:** never add `Co-Authored-By: Claude …` to commits or "Generated with Claude Code" to PRs/issues/comments (graded university project; explicit request from the team).
- When a phase advances, update its checklist in `ROADMAP.md` and the matching section in `docs/crisp-dm.md`. `ROADMAP.md` also maps each phase to rubric criteria (`docs/Rubrica_proyecto.xlsx`); the report draft lives in `docs/informe/informe_final.md`.
