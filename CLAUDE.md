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
python -m src.main segment                # U²-Net (rembg) background removal → data/interim/seg128/ + mask128/ (incremental, ~0.7 s/img with 2 workers)
python -m src.main crop                   # crop each fruit to its mask box → data/interim/crop128/ + cropmask128/ (fast)
python -m src.main features               # φ(x) (68 color/texture/shape features) → data/processed/features.npz
python -m src.evaluation.shortcut         # do background / silhouette alone predict the class?
streamlit run src/app/anotar.py           # tool to re-label Kaggle `Bad` into Defectuosa/Podrida
python -m src.data.relabel                # re-labeling progress + inter-annotator kappa
python -m src.training.train --model classic      # baselines + LogReg/SVM/RF with GridSearchCV → experiments/
python -m src.training.train --model cnn          # CNN with CNN_DEFAULT (~2–4 min/epoch on CPU); cnn_search / cnn_stable run SEARCHES in models/cnn.py (resumable)
python -m src.training.colab                      # zip (src/ + index + train/val/extra crops, no test/externa) for GPU training via notebooks/colab_entrenar_cnn
                                                  # transfer learning (models/transfer.py, EfficientNetB0): searches tl_busqueda / tl_entre_fuentes (leave-one-source-out via holdout=) run in Colab
python -m src.evaluation.evaluate --model svm     # metrics + group-bootstrap 95% CI + error list on val; test/externa need --final (run once, final model)
python -m src.evaluation.ablation         # feature ablation + cross-source generalization (uses saved SVM)
python -m src.evaluation.figuras_modelado # Phase 4 figures from experiments/ → assets/modelado/
streamlit run src/app/app.py              # deployment GUI: src/models/predict.Predictor (segment_and_crop + config.FINAL_MODEL); app.py prepends repo root to sys.path
python -m src.models.predict foto.jpg      # predict one image from the terminal (--model svm)

python -m src.data.own_images redimensionar   # shrink team photos to ≤1024 px before committing
python -m src.data.own_images validar         # check anotaciones.csv vs images (also run by pytest)
```

Notebooks are authored as **jupytext percent-format `.py` files** (`notebooks/01_eda.py`) and paired with an executed `.ipynb`. Edit the `.py`, then regenerate/execute the notebook (e.g. `jupytext --to ipynb notebooks/01_eda.py && jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb`). Figures for the report are saved to `assets/<phase>/` as both SVG and PNG.

## Architecture

- `src/config.py` is the single source of truth for paths (`RAW_DIR`, `PROCESSED_DIR`, `OWN_DIR`/`OWN_IMAGES_DIR`/`OWN_ANNOTATIONS` for team-collected images, `CHECKPOINTS_DIR`, `RESULTS_DIR`), class lists, the Kaggle dataset id, and default hyperparameters (`IMAGE_SIZE=(128,128)`, `RANDOM_SEED=42`, splits). Don't hardcode these elsewhere.
- Pipeline (see `docs/arquitectura.md`): `data/download` → `data/dataset` (index) → `data/preprocess` → `data/segment` (U²-Net mask, white background, then `crop_fruit`) → two branches: `features/extract_features` (color HSV / texture GLCM-LBP / shape → φ(x), fruit pixels only) feeding `models/classic_ml` (`build_models()` returns `{name: (pipeline, param_grid)}` with the classifier as step `clf`; `tune()` runs GridSearchCV with `f1_macro` on `StratifiedGroupKFold`), and `models/cnn` (Keras Sequential; rescaling + augmentation live inside the model, input is uint8) → `training/train` (models to `experiments/checkpoints/`, metrics JSON to `experiments/results/`, CNN history to `experiments/logs/`) → `evaluation/` → `app/app.py`.
- Model selection uses `train` + `val` only; `test` and `externa` are reserved for Phase 5. The chosen models are fixed in `config.FINAL_MODEL` (CNN `cnn_estable_1`) and `config.FINAL_CLASSIC` (SVM) — the app and Phase 5 read them from there. `sample_weights()` in `training/train.py` (1/group size, classes balanced) is the single imbalance mechanism.
- Heavy dependencies (`cv2`, `tensorflow`, `sklearn`, `kaggle`) are **imported lazily inside functions** so `src` modules and the smoke tests import without them. Follow this pattern.
- Many functions are stubs raising `NotImplementedError("Implementar en Fase N")`; the phase number tells where they belong in `ROADMAP.md`.
- Notebooks should reuse code from `src/` rather than duplicate logic.

## Data strategy (decided in Phase 3 — see `docs/criterios_anotacion.md`, `docs/crisp-dm.md` §3.1)

- One fruit-agnostic model, 3 quality classes. The required Kaggle dataset (FruitNet, 6 fruits) only has `Good`/`Bad`, and `Bad` mixes mild defects with rot, so a fixed sample of `Bad` (200/fruit, seed 42) is **manually re-labeled** (`src/data/relabel.py`; one CSV per annotator in `data/anotaciones/` — versioned — to avoid merge conflicts; 10/fruit are labeled by everyone for Cohen's kappa; majority vote, ties → `Descartar`).
- Complemented by **Healthy-Defective-Fruits** (apple + mango, real images only; folder→class map in `config.HDF_LABELS`).
- All 3 classes must draw from the same sources, otherwise the model learns the source instead of quality. Known shortcut risk: half of HDF `rot` is PNG and HDF is all 256×256 → preprocessing must normalize format/size.
- **FruQ** (Zenodo, CC BY 4.0; `download_fruq` pulls only selected frames from the 4.6 GB zip via HTTP ranges, rate-limited — Zenodo answers 429 otherwise): time-lapse frames, each fruit is ONE object going Fresh→Mild→Rotten. Only whole single fruits (`config.FRUQ_FRUITS`), 1 of every `FRUQ_STEP` frames, groups = windows of `FRUQ_GROUP_FRAMES` frames. It is a **train-only source**: split `extra` (`config.TRAIN_ONLY_SOURCES`), never in val/test; training adds it only with `extra=True`.
- Team photos (`data/external/propias/`) are an **external test set**, never used for training. They were already evaluated once (Phase 5); any later model evaluated on them must be reported as a second look.
- **Everything downstream reads `data/processed/index.csv`** (id, ruta, fuente, fruta, calidad, grupo, split). Adding data = add a source loader in `src/data/dataset.py`, re-run `index` + `preprocess`; model/eval code must not special-case sources.
- Splits (`src/data/split.py`) are **hash-based per group**, so existing images never change split when data is added. Kaggle contains augmented copies (`<timestamp>_<n>.jpg`, 79 % of Pomegranate) → groups = folder + capture timestamp; HDF repeats the same fruit/session hundreds of times → groups = near-duplicate clusters (`near_duplicate_groups`, color-histogram single linkage per folder). Never split or cross-validate by image (doing so inflated HDF F1 to 0.99). Weight samples by 1/group size when training.
- `prepare_image` (`src/data/preprocess.py`) + `segment_image` (`src/data/segment.py`) are the single preprocessing path — the app must use them too. Models train on the **cropped** cache (`load_split()` default `variant="crop"`: fruit on white, centered, same zoom) because raw background and framing predict the class (shortcuts). The uncropped mask (`mask128`) is kept for size estimation.
- Known limitation: a model trained on one source fails on the other (SVM F1 0.06–0.42 cross-source vs. 0.79 in-source) → the team's own photos are the evaluation that matters; don't report val/test numbers as real-world performance.
- Grayscale images are excluded at index time (`data/processed/excluidas.csv`): they are augmented copies that only exist in `Buena`.
- Strong class imbalance (~5×) → use F1-macro and class weights / resampling (`imbalanced-learn` available); always compare against a majority-class baseline.
- Image sizes vary from 256×256 to 8000×6000 → resizing is mandatory.

## Repo conventions

- `data/*` and `experiments/*` contents are git-ignored (only `.gitkeep` tracked) **except `data/external/propias/`**: the team's own photos (white background, good lighting, one fruit per shot) and `anotaciones.csv` are versioned; protocol in its `README.md`. `tests/test_own_images.py` fails if the CSV and images get out of sync. Model files (`*.h5`, `*.keras`, `*.pkl`, `*.joblib`, …) and `kaggle.json` must never be committed.
- Git flow: `main` (stable) ← `dev` ← `feature/*` branches; PRs target `dev`.
- **No AI attribution:** never add `Co-Authored-By: Claude …` to commits or "Generated with Claude Code" to PRs/issues/comments (graded university project; explicit request from the team).
- When a phase advances, update its checklist in `ROADMAP.md` and the matching section in `docs/crisp-dm.md`. `ROADMAP.md` also maps each phase to rubric criteria (`docs/Rubrica_proyecto.xlsx`); the report draft lives in `docs/informe/informe_final.md`.
