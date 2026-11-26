"""Descarga de los modelos entrenados para la app desplegada (Fase 6 de CRISP-DM).

Los modelos (`*.keras`, `*.joblib`) no se versionan en git. Para la app en Streamlit Community Cloud
se publican como archivos de una *release* de GitHub (`config.MODELS_RELEASE_TAG`), junto con las
métricas guardadas (`eval_*.json`), y se descargan la primera vez que arranca la app.

Si el repositorio es privado, la descarga necesita un token de GitHub con permiso de lectura de
contenidos, guardado como secreto `GITHUB_TOKEN` en Streamlit (o como variable de entorno).

Publicar o actualizar la release (desde la raíz del repo, con `gh` autenticado):
    python -m src.app.modelos publicar
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from src import config

API = "https://api.github.com"


def release_files() -> dict[str, Path]:
    """Nombre del archivo en la release → ruta local donde lo espera la app."""
    models = [config.FINAL_TRANSFER, config.FINAL_MODEL]
    files = {f"{m}.keras": config.CHECKPOINTS_DIR / f"{m}.keras" for m in models}
    files[f"{config.FINAL_CLASSIC}.joblib"] = config.CHECKPOINTS_DIR / f"{config.FINAL_CLASSIC}.joblib"
    for m in (*models, config.FINAL_CLASSIC):
        for split in ("val", "test", "externa"):
            files[f"eval_{m}_{split}.json"] = config.RESULTS_DIR / f"eval_{m}_{split}.json"
    return files


def _request(url: str, token: str | None, accept: str = "application/vnd.github+json"):
    headers = {"Accept": accept, "User-Agent": "fruit-quality-app"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download_missing(token: str | None = None) -> list[str]:
    """Descarga los archivos de la release que falten en local. Devuelve los nombres descargados."""
    files = release_files()
    missing = {name: path for name, path in files.items() if not path.exists()}
    if not missing:
        return []
    token = token or os.environ.get("GITHUB_TOKEN")
    with _request(f"{API}/repos/{config.MODELS_RELEASE_REPO}/releases/tags/{config.MODELS_RELEASE_TAG}", token) as r:
        assets = {a["name"]: a["url"] for a in json.load(r)["assets"]}
    done = []
    for name, path in missing.items():
        if name not in assets:
            continue                      # p. ej. una métrica que no existe: la app funciona sin ella
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with _request(assets[name], token, accept="application/octet-stream") as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
        done.append(name)
    return done


def publish() -> None:
    """Crea (o actualiza) la release con los modelos y métricas locales."""
    import subprocess

    paths = [str(p) for p in release_files().values() if p.exists()]
    tag, repo = config.MODELS_RELEASE_TAG, config.MODELS_RELEASE_REPO
    exists = subprocess.run(["gh", "release", "view", tag, "-R", repo], capture_output=True).returncode == 0
    if not exists:
        subprocess.run(["gh", "release", "create", tag, "-R", repo, "--title", "Modelos entrenados para la app",
                        "--notes", "Modelos finales (transfer learning, CNN, SVM) y sus métricas, para la app "
                                   "desplegada. Generados con el código de este repositorio; ver docs/crisp-dm.md."],
                       check=True)
    subprocess.run(["gh", "release", "upload", tag, "-R", repo, "--clobber", *paths], check=True)
    print(f"{len(paths)} archivos en la release {tag}")


if __name__ == "__main__":
    publish() if sys.argv[1:] == ["publicar"] else print(download_missing() or "Nada que descargar.")
