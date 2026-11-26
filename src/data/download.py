"""Descarga del dataset base desde Kaggle hacia data/raw/ y de los complementarios a data/external/.

Requiere tener configurado el token de Kaggle (ver docs/instalacion.md).
Uso:  python -m src.data.download
"""
from __future__ import annotations

from src import config


def download_dataset() -> None:
    """Descarga y descomprime el dataset Fruit Quality Classification en data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Import diferido: la librería kaggle valida credenciales al importarse.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Descargando '{config.KAGGLE_DATASET}' en {config.RAW_DIR} ...")
    api.dataset_download_files(
        config.KAGGLE_DATASET, path=str(config.RAW_DIR), unzip=True
    )
    print("Descarga completada.")


def download_healthy_defective() -> None:
    """Descarga solo las imágenes reales de Healthy-Defective-Fruits en data/external/.

    Usa un clon parcial (sparse checkout) para no bajar las imágenes sintéticas (~50 % del repo).
    """
    import subprocess

    if config.HDF_DIR.exists():
        print(f"Ya existe {config.HDF_DIR}; se omite.")
        return
    print(f"Descargando Healthy-Defective-Fruits (solo reales) en {config.HDF_DIR} ...")
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    config.HDF_REPO, str(config.HDF_DIR)], check=True)
    subprocess.run(["git", "-C", str(config.HDF_DIR), "sparse-checkout", "set",
                    *config.HDF_REAL_DIRS.values()], check=True)
    print("Descarga completada.")


_REQUEST_INTERVAL = 1.1  # s entre peticiones: Zenodo responde 429 si se le piden muchas por minuto
_last_request = [0.0]


def _urlopen(req, retries: int = 8):
    """urlopen con ritmo limitado (todas las hebras comparten el reloj) y reintentos ante 429/5xx."""
    import threading
    import time
    import urllib.error
    import urllib.request

    lock = _urlopen.__dict__.setdefault("lock", threading.Lock())
    for attempt in range(retries):
        with lock:
            wait = _last_request[0] + _REQUEST_INTERVAL - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_request[0] = time.time()
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            time.sleep(int(exc.headers.get("Retry-After") or 30 * (attempt + 1)))
        except urllib.error.URLError:
            time.sleep(10 * (attempt + 1))
    raise OSError(f"Sin respuesta de {req.full_url}")


class _RemoteFile:
    """Archivo remoto de solo lectura con `seek` (peticiones HTTP Range): permite abrir un zip de
    Zenodo y extraer solo algunos archivos, sin bajar el zip completo."""

    def __init__(self, url: str):
        import urllib.request

        head = _urlopen(urllib.request.Request(url, method="HEAD"))
        self.url, self.size, self.pos = head.geturl(), int(head.headers["Content-Length"]), 0

    def seekable(self):
        """Interfaz de archivo para `zipfile`: el zip remoto admite saltos (lecturas por rangos HTTP)."""
        return True

    def tell(self):
        """Posición actual de lectura dentro del zip remoto."""
        return self.pos

    def seek(self, offset, whence=0):
        """Mueve la posición de lectura (como `io.IOBase.seek`) sin descargar nada."""
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        """Lee `n` bytes desde la posición actual con una petición HTTP Range."""
        import urllib.request

        end = self.size - 1 if n is None or n < 0 else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = _urlopen(req).read()
        self.pos += len(data)
        return data


def fruq_members(names: list[str]) -> list[str]:
    """Archivos de FruQ-multi a usar: frutas de FRUQ_FRUITS, 1 de cada FRUQ_STEP fotogramas."""
    import re
    from collections import defaultdict

    folders = defaultdict(list)
    for n in names:
        parts = n.split("/")
        if len(parts) == 4 and parts[1] in config.FRUQ_FRUITS and parts[3]:
            if (parts[1], parts[2].lower()) not in config.FRUQ_SKIP and parts[2].lower() in config.FRUQ_LABELS:
                folders[(parts[1], parts[2])].append(n)
    frame = lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1))  # noqa: E731
    return [n for members in folders.values() for n in sorted(members, key=frame)[::config.FRUQ_STEP]]


def _fetch_member(url: str, info, retries: int = 5) -> bytes:
    """Un archivo del zip remoto con UNA petición Range (cabecera local + datos), con reintentos."""
    import struct
    import time
    import urllib.request
    import zlib

    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size  # 1 KB holgura para "extra"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            blob = _urlopen(req).read()
            name_len, extra_len = struct.unpack("<HH", blob[26:30])
            data = blob[30 + name_len + extra_len:][:info.compress_size]
            if info.compress_type == 8:  # deflate
                data = zlib.decompress(data, -15)
            if zlib.crc32(data) != info.CRC:
                raise OSError("CRC distinto")
            return data
        except OSError:
            time.sleep(2 ** attempt)
    raise OSError(f"No se pudo descargar {info.filename}")


def download_fruq(workers: int = 3) -> None:
    """Extrae de FruQ-multi (4,6 GB en Zenodo) solo los fotogramas usados, en paralelo.

    Destino: data/external/fruq/<carpeta de fruta>/<clase>/<archivo>.
    """
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    remote = _RemoteFile(config.FRUQ_URL)
    with zipfile.ZipFile(remote) as z:  # solo lee el directorio central
        infos = {i.filename: i for i in z.infolist()}
    todo = [infos[n] for n in fruq_members(list(infos))
            if not (config.FRUQ_DIR / n.split("/", 1)[1]).exists()]
    print(f"FruQ: {len(todo)} fotogramas por descargar en {config.FRUQ_DIR}")

    def fetch(info) -> None:
        """Descarga un cuadro del zip de FruQ a su ruta local (escritura atómica vía `.part`)."""
        out = config.FRUQ_DIR / info.filename.split("/", 1)[1]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(_fetch_member(remote.url, info))
        tmp.rename(out)

    with ThreadPoolExecutor(workers) as pool:
        for done, _ in enumerate(pool.map(fetch, todo), 1):
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    print("Descarga completada.")


if __name__ == "__main__":
    download_dataset()
    download_healthy_defective()
    download_fruq()
