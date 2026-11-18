"""Descarga las fuentes completas en Bronze, tal como llegan.

- Kaggle "threnjen/board-games-database-from-boardgamegeek"  ->  data/bronze/kaggle_threnjen/
- Ranking de BGG (GitHub beefsack/bgg-ranking-historicals)    ->  data/bronze/bgg_ranks/

No cambia ningún valor. Descarga, comprueba que están los ficheros y las columnas que usa
el pipeline, y deja en data/bronze/_manifiesto.json la versión, el tamaño, el SHA-256 y el
número de filas de cada fichero.

user_ratings.csv trae los nombres de usuario reales: data/bronze/ no se sube al repositorio
y la seudonimización se hace al pasar a Silver.

Requisito: token de Kaggle en .env (KAGGLE_API_TOKEN=...), generado en kaggle.com/settings.

Uso (desde la raíz del repositorio):
    python scripts/descargar_bronze.py            # descarga solo lo que falte
    python scripts/descargar_bronze.py --forzar   # vuelve a descargarlo todo
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import kagglehub
import requests
from dotenv import load_dotenv
from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError

from siguiente_partida import configurar_logging

# --------------------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze"
DIR_KAGGLE = BRONZE / "kaggle_threnjen"
DIR_RANKING = BRONZE / "bgg_ranks"
MANIFIESTO = BRONZE / "_manifiesto.json"

# --------------------------------------------------------------------------- fuentes
KAGGLE_DATASET = "threnjen/board-games-database-from-boardgamegeek"
KAGGLE_VERSION: int | None = 4
URL_KAGGLE = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"

FECHA_RANKING = "2026-11-12"  # foto fija del ranking
URL_RANKING = ("https://raw.githubusercontent.com/beefsack/bgg-ranking-historicals/"
               f"master/{FECHA_RANKING}.csv")

# Ficheros que usa el pipeline y columnas sin las que no puede funcionar.
# El resto de ficheros del dataset se descargan igual, pero no se validan.
COLUMNAS_CLAVE = {
    DIR_KAGGLE / "games.csv": {"BGGId", "Name"},
    DIR_KAGGLE / "mechanics.csv": {"BGGId"},
    DIR_KAGGLE / "themes.csv": {"BGGId"},
    DIR_KAGGLE / "user_ratings.csv": {"BGGId", "Rating", "Username"},
    DIR_RANKING / f"{FECHA_RANKING}.csv": {"ID", "Rank"},
}

# En Windows sys.maxsize no cabe en un long de C; 2**31-1 sí.
csv.field_size_limit(2**31 - 1)

log = logging.getLogger("bronze")


class ErrorBronze(Exception):
    """Fallo esperado con un mensaje claro para el usuario."""


# --------------------------------------------------------------------------- descargas
def descargar_kaggle(forzar: bool) -> int | None:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise ErrorBronze("Falta KAGGLE_API_TOKEN en .env. Genera uno en "
                          "kaggle.com/settings > API > Generate New Token.")

    handle = f"{KAGGLE_DATASET}/versions/{KAGGLE_VERSION}" if KAGGLE_VERSION else KAGGLE_DATASET
    if KAGGLE_VERSION is None:
        log.warning("Versión de Kaggle sin fijar: se descarga la última publicada")
    log.info("Kaggle: descargando %s en %s", handle, DIR_KAGGLE.relative_to(ROOT))
    try:
        kagglehub.dataset_download(handle, output_dir=str(DIR_KAGGLE), force_download=forzar)
    except UnauthenticatedError as e:
        raise ErrorBronze("Kaggle rechaza el token. Genera uno nuevo y cópialo en .env.") from e
    except KaggleApiHTTPError as e:
        raise ErrorBronze(f"Error de la API de Kaggle: {e}") from e
    except FileExistsError as e:
        raise ErrorBronze(f"{e}. Si cambió la versión del dataset, ejecuta con --forzar.") from e

    version = KAGGLE_VERSION or leer_version_descargada()
    log.info("Kaggle: descarga completa (versión %s)", version or "desconocida")
    return version


def leer_version_descargada() -> int | None:
    """kagglehub deja una marca en .complete/datasets/<dueño>/<dataset>/<versión>/."""
    marcas = DIR_KAGGLE.joinpath(".complete", "datasets", *KAGGLE_DATASET.split("/"))
    if not marcas.is_dir():
        return None
    versiones = [int(p.name) for p in marcas.iterdir() if p.name.isdigit()]
    return max(versiones) if versiones else None


def descargar_ranking(forzar: bool) -> None:
    destino = DIR_RANKING / f"{FECHA_RANKING}.csv"
    if destino.exists() and not forzar:
        log.info("Ranking: ya estaba en %s, no se descarga", destino.relative_to(ROOT))
        return
    log.info("Ranking: descargando la foto del %s", FECHA_RANKING)
    try:
        r = requests.get(URL_RANKING, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise ErrorBronze(f"No se pudo descargar el ranking ({URL_RANKING}): {e}") from e
    DIR_RANKING.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".part")
    temporal.write_bytes(r.content)   # los bytes tal cual llegan
    temporal.replace(destino)         # sin ficheros a medias si se corta
    log.info("Ranking: guardado en %s", destino.relative_to(ROOT))


# --------------------------------------------------------------------------- descripción y validación
def describir_csv(ruta: Path) -> dict:
    """Tamaño, SHA-256, columnas y filas de datos (sin cabecera) de un CSV."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    # csv.reader y no contar líneas: un campo entre comillas puede tener saltos de línea
    with open(ruta, encoding="utf-8", newline="") as f:
        lector = csv.reader(f)
        columnas = next(lector, [])
        filas = sum(1 for _ in lector)
    return {"bytes": ruta.stat().st_size, "sha256": h.hexdigest(),
            "filas": filas, "columnas": columnas}


def validar(descripciones: dict[Path, dict]) -> None:
    errores = []
    for ruta, clave in COLUMNAS_CLAVE.items():
        nombre = ruta.relative_to(BRONZE).as_posix()
        if ruta not in descripciones:
            errores.append(f"falta {nombre}")
            continue
        d = descripciones[ruta]
        if faltan := clave - set(d["columnas"]):
            errores.append(f"{nombre}: faltan las columnas {sorted(faltan)}")
        if d["filas"] == 0:
            errores.append(f"{nombre}: no tiene filas")
    for e in errores:
        log.error("Validación: %s", e)
    if errores:
        raise ErrorBronze("Bronze no supera la validación: revisa los errores anteriores")
    log.info("Validación: los %d ficheros que usa el pipeline tienen sus columnas clave",
             len(COLUMNAS_CLAVE))


# --------------------------------------------------------------------------- principal
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forzar", action="store_true", help="vuelve a descargarlo todo")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    configurar_logging("descarga_bronze.log", carpeta=ROOT / "logs")
    inicio = time.perf_counter()
    log.info("Bronze: inicio de la descarga")

    try:
        version = descargar_kaggle(args.forzar)
        descargar_ranking(args.forzar)

        descripciones = {}
        for ruta in sorted([*DIR_KAGGLE.glob("*.csv"), *DIR_RANKING.glob("*.csv")]):
            d = descripciones[ruta] = describir_csv(ruta)
            log.info("%-36s %12s filas  %8.1f MB", ruta.relative_to(BRONZE).as_posix(),
                     f"{d['filas']:,}".replace(",", "."), d["bytes"] / 1e6)
        validar(descripciones)
    except ErrorBronze as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado en la descarga de Bronze")
        return 1

    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ficheros = {ruta.relative_to(BRONZE).as_posix(): d for ruta, d in descripciones.items()}
    manifiesto = {
        "creado": ahora,
        "nota": ("Ficheros sin modificar. user_ratings.csv contiene nombres de usuario "
                 "reales: data/bronze/ no se sube al repositorio."),
        "fuentes": {
            "kaggle_threnjen": {"url": URL_KAGGLE, "dataset": KAGGLE_DATASET,
                                "version": version, "carpeta": "kaggle_threnjen"},
            "bgg_ranks": {"url": URL_RANKING, "fecha_foto": FECHA_RANKING,
                          "carpeta": "bgg_ranks"},
        },
        "ficheros": ficheros,
    }
    MANIFIESTO.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Manifiesto: %s", MANIFIESTO.relative_to(ROOT))
    log.info("Bronze: fin (%d ficheros, %.0f s)", len(ficheros), time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())