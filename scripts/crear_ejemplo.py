"""Crea la muestra sintética de data/ejemplo/ a partir de Bronze.

La muestra tiene los mismos ficheros, carpetas y columnas que Bronze, para que el pipeline,
los tests y Docker funcionen sin cuenta de Kaggle:

- Catálogo real: los N_JUEGOS juegos con más valoraciones en BGG. Sus filas de games,
  mechanics, themes, subcategories y del ranking se copian sin cambiar ningún valor.
- Valoraciones inventadas: N_USUARIOS usuarios ficticios (ejemplo_00001, ...) con gustos
  generados al azar. De user_ratings.csv real solo se lee la columna Rating, y solo para
  copiar su formato de escritura; no se guarda nada de ese fichero.

Cómo se inventan las valoraciones (reproducible con SEMILLA):
1. Cada usuario recibe un gusto por las 8 categorías de BGG (Cat:Strategy, Cat:Family...)
   y una complejidad preferida (GameWeight).
2. Elige qué juegos valora con una probabilidad que sube con la popularidad real del juego
   y con lo que encaja con su gusto.
3. La nota parte de la nota media real del juego, más un sesgo propio del usuario, más un
   término de afinidad y ruido. Se redondea como lo haría una persona (sobre todo enteros).

Las columnas agregadas de games.csv (AvgRating, NumUserRatings...) son las reales de BGG y
no se corresponden con las valoraciones inventadas.

Uso (desde la raíz del repositorio, con Bronze ya descargado):
    python scripts/crear_ejemplo.py
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from descargar_bronze import (DIR_KAGGLE, DIR_RANKING, FECHA_RANKING, KAGGLE_DATASET,
                              MANIFIESTO, ROOT, describir_csv)
from siguiente_partida import configurar_logging

# --------------------------------------------------------------------------- rutas
EJEMPLO = ROOT / "data" / "ejemplo"
EJ_KAGGLE = EJEMPLO / DIR_KAGGLE.name
EJ_RANKING = EJEMPLO / DIR_RANKING.name

# --------------------------------------------------------------------------- parámetros
N_JUEGOS = 300            # catálogo: los 300 juegos con más valoraciones en BGG
N_USUARIOS = 2000         # usuarios ficticios
MEDIANA_VAL_USUARIO = 25  # valoraciones por usuario: log-normal con esta mediana...
MIN_VAL_USUARIO = 10      # ...recortada a [10, 150]
MAX_VAL_USUARIO = 150
POTENCIA_POP = 0.7        # peso de la popularidad real al elegir juegos (0 = ninguno)
PESO_AFINIDAD = 1.5       # peso del gusto del usuario al elegir juegos
SEMILLA = 2026

FICHEROS_CATALOGO = ["games.csv", "mechanics.csv", "themes.csv", "subcategories.csv"]
COLS_CAT = ["Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family",
            "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens"]

log = logging.getLogger("ejemplo")


def leer_texto(ruta: Path, **kw) -> pd.DataFrame:
    """Todo como texto y sin convertir 'NA', 'null'... en nulos: los valores se copian tal cual."""
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, **kw)


def escribir_texto(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8", lineterminator="\n")


def numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


# --------------------------------------------------------------------------- catálogo real
def seleccionar_catalogo(games: pd.DataFrame) -> pd.DataFrame:
    orden = numero(games["NumUserRatings"]).sort_values(ascending=False, kind="stable")
    catalogo = games.loc[orden.index[:N_JUEGOS]]
    minimo = numero(catalogo["NumUserRatings"]).min()
    log.info("Catálogo: %d juegos; el que menos valoraciones tiene en BGG suma %d",
             len(catalogo), minimo)
    return catalogo


def copiar_catalogo(ids: set[str]) -> None:
    for nombre in FICHEROS_CATALOGO:
        df = leer_texto(DIR_KAGGLE / nombre)
        sub = df[df["BGGId"].isin(ids)]
        escribir_texto(sub, EJ_KAGGLE / nombre)
        log.info("Copiado %-20s %5d de %6d filas", nombre, len(sub), len(df))

    nombre = f"{FECHA_RANKING}.csv"
    ranking = leer_texto(DIR_RANKING / nombre)
    sub = ranking[ranking["ID"].isin(ids)]
    escribir_texto(sub, EJ_RANKING / nombre)
    log.info("Copiado ranking %-12s %5d de %6d filas (%d juegos del catálogo no están en él)",
             nombre, len(sub), len(ranking), len(ids) - len(sub))


# --------------------------------------------------------------------------- valoraciones inventadas
def formato_notas() -> callable:
    """Copia cómo escribe Kaggle las notas enteras: '8.0' o '8'."""
    muestra = leer_texto(DIR_KAGGLE / "user_ratings.csv", usecols=["Rating"], nrows=100_000)["Rating"]
    enteras = muestra[numero(muestra).mod(1).eq(0)]
    con_punto = enteras.str.contains(".", regex=False).mean() if len(enteras) else 1.0
    if con_punto >= 0.5:
        log.info("Formato de Rating: enteros con decimal ('8.0')")
        return lambda x: f"{x:.1f}"
    log.info("Formato de Rating: enteros sin decimal ('8')")
    return lambda x: f"{x:g}"


def generar_valoraciones(catalogo: pd.DataFrame, rng: np.random.Generator,
                         formato: callable) -> pd.DataFrame:
    ids = catalogo["BGGId"].to_numpy()
    n_cat = len(COLS_CAT)

    # Categorías normalizadas por juego; un juego sin categoría cuenta como neutro
    cats = catalogo[COLS_CAT].apply(numero).fillna(0).clip(0, 1).to_numpy(float, copy=True)
    sin_cat = cats.sum(1) == 0
    cats[sin_cat] = 1.0
    cats /= cats.sum(1, keepdims=True)

    peso = numero(catalogo["GameWeight"])
    peso = peso.where(peso > 0).fillna(peso[peso > 0].median()).to_numpy(float, copy=True)
    media = numero(catalogo["AvgRating"]).fillna(6.5).to_numpy()
    log_pop = POTENCIA_POP * np.log(numero(catalogo["NumUserRatings"]).clip(lower=1).to_numpy())

    usuarios, juegos, notas = [], [], []
    for u in range(N_USUARIOS):
        gusto = rng.dirichlet(np.full(n_cat, 0.4))       # concentrado en pocas categorías
        peso_pref = rng.normal(2.5, 0.7)
        sesgo = rng.normal(0, 0.6)                         # usuarios generosos y exigentes

        afinidad = np.log(n_cat * cats @ gusto + 0.1) - (peso - peso_pref) ** 2 / (2 * 0.8 ** 2)
        logit = log_pop + PESO_AFINIDAD * afinidad
        p = np.exp(logit - logit.max())
        p /= p.sum()

        n = int(np.clip(round(rng.lognormal(np.log(MEDIANA_VAL_USUARIO), 0.7)),
                        MIN_VAL_USUARIO, MAX_VAL_USUARIO))
        elegidos = rng.choice(len(ids), size=n, replace=False, p=p)

        # Afinidad centrada en los juegos elegidos: ordena las notas del usuario sin
        # subir su media (si no, elegir lo que te gusta inflaría todas las notas)
        z = (afinidad - afinidad.mean()) / afinidad.std()
        z_el = z[elegidos] - z[elegidos].mean()
        nota = media[elegidos] + sesgo + 0.9 * z_el + rng.normal(0, 0.9, n)
        nota = np.clip(nota, 1, 10)
        entera = rng.random(n) < 0.7                       # la mayoría pone notas enteras
        nota = np.where(entera, np.round(nota), np.round(nota * 2) / 2)

        usuarios += [f"ejemplo_{u + 1:05d}"] * n
        juegos += list(ids[elegidos])
        notas += [formato(x) for x in nota]

    df = pd.DataFrame({"BGGId": juegos, "Rating": notas, "Username": usuarios})
    return df.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)


# --------------------------------------------------------------------------- principal
def main() -> int:
    configurar_logging("crear_ejemplo.log", carpeta=ROOT / "logs")
    inicio = time.perf_counter()
    log.info("Ejemplo: inicio")

    if not MANIFIESTO.exists():
        log.error("No hay Bronze (%s). Ejecuta antes scripts/descargar_bronze.py",
                  MANIFIESTO.relative_to(ROOT))
        return 1
    version = json.loads(MANIFIESTO.read_text(encoding="utf-8"))["fuentes"]["kaggle_threnjen"]["version"]

    try:
        if EJEMPLO.exists():
            shutil.rmtree(EJEMPLO)
        rng = np.random.default_rng(SEMILLA)

        catalogo = seleccionar_catalogo(leer_texto(DIR_KAGGLE / "games.csv"))
        ids = set(catalogo["BGGId"])
        copiar_catalogo(ids)

        val = generar_valoraciones(catalogo, rng, formato_notas())
        escribir_texto(val, EJ_KAGGLE / "user_ratings.csv")

        por_usuario = val.groupby("Username").size()
        por_juego = val.groupby("BGGId").size()
        log.info("Valoraciones inventadas: %d de %d usuarios sobre %d juegos",
                 len(val), por_usuario.size, por_juego.size)
        log.info("Por usuario: mediana %d (mín. %d, máx. %d). Por juego: mediana %d (mín. %d)",
                 por_usuario.median(), por_usuario.min(), por_usuario.max(),
                 por_juego.median(), por_juego.min())
        if por_juego.size < len(ids):
            log.warning("%d juegos del catálogo no reciben ninguna valoración",
                        len(ids) - por_juego.size)

        ficheros = {r.relative_to(EJEMPLO).as_posix(): describir_csv(r)
                    for r in sorted(EJEMPLO.rglob("*.csv"))}
    except Exception:
        log.exception("Error al crear la muestra")
        return 1

    manifiesto = {
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo": "muestra sintética",
        "nota": ("Catálogo real copiado sin cambios desde Bronze; valoraciones y usuarios "
                 "inventados. Las columnas agregadas de games.csv son las reales de BGG."),
        "origen": {"dataset": KAGGLE_DATASET, "version": version, "fecha_ranking": FECHA_RANKING},
        "parametros": {"n_juegos": N_JUEGOS, "n_usuarios": N_USUARIOS,
                       "mediana_val_usuario": MEDIANA_VAL_USUARIO,
                       "min_val_usuario": MIN_VAL_USUARIO, "max_val_usuario": MAX_VAL_USUARIO,
                       "potencia_pop": POTENCIA_POP, "peso_afinidad": PESO_AFINIDAD,
                       "semilla": SEMILLA},
        "ficheros": ficheros,
    }
    (EJEMPLO / "_manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    tamano = sum(d["bytes"] for d in ficheros.values()) / 1e6
    log.info("Ejemplo: fin (%d ficheros, %.1f MB, %.0f s)", len(ficheros), tamano,
             time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())