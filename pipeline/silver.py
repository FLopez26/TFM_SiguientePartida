"""Bronze -> Silver: tipos, nulos, duplicados, formato largo y seudonimización.

Entrada: data/bronze/ si la descarga está completa; si no, data/ejemplo/ (misma estructura).
Salida (data/silver/, Parquet):
    juegos.parquet         una fila por juego, columnas en español y con su tipo
    etiquetas.parquet      formato largo (bgg_id, tipo, etiqueta): categorías, subcategorías,
                           mecánicas y temas
    valoraciones.parquet   (usuario, bgg_id, nota); usuario = HMAC-SHA256 del nombre
    ranking_2026.parquet   foto del ranking de BGG del 2026-11-12
    _informe_calidad.json  filas de entrada y salida y número de correcciones de cada regla

El nombre de usuario real no sale de esta capa: se sustituye por su seudónimo al leerlo.

Uso (desde la raíz del repositorio):
    python -m pipeline.silver                    # Bronze si existe; si no, la muestra
    python -m pipeline.silver --origen ejemplo   # fuerza la muestra
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from dotenv import load_dotenv

from pipeline.rutas import (FECHA_RANKING, LOGS, MANIFIESTO, ROOT, SILVER, SUB_KAGGLE,
                            SUB_RANKING, resolver_origen)
from siguiente_partida import configurar_logging, es_seudonimo, seudonimizar

log = logging.getLogger("silver")

# --------------------------------------------------------------------------- reglas
NOTA_MIN, NOTA_MAX = 1.0, 10.0          # escala de valoración de BGG
SAL_EJEMPLO = "muestra-sintetica-sin-datos-reales"  # solo para data/ejemplo/ (usuarios inventados)

RENOMBRE_JUEGOS = {
    "BGGId": "bgg_id", "Name": "nombre", "Description": "descripcion",
    "YearPublished": "anio", "GameWeight": "peso",
    "AvgRating": "nota_media_bgg", "BayesAvgRating": "nota_bayes_bgg", "StdDev": "desv_nota_bgg",
    "MinPlayers": "min_jugadores", "MaxPlayers": "max_jugadores",
    "ComAgeRec": "edad_rec_comunidad", "LanguageEase": "dependencia_idioma",
    "BestPlayers": "mejor_n_jugadores", "GoodPlayers": "buenos_n_jugadores",
    "NumOwned": "n_propietarios", "NumWant": "n_quieren", "NumWish": "n_deseados",
    "NumWeightVotes": "n_votos_peso", "MfgPlaytime": "duracion_fabricante",
    "ComMinPlaytime": "duracion_min_comunidad", "ComMaxPlaytime": "duracion_max_comunidad",
    "MfgAgeRec": "edad_min_fabricante", "NumUserRatings": "n_valoraciones_bgg",
    "NumComments": "n_comentarios", "NumAlternates": "n_nombres_alternativos",
    "NumExpansions": "n_expansiones", "NumImplementations": "n_reimplementaciones",
    "IsReimplementation": "es_reimplementacion", "Family": "familia",
    "Kickstarted": "kickstarter", "ImagePath": "imagen_url",
    "Rank:boardgame": "rank_general", "Rank:strategygames": "rank_estrategia",
    "Rank:abstracts": "rank_abstractos", "Rank:familygames": "rank_familiares",
    "Rank:thematic": "rank_tematicos", "Rank:cgs": "rank_cartas_coleccionables",
    "Rank:wargames": "rank_wargames", "Rank:partygames": "rank_party",
    "Rank:childrensgames": "rank_infantiles",
}
COLS_CAT = ["Cat:Thematic", "Cat:Strategy", "Cat:War", "Cat:Family",
            "Cat:CGS", "Cat:Abstract", "Cat:Party", "Cat:Childrens"]

TEXTO = ["nombre", "descripcion", "familia", "imagen_url"]
DECIMALES = ["peso", "nota_media_bgg", "nota_bayes_bgg", "desv_nota_bgg",
             "edad_rec_comunidad", "dependencia_idioma"]
ENTEROS = ["anio", "min_jugadores", "max_jugadores", "mejor_n_jugadores",
           "n_propietarios", "n_quieren", "n_deseados", "n_votos_peso",
           "duracion_fabricante", "duracion_min_comunidad", "duracion_max_comunidad",
           "edad_min_fabricante", "n_valoraciones_bgg", "n_comentarios",
           "n_nombres_alternativos", "n_expansiones", "n_reimplementaciones"]
BOOLEANOS = ["es_reimplementacion", "kickstarter"]
RANKS = [c for c in RENOMBRE_JUEGOS.values() if c.startswith("rank_")]

# En BGG un 0 en estas columnas significa "sin dato" (nadie votó o no se informó)
CERO_ES_NULO = ["anio", "peso", "min_jugadores", "max_jugadores", "mejor_n_jugadores",
                "edad_rec_comunidad", "dependencia_idioma", "duracion_fabricante",
                "duracion_min_comunidad", "duracion_max_comunidad", "edad_min_fabricante"]
# threnjen marca "sin ranking" con el número de juegos del dataset + 1 (21.925 + 1).
# Valor fijo y no len(juegos): en la muestra solo hay 300 juegos y sus rankings son reales.
RANK_SIN_RANKING = 21926
# Pares (mínimo, máximo) que deben ser coherentes
PARES_MIN_MAX = [("min_jugadores", "max_jugadores"),
                 ("duracion_min_comunidad", "duracion_max_comunidad")]

RENOMBRE_RANKING = {"ID": "bgg_id", "Name": "nombre", "Year": "anio", "Rank": "rank",
                    "Average": "nota_media", "Bayes average": "nota_bayes",
                    "Users rated": "n_valoraciones", "URL": "url", "Thumbnail": "miniatura_url"}


class Informe:
    """Acumula filas y correcciones por tabla y las registra en el log."""

    def __init__(self) -> None:
        self.datos: dict[str, dict] = {}

    def tabla(self, nombre: str, filas_entrada: int) -> None:
        self.datos[nombre] = {"filas_entrada": filas_entrada, "correcciones": {}}
        log.info("%s: %s filas de entrada", nombre, _miles(filas_entrada))

    def correccion(self, tabla: str, regla: str, n: int) -> None:
        self.datos[tabla]["correcciones"][regla] = int(n)
        if n:
            log.info("%s: %-55s %s", tabla, regla, _miles(n))

    def salida(self, tabla: str, filas: int) -> None:
        self.datos[tabla]["filas_salida"] = int(filas)
        log.info("%s: %s filas de salida", tabla, _miles(filas))


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def escribir_parquet(df: pd.DataFrame, ruta: Path) -> None:
    """Escribe con pyarrow. Las categóricas se guardan como texto: Parquet ya comprime los
    valores repetidos, y una categórica con cientos de miles de valores repetiría su
    diccionario entero en cada bloque del fichero."""
    tabla = pa.Table.from_pandas(df, preserve_index=False)
    for i, campo in enumerate(tabla.schema):
        if pa.types.is_dictionary(campo.type):
            tabla = tabla.set_column(i, campo.name, tabla.column(i).cast(campo.type.value_type))
    pq.write_table(tabla, ruta)


def leer_texto(ruta: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, **kw)


def numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.replace("", np.nan), errors="coerce")


# --------------------------------------------------------------------------- juegos
def limpiar_juegos(games: pd.DataFrame, inf: Informe) -> pd.DataFrame:
    t = "juegos"
    inf.tabla(t, len(games))
    df = games.rename(columns=RENOMBRE_JUEGOS)

    df["bgg_id"] = numero(df["bgg_id"])
    inf.correccion(t, "bgg_id no numérico: fila descartada", df["bgg_id"].isna().sum())
    df = df[df["bgg_id"].notna()]
    dup = df["bgg_id"].duplicated()
    inf.correccion(t, "bgg_id repetido: se queda la primera fila", dup.sum())
    df = df[~dup].copy()
    df["bgg_id"] = df["bgg_id"].astype("int32")

    for c in TEXTO:
        df[c] = df[c].str.strip().replace("", pd.NA)
    inf.correccion(t, "nombre vacío (se mantiene el juego)", df["nombre"].isna().sum())

    for c in DECIMALES:
        df[c] = numero(df[c]).astype("float64")
    for c in ENTEROS:
        df[c] = numero(df[c]).round().astype("Int64")
    for c in BOOLEANOS:
        df[c] = numero(df[c]).fillna(0).astype(bool)

    for c in CERO_ES_NULO:
        ceros = df[c].eq(0).fillna(False)
        inf.correccion(t, f"{c} = 0 -> nulo (sin dato)", ceros.sum())
        df.loc[ceros, c] = pd.NA if df[c].dtype == "Int64" else np.nan

    for cmin, cmax in PARES_MIN_MAX:
        mal = (df[cmax] < df[cmin]).fillna(False)
        inf.correccion(t, f"{cmax} < {cmin} -> {cmax} nulo", mal.sum())
        df.loc[mal, cmax] = pd.NA

    for c in RANKS:
        r = numero(df[c])
        sin_rank = r.ge(RANK_SIN_RANKING) | r.le(0)
        inf.correccion(t, f"{c} sin ranking -> nulo", sin_rank.fillna(False).sum())
        df[c] = r.mask(sin_rank).round().astype("Int64")

    def a_lista(v: str) -> list[int] | None:
        """"['2', '3', '4+']" -> [2, 3, 4]: "4+" (4 o más) cuenta como 4. None si no se entiende."""
        try:
            valores = ast.literal_eval(v) if v else []
            return sorted({int(str(x).rstrip("+")) for x in valores} - {0})
        except (ValueError, SyntaxError, TypeError):
            return None
    listas = df["buenos_n_jugadores"].map(a_lista)
    inf.correccion(t, "buenos_n_jugadores ilegible -> nulo", listas.isna().sum())
    vacias = listas.map(lambda x: x == [])
    inf.correccion(t, "buenos_n_jugadores vacío -> nulo (sin dato)", vacias.sum())
    df["buenos_n_jugadores"] = listas.map(lambda x: x if x else None)

    df = df[list(RENOMBRE_JUEGOS.values())].sort_values("bgg_id").reset_index(drop=True)
    inf.salida(t, len(df))
    return df


# --------------------------------------------------------------------------- etiquetas
def a_largo(ancho: pd.DataFrame, tipo: str, ids_validos: set[int],
            quitar_prefijo: str = "") -> pd.DataFrame:
    """Tabla 0/1 (una columna por etiqueta) -> filas (bgg_id, tipo, etiqueta) donde vale > 0."""
    ancho = ancho.copy()
    ancho["BGGId"] = numero(ancho["BGGId"])
    ancho = ancho[ancho["BGGId"].isin(ids_validos)].drop_duplicates("BGGId")
    largo = ancho.melt(id_vars="BGGId", var_name="etiqueta", value_name="valor")
    largo = largo[numero(largo["valor"]).gt(0)]
    etiqueta = largo["etiqueta"].str.strip()
    if quitar_prefijo:
        etiqueta = etiqueta.str.removeprefix(quitar_prefijo)
    return pd.DataFrame({"bgg_id": largo["BGGId"].astype("int32"), "tipo": tipo,
                         "etiqueta": etiqueta})


def construir_etiquetas(origen: Path, games: pd.DataFrame, ids: set[int],
                        inf: Informe) -> pd.DataFrame:
    t = "etiquetas"
    k = origen / SUB_KAGGLE
    partes = [
        a_largo(games[["BGGId", *COLS_CAT]], "categoria", ids, quitar_prefijo="Cat:"),
        a_largo(leer_texto(k / "subcategories.csv"), "subcategoria", ids),
        a_largo(leer_texto(k / "mechanics.csv"), "mecanica", ids),
        # threnjen pone "Theme_" delante de los temas poco frecuentes
        a_largo(leer_texto(k / "themes.csv"), "tema", ids, quitar_prefijo="Theme_"),
    ]
    df = pd.concat(partes, ignore_index=True)
    inf.tabla(t, len(df))
    dup = df.duplicated()
    inf.correccion(t, "etiqueta repetida en el mismo juego", dup.sum())
    df = df[~dup].sort_values(["bgg_id", "tipo", "etiqueta"]).reset_index(drop=True)
    for tipo, n in df.groupby("tipo").size().items():
        log.info("%s: %-12s %s filas, %d etiquetas distintas", t, tipo, _miles(n),
                 df.loc[df["tipo"] == tipo, "etiqueta"].nunique())
    sin = len(ids - set(df["bgg_id"]))
    inf.correccion(t, "juegos sin ninguna etiqueta (informativo)", sin)
    inf.salida(t, len(df))
    return df


# --------------------------------------------------------------------------- valoraciones
def _desde_categorias(col: pd.Series, convertir) -> np.ndarray:
    """Aplica `convertir` a los valores distintos de una columna categórica y lo expande a
    todas las filas. Con ~19 millones de filas y pocos valores distintos ahorra mucha memoria."""
    return convertir(pd.Series(col.cat.categories)).to_numpy("float64")[col.cat.codes.to_numpy()]


def construir_valoraciones(origen: Path, ids: set[int], sal: str, inf: Informe) -> pd.DataFrame:
    """Con ~19 millones de filas se trabaja con arrays de numpy y una sola ordenación,
    sin copias intermedias del DataFrame."""
    t = "valoraciones"
    cols = ("BGGId", "Rating", "Username")
    # Leídas como diccionario (categóricas): cada texto distinto se guarda una sola vez
    tabla = pv.read_csv(origen / SUB_KAGGLE / "user_ratings.csv",
                        convert_options=pv.ConvertOptions(
                            column_types={c: pa.dictionary(pa.int32(), pa.string()) for c in cols},
                            strings_can_be_null=False))
    crudo = tabla.to_pandas()
    del tabla
    pa.default_memory_pool().release_unused()   # devuelve al sistema la memoria de la lectura
    n = len(crudo)
    inf.tabla(t, n)

    # Seudonimizar los nombres únicos (cientos de miles), no las filas (millones).
    # El código de usuario sigue el orden alfabético del seudónimo: así, ordenar por
    # (código, bgg_id) deja la tabla ordenada por (usuario, bgg_id).
    nombres = pd.Series(crudo["Username"].cat.categories).str.strip().replace("", np.nan)
    cod_nombre, nombres_unicos = pd.factorize(nombres)          # -1 = nombre vacío
    seudos = np.array(seudonimizar(nombres_unicos.astype(str), sal))
    del nombres_unicos
    orden = np.argsort(seudos)
    seudos = seudos[orden]
    rango = np.empty(len(orden), dtype=np.int64)
    rango[orden] = np.arange(len(orden))
    cod_nombre = np.where(cod_nombre >= 0, rango[np.maximum(cod_nombre, 0)], -1)
    u = cod_nombre[crudo["Username"].cat.codes.to_numpy()]
    bgg = _desde_categorias(crudo["BGGId"], numero)
    nota = _desde_categorias(crudo["Rating"], numero)
    del crudo

    # Reglas de descarte: cada fila se cuenta en la primera regla que incumple
    reglas = [
        ("usuario vacío: descartada", u < 0),
        ("bgg_id no numérico: descartada", np.isnan(bgg)),
        ("nota vacía o no numérica: descartada", np.isnan(nota)),
        (f"nota fuera de [{NOTA_MIN:g}, {NOTA_MAX:g}]: descartada",
         (nota < NOTA_MIN) | (nota > NOTA_MAX)),
        ("juego que no está en juegos: descartada",
         ~np.isin(bgg, np.fromiter(ids, dtype="float64"))),
    ]
    excluida = np.zeros(n, dtype=bool)
    for regla, mascara in reglas:
        inf.correccion(t, regla, (mascara & ~excluida).sum())
        excluida |= mascara
    del reglas, mascara
    seguir = ~excluida
    u, bgg, nota = u[seguir], bgg[seguir].astype(np.int64), nota[seguir].astype(np.float32)
    del excluida, seguir

    # Ordenar por usuario, juego, nota y comparar cada fila con la anterior
    clave = (u << 32) | bgg
    del u, bgg
    orden = np.lexsort((nota, clave))
    clave, nota = clave[orden], nota[orden]
    del orden
    exacta = np.r_[False, (clave[1:] == clave[:-1]) & (nota[1:] == nota[:-1])]
    inf.correccion(t, "fila repetida exacta: descartada", exacta.sum())
    clave, nota = clave[~exacta], nota[~exacta]
    del exacta

    inicio = np.r_[True, clave[1:] != clave[:-1]]
    grupo = np.cumsum(inicio) - 1
    cuantas = np.bincount(grupo)
    inf.correccion(t, "mismo usuario y juego con notas distintas: se promedian (pares)",
                   (cuantas > 1).sum())
    nota = (np.bincount(grupo, weights=nota) / cuantas).astype(np.float32)
    clave = clave[inicio]
    del inicio, grupo, cuantas

    usuario = pd.Categorical.from_codes((clave >> 32).astype(np.int32), categories=seudos)
    df = pd.DataFrame({"usuario": usuario.remove_unused_categories(),
                       "bgg_id": (clave & 0xFFFFFFFF).astype(np.int32), "nota": nota})
    inf.salida(t, len(df))
    log.info("%s: %s usuarios y %s juegos con al menos una valoración", t,
             _miles(df["usuario"].cat.categories.size), _miles(df["bgg_id"].nunique()))
    return df


# --------------------------------------------------------------------------- ranking
def limpiar_ranking(origen: Path, inf: Informe) -> pd.DataFrame:
    t = "ranking_2026"
    crudo = leer_texto(origen / SUB_RANKING / f"{FECHA_RANKING}.csv")
    inf.tabla(t, len(crudo))
    df = crudo.rename(columns=RENOMBRE_RANKING)[list(RENOMBRE_RANKING.values())]
    for c in ["bgg_id", "anio", "rank", "n_valoraciones"]:
        df[c] = numero(df[c]).round().astype("Int64")
    for c in ["nota_media", "nota_bayes"]:
        df[c] = numero(df[c]).astype("float64")
    for c in ["nombre", "url", "miniatura_url"]:
        df[c] = df[c].str.strip().replace("", pd.NA)

    sin_id = df["bgg_id"].isna()
    inf.correccion(t, "bgg_id no numérico: fila descartada", sin_id.sum())
    df = df[~sin_id]
    # Los repetidos tienen el mismo rank; a veces solo cambia nota_bayes en decimales
    dup = df["bgg_id"].duplicated()
    con_bayes_distinta = (df[df["bgg_id"].duplicated(keep=False)]
                          .groupby("bgg_id")["nota_bayes"].nunique().gt(1).sum())
    inf.correccion(t, "bgg_id repetido: se queda la primera fila", dup.sum())
    inf.correccion(t, "...de ellos con nota_bayes distinta (informativo)", con_bayes_distinta)
    df = df[~dup]
    sin_rank = df["rank"].isna() | df["rank"].le(0)
    inf.correccion(t, "rank vacío o <= 0 -> nulo", sin_rank.sum())
    df.loc[sin_rank, "rank"] = pd.NA
    inf.correccion(t, "anio = 0 -> nulo", df["anio"].eq(0).fillna(False).sum())
    df.loc[df["anio"].eq(0).fillna(False), "anio"] = pd.NA

    df = df.astype({"bgg_id": "int32"}).sort_values("bgg_id").reset_index(drop=True)
    inf.salida(t, len(df))
    return df


# --------------------------------------------------------------------------- validación final
def validar(juegos: pd.DataFrame, etiquetas: pd.DataFrame, val: pd.DataFrame) -> None:
    errores = []
    if not juegos["bgg_id"].is_unique:
        errores.append("juegos.bgg_id no es único")
    if not etiquetas["bgg_id"].isin(juegos["bgg_id"]).all():
        errores.append("etiquetas con bgg_id que no está en juegos")
    if not val["bgg_id"].isin(juegos["bgg_id"]).all():
        errores.append("valoraciones con bgg_id que no está en juegos")
    # valoraciones está ordenada por (usuario, bgg_id): un repetido quedaría junto al anterior
    cod = val["usuario"].cat.codes.to_numpy()
    bgg = val["bgg_id"].to_numpy()
    if ((cod[1:] == cod[:-1]) & (bgg[1:] == bgg[:-1])).any():
        errores.append("valoraciones con (usuario, bgg_id) repetido")
    if not val["nota"].between(NOTA_MIN, NOTA_MAX).all():
        errores.append("valoraciones con nota fuera de rango")
    if not all(es_seudonimo(u) for u in val["usuario"].cat.categories):
        errores.append("valoraciones con un usuario que no es un seudónimo")
    for e in errores:
        log.error("Validación: %s", e)
    if errores:
        raise ValueError("Silver no supera la validación")
    log.info("Validación: claves únicas, integridad referencial, notas en rango y "
             "usuarios seudonimizados")


# --------------------------------------------------------------------------- principal
def obtener_sal(nombre_origen: str) -> str:
    sal = os.environ.get("SAL_SEUDONIMOS", "")
    if sal:
        return sal
    if nombre_origen == "ejemplo":
        log.warning("Sin SAL_SEUDONIMOS en .env: la muestra usa una sal pública "
                    "(sus usuarios son inventados)")
        return SAL_EJEMPLO
    raise ValueError("Falta SAL_SEUDONIMOS en .env (obligatoria con datos reales). Genera una con:"
                     " python -c \"import secrets; print(secrets.token_hex(32))\"")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", choices=["auto", "bronze", "ejemplo"], default="auto")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    configurar_logging("silver.log", carpeta=LOGS)
    inicio = time.perf_counter()

    try:
        nombre_origen, origen = resolver_origen(args.origen)
        log.info("Silver: inicio, origen %s (%s)", nombre_origen, origen.relative_to(ROOT))
        sal = obtener_sal(nombre_origen)
        inf = Informe()

        games = leer_texto(origen / SUB_KAGGLE / "games.csv")
        juegos = limpiar_juegos(games, inf)
        ids = set(juegos["bgg_id"].tolist())
        etiquetas = construir_etiquetas(origen, games, ids, inf)
        del games
        valoraciones = construir_valoraciones(origen, ids, sal, inf)
        ranking = limpiar_ranking(origen, inf)
        validar(juegos, etiquetas, valoraciones)

        SILVER.mkdir(parents=True, exist_ok=True)
        for nombre, df in [("juegos", juegos), ("etiquetas", etiquetas),
                           ("valoraciones", valoraciones), ("ranking_2026", ranking)]:
            escribir_parquet(df, SILVER / f"{nombre}.parquet")
            log.info("Escrito %s.parquet (%.1f MB)", nombre,
                     (SILVER / f"{nombre}.parquet").stat().st_size / 1e6)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado en Silver")
        return 1

    manifiesto_origen = json.loads((origen / MANIFIESTO).read_text(encoding="utf-8"))
    informe = {
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origen": nombre_origen,
        "manifiesto_origen_creado": manifiesto_origen.get("creado"),
        "tablas": inf.datos,
    }
    (SILVER / "_informe_calidad.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Silver: fin (%.0f s)", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())