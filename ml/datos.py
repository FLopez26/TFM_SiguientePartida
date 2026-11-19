"""Gold -> matrices y rasgos para el ML.

Lee Gold de los CSV de data/gold/ (por defecto) o del esquema gold de MySQL. Son los mismos
datos: pipeline.gold comprueba que MySQL tiene las mismas filas que cada CSV. Los CSV se leen
en segundos; desde MySQL, las 18 millones de valoraciones tardan más.

Las columnas de COLUMNAS_CON_FUGA (pipeline/gold.py) no llegan a leerse: de `juego` solo se
cargan las columnas que usa el ML, y el módulo falla al importarse si alguna tuviera fuga.

Partición, por usuario y al azar con semilla fija (ver notebooks/01_por_que_validacion.ipynb):
    test        20 % de las notas de cada usuario. Solo se usa en la medida final.
    validación  20 % de lo que queda. Sirve para elegir variante y número de vecinos.
    ajuste      el resto (64 %). Con esto se entrenan las variantes que se comparan.

Uso como comprobación (lee Gold, parte y construye los rasgos; deja el resumen en logs/ml.log):
    python -m ml.datos
    python -m ml.datos --origen mysql
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
from dotenv import load_dotenv
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from pipeline.gold import COLUMNAS_CON_FUGA, columnas
from pipeline.rutas import GOLD, LOGS, ROOT
from siguiente_partida import configurar_logging
from siguiente_partida.particion import reservar_por_usuario

log = logging.getLogger("ml.datos")

UMBRAL_RELEVANTE = 7.0      # una nota >= 7 en las reservadas cuenta como acierto
FRACCION_TEST = 0.2
FRACCION_VALIDACION = 0.2   # del 80 % que queda tras apartar el test
SEMILLA = 2026              # test; la validación usa SEMILLA + 1
AJUSTE, VALIDACION, TEST = 0, 1, 2
NOMBRE_PARTE = {AJUSTE: "ajuste", VALIDACION: "validacion", TEST: "test"}

# Rasgos permitidos para el modelo por contenido (además de etiquetas y descripción)
COLUMNAS_NUMERICAS = ["peso", "min_jugadores", "max_jugadores", "duracion_fabricante",
                      "edad_min_fabricante", "dependencia_idioma"]
CONJUNTOS_RASGOS = ("etiquetas", "etiquetas+numericos", "etiquetas+numericos+descripcion")
_PROHIBIDAS = set(COLUMNAS_NUMERICAS + ["descripcion"]) & set(COLUMNAS_CON_FUGA)
if _PROHIBIDAS:
    raise ImportError(f"Rasgos con fuga de información: {sorted(_PROHIBIDAS)}")

# Restos de HTML que pueden quedar en las descripciones de BGG
_PALABRAS_VACIAS = sorted(ENGLISH_STOP_WORDS | {"quot", "amp", "nbsp", "mdash", "ndash", "rsquo",
                                                 "lsquo", "ldquo", "rdquo", "hellip", "br"})


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


# --------------------------------------------------------------------------- lectura de Gold
def origen_gold() -> str:
    """'bronze' o 'ejemplo', según con qué se generó Gold (data/gold/_informe_gold.json)."""
    informe = GOLD / "_informe_gold.json"
    if not informe.exists():
        return "desconocido"
    return json.loads(informe.read_text(encoding="utf-8")).get("origen", "desconocido")


def _leer_csv(tabla: str, tipos: dict | None = None) -> pa.Table:
    ruta = GOLD / f"{tabla}.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}: ejecuta python -m pipeline.gold")
    return pv.read_csv(ruta, parse_options=pv.ParseOptions(newlines_in_values=True),
                       convert_options=pv.ConvertOptions(column_types=tipos or {}))


def _leer_mysql(sql: str, nombres: list[str], tam_lote: int = 500_000):
    """Filas de una consulta en bloques, sin cargar todo el resultado de golpe en Python."""
    import pymysql

    from pipeline.mysql import conectar

    try:
        con = conectar(cursorclass=pymysql.cursors.SSCursor)
    except pymysql.err.OperationalError as e:
        raise ValueError(f"No se puede conectar a MySQL ({e.args[-1]}). "
                         "Arráncalo con: docker compose up -d mysql") from None
    try:
        cur = con.cursor()
        cur.execute(sql)
        while filas := cur.fetchmany(tam_lote):
            yield pd.DataFrame(filas, columns=nombres)
    finally:
        con.close()


def leer_valoraciones(origen: str = "csv") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(id_usuario, bgg_id, nota) ordenados por usuario y juego, para que la partición no
    dependa del orden de lectura."""
    inicio = time.perf_counter()
    if origen == "csv":
        t = _leer_csv("valoracion", {"id_usuario": pa.int32(), "bgg_id": pa.int32(),
                                     "nota": pa.float32()})
        u, j, n = (t.column(c).to_numpy() for c in ("id_usuario", "bgg_id", "nota"))
    elif origen == "mysql":
        partes = list(_leer_mysql("SELECT id_usuario, bgg_id, nota FROM gold.valoracion",
                                  ["id_usuario", "bgg_id", "nota"]))
        df = pd.concat(partes, ignore_index=True)
        u, j, n = (df["id_usuario"].to_numpy(np.int32), df["bgg_id"].to_numpy(np.int32),
                   df["nota"].to_numpy(np.float32))
    else:
        raise ValueError("origen tiene que ser 'csv' o 'mysql'")
    orden = np.lexsort((j, u))
    log.info("Valoraciones leídas (%s): %s (%.0f s)", origen, _miles(len(u)),
             time.perf_counter() - inicio)
    return u[orden], j[orden], n[orden]


def leer_catalogo(origen: str = "csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(juego con solo las columnas que usa el ML, etiquetas en formato largo: bgg_id, tipo, nombre).

    Solo se leen columnas permitidas: las de COLUMNAS_CON_FUGA no llegan a cargarse.
    """
    permitidas = ["bgg_id", "nombre", "descripcion", *COLUMNAS_NUMERICAS]
    if not set(permitidas) <= set(columnas("juego")):
        raise ValueError("Alguna columna del ML no está en la definición de gold.juego")
    if origen == "csv":
        tipos = {"bgg_id": pa.int32(), "nombre": pa.string(), "descripcion": pa.string(),
                 **{c: pa.float64() for c in COLUMNAS_NUMERICAS}}
        ruta = GOLD / "juego.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}: ejecuta python -m pipeline.gold")
        juego = pv.read_csv(ruta, parse_options=pv.ParseOptions(newlines_in_values=True),
                            convert_options=pv.ConvertOptions(include_columns=permitidas,
                                                              column_types=tipos)).to_pandas()
        etiqueta = _leer_csv("etiqueta").to_pandas()
        je = _leer_csv("juego_etiqueta").to_pandas()
        etiquetas = je.merge(etiqueta, on="id_etiqueta")[["bgg_id", "tipo", "nombre"]]
    elif origen == "mysql":
        juego = pd.concat(_leer_mysql(f"SELECT {', '.join(permitidas)} FROM gold.juego", permitidas))
        etiquetas = pd.concat(_leer_mysql(
            "SELECT je.bgg_id, e.tipo, e.nombre FROM gold.juego_etiqueta je "
            "JOIN gold.etiqueta e USING (id_etiqueta)", ["bgg_id", "tipo", "nombre"]))
    else:
        raise ValueError("origen tiene que ser 'csv' o 'mysql'")
    if set(juego.columns) & set(COLUMNAS_CON_FUGA):
        raise ValueError("Se ha leído una columna con fuga de información")
    log.info("Catálogo leído (%s): %d juegos, %d etiquetas asignadas; sin las %d columnas con fuga",
             origen, len(juego), len(etiquetas), len(COLUMNAS_CON_FUGA))
    return juego.reset_index(drop=True), etiquetas.reset_index(drop=True)


# --------------------------------------------------------------------------- matrices y partición
@dataclass
class Valoraciones:
    """Valoraciones como índices de matriz, con la parte (ajuste, validación o test) de cada una."""

    usuarios: np.ndarray   # id_usuario de cada fila de las matrices
    juegos: np.ndarray     # bgg_id de cada columna
    fila: np.ndarray
    col: np.ndarray
    nota: np.ndarray
    parte: np.ndarray

    @property
    def forma(self) -> tuple[int, int]:
        return len(self.usuarios), len(self.juegos)

    def matriz(self, *partes: int) -> sparse.csr_matrix:
        """Notas de las partes pedidas (usuarios x juegos); el resto, 0."""
        m = np.isin(self.parte, partes)
        return sparse.csr_matrix((self.nota[m], (self.fila[m], self.col[m])), shape=self.forma)

    def relevantes(self, parte: int) -> sparse.csr_matrix:
        """1 donde la nota reservada en esa parte es >= UMBRAL_RELEVANTE."""
        m = (self.parte == parte) & (self.nota >= UMBRAL_RELEVANTE)
        return sparse.csr_matrix((np.ones(m.sum(), np.float32), (self.fila[m], self.col[m])),
                                 shape=self.forma)

    def resumen(self) -> dict:
        salida = {"valoraciones": int(len(self.nota)), "usuarios": self.forma[0],
                  "juegos": self.forma[1]}
        for p, nombre in NOMBRE_PARTE.items():
            salida[f"valoraciones_{nombre}"] = int((self.parte == p).sum())
        return salida


def partir(fila: np.ndarray) -> np.ndarray:
    """Parte de cada valoración: primero el test sobre todo, luego la validación sobre el resto."""
    parte = np.full(len(fila), AJUSTE, dtype=np.int8)
    test = reservar_por_usuario(fila, FRACCION_TEST, SEMILLA)
    parte[test] = TEST
    resto = np.flatnonzero(~test)
    parte[resto[reservar_por_usuario(fila[resto], FRACCION_VALIDACION, SEMILLA + 1)]] = VALIDACION
    return parte


def preparar_valoraciones(origen: str = "csv") -> Valoraciones:
    u, j, nota = leer_valoraciones(origen)
    usuarios, fila = np.unique(u, return_inverse=True)
    juegos, col = np.unique(j, return_inverse=True)
    datos = Valoraciones(usuarios, juegos, fila.astype(np.int32), col.astype(np.int32), nota,
                         partir(fila))
    r = datos.resumen()
    log.info("Matriz: %s usuarios x %s juegos", _miles(r["usuarios"]), _miles(r["juegos"]))
    for p, nombre in NOMBRE_PARTE.items():
        log.info("Partición: %-10s %12s valoraciones (%.1f %%)", nombre,
                 _miles(r[f"valoraciones_{nombre}"]),
                 100 * r[f"valoraciones_{nombre}"] / r["valoraciones"])
    # Cada usuario tiene que tener notas en las tres partes (Gold garantiza 10 o más)
    por_parte = np.zeros((len(usuarios), 3), dtype=np.int64)
    np.add.at(por_parte, (datos.fila, datos.parte), 1)
    sin_alguna = int((por_parte == 0).any(axis=1).sum())
    if sin_alguna:
        log.warning("Partición: %d usuarios sin notas en alguna parte", sin_alguna)
    else:
        log.info("Partición: todos los usuarios tienen notas en ajuste, validación y test")
    return datos


# --------------------------------------------------------------------------- rasgos de contenido
def rasgos_contenido(juego: pd.DataFrame, etiquetas: pd.DataFrame, juegos: np.ndarray,
                     conjunto: str) -> sparse.csr_matrix:
    """Matriz juegos x rasgos, en el orden de `juegos` (bgg_id de las columnas de R).

    Cada bloque se normaliza por fila para que pesen lo mismo:
        etiquetas    categorías, subcategorías, mecánicas y temas, con peso IDF (las raras pesan más)
        numericos    COLUMNAS_NUMERICAS estandarizadas; los nulos quedan en la media
        descripcion  TF-IDF de la descripción (inglés)
    """
    if conjunto not in CONJUNTOS_RASGOS:
        raise ValueError(f"conjunto desconocido: {conjunto} (usa uno de {CONJUNTOS_RASGOS})")
    inicio = time.perf_counter()
    catalogo = juego.set_index("bgg_id")
    faltan = np.setdiff1d(juegos, catalogo.index.to_numpy())
    if len(faltan):
        raise ValueError(f"{len(faltan)} juegos con notas no están en el catálogo")
    n = len(juegos)
    pos = pd.Series(np.arange(n), index=juegos)

    e = etiquetas[etiquetas["bgg_id"].isin(juegos)]
    codigo, _ = pd.factorize(e["tipo"].astype(str) + ":" + e["nombre"].astype(str))
    T = sparse.csr_matrix((np.ones(len(e), np.float32), (pos[e["bgg_id"]].to_numpy(), codigo)),
                          shape=(n, codigo.max() + 1 if len(codigo) else 0))
    T.data[:] = 1.0  # por si una etiqueta se repite
    idf = np.log((1 + n) / (1 + np.diff(T.tocsc().indptr))) + 1
    bloques = [normalize(T @ sparse.diags(idf.astype(np.float32)))]

    if "numericos" in conjunto:
        X = catalogo.loc[juegos, COLUMNAS_NUMERICAS].astype(float)
        X["duracion_fabricante"] = np.log1p(X["duracion_fabricante"].clip(lower=0))
        X["max_jugadores"] = X["max_jugadores"].clip(upper=12)
        z = ((X - X.mean()) / X.std(ddof=0).replace(0, 1)).fillna(0.0)
        bloques.append(normalize(sparse.csr_matrix(z.to_numpy(np.float32))))

    if "descripcion" in conjunto:
        texto = catalogo.loc[juegos, "descripcion"].fillna("").astype(str)
        tfidf = TfidfVectorizer(max_features=20_000, min_df=5, max_df=0.5, sublinear_tf=True,
                                stop_words=_PALABRAS_VACIAS, dtype=np.float32,
                                token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
        bloques.append(tfidf.fit_transform(texto))

    F = sparse.hstack(bloques, format="csr", dtype=np.float32)
    log.info("Rasgos %-32s %6d juegos x %6d rasgos (%.0f s)", conjunto, *F.shape,
             time.perf_counter() - inicio)
    return F


# --------------------------------------------------------------------------- comprobación
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", choices=["csv", "mysql"], default="csv")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    configurar_logging("ml.log", carpeta=LOGS)
    inicio = time.perf_counter()
    log.info("Datos ML: inicio (Gold desde %s, generado con %s)", args.origen, origen_gold())
    try:
        datos = preparar_valoraciones(args.origen)
        juego, etiquetas = leer_catalogo(args.origen)
        for conjunto in CONJUNTOS_RASGOS:
            rasgos_contenido(juego, etiquetas, datos.juegos, conjunto)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado preparando los datos de ML")
        return 1
    log.info("Datos ML: fin (%.0f s)", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())