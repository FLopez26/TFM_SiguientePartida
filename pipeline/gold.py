"""Silver -> Gold: modelo relacional con claves, en CSV (data/gold/) y en MySQL (esquema gold).

Tablas:
    juego             catálogo completo, una fila por juego (sin filtrar)
    etiqueta          categorías, subcategorías, mecánicas y temas, con id propio
    juego_etiqueta    relación juego-etiqueta (N:M)
    juego_jugadores   números de jugadores recomendados por la comunidad (N:M)
    usuario           usuarios de BGG seudonimizados que pasan el filtro, con id entero
    valoracion        notas de esos usuarios, filtradas para entrenar y evaluar
    ranking_2026      foto del ranking de BGG del 2026-11-12 (solo para la prueba aparte)

Filtro de valoraciones: usuarios con >= MIN_VAL_USUARIO y juegos con >= MIN_VAL_JUEGO,
repetido hasta que no cambia. Así cada usuario tiene notas suficientes para reservar el 20 % en la evaluación.

La carga en MySQL borra y vuelve a crear las tablas de gold (se regenera entero). No toca el
esquema app: solo avisa si hay juegos guardados por usuarios que ya no están en el catálogo.

Uso (desde la raíz del repositorio, con Silver ya generado):
    python -m pipeline.gold              # CSV y carga en MySQL
    python -m pipeline.gold --sin-mysql  # solo los CSV
"""
from __future__ import annotations

import argparse
import json
import logging
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

from pipeline.mysql import conectar
from pipeline.rutas import GOLD, LOGS, ROOT, SILVER
from siguiente_partida import configurar_logging

log = logging.getLogger("gold")

MIN_VAL_USUARIO = 10
MIN_VAL_JUEGO = 5

FUGA = ("COMMENT 'FUGA: calculada por BGG con todas las valoraciones, incluidas las de test; "
        "no usar como rasgo de un modelo'")
FUGA_POP = ("COMMENT 'FUGA de popularidad: la generan los mismos usuarios, también los de test; "
            "no usar como rasgo de un modelo'")
RANKS = ["rank_general", "rank_estrategia", "rank_abstractos", "rank_familiares",
         "rank_tematicos", "rank_cartas_coleccionables", "rank_wargames", "rank_party",
         "rank_infantiles"]
# Columnas de juego que no pueden entrar en ningún modelo (las importa la fase de ML).
# Siguen en la tabla porque la ficha del juego las muestra.
COLUMNAS_CON_FUGA = [
    "nota_media_bgg", "nota_bayes_bgg", "desv_nota_bgg", "n_valoraciones_bgg",  # notas
    *RANKS,                                     # BGG los calcula con la nota bayesiana
    "n_propietarios", "n_quieren", "n_deseados", "n_comentarios", "n_votos_peso",  # contadores
]

# --------------------------------------------------------------------------- definición de tablas
# El orden de las columnas es el de los CSV. Las claves primarias van en el CREATE; los
# índices secundarios y las claves foráneas se añaden después de cargar (carga más rápida).
TABLAS: dict[str, str] = {
        "juego": f"""
        bgg_id                  INT NOT NULL,
        nombre                  VARCHAR(500) NULL,
        descripcion             MEDIUMTEXT NULL,
        anio                    SMALLINT NULL COMMENT 'Año de publicación; negativo = a. C.',
        peso                    DOUBLE NULL COMMENT 'Complejidad 1-5 votada por la comunidad',
        nota_media_bgg          DOUBLE NULL {FUGA},
        nota_bayes_bgg          DOUBLE NULL {FUGA},
        desv_nota_bgg           DOUBLE NULL {FUGA},
        min_jugadores           INT NULL,
        max_jugadores           INT NULL,
        edad_rec_comunidad      DOUBLE NULL,
        dependencia_idioma      DOUBLE NULL,
        mejor_n_jugadores       INT NULL COMMENT 'Nulo en ~91 % de los juegos',
        n_propietarios          INT NULL {FUGA_POP},
        n_quieren               INT NULL {FUGA_POP},
        n_deseados              INT NULL {FUGA_POP},
        n_votos_peso            INT NULL {FUGA_POP},
        duracion_fabricante     INT NULL COMMENT 'Minutos',
        duracion_min_comunidad  INT NULL COMMENT 'Minutos',
        duracion_max_comunidad  INT NULL COMMENT 'Minutos',
        edad_min_fabricante     INT NULL,
        n_valoraciones_bgg      INT NULL {FUGA},
        n_comentarios           INT NULL {FUGA_POP},
        n_nombres_alternativos  INT NULL,
        n_expansiones           INT NULL,
        n_reimplementaciones    INT NULL,
        es_reimplementacion     BOOLEAN NOT NULL,
        familia                 VARCHAR(500) NULL,
        kickstarter             BOOLEAN NOT NULL,
        imagen_url              VARCHAR(1000) NULL,
        rank_general            INT NULL {FUGA},
        rank_estrategia         INT NULL {FUGA},
        rank_abstractos         INT NULL {FUGA},
        rank_familiares         INT NULL {FUGA},
        rank_tematicos          INT NULL {FUGA},
        rank_cartas_coleccionables INT NULL {FUGA},
        rank_wargames           INT NULL {FUGA},
        rank_party              INT NULL {FUGA},
        rank_infantiles         INT NULL {FUGA},
        PRIMARY KEY (bgg_id)""",
    "etiqueta": """
        id_etiqueta  INT NOT NULL,
        tipo         ENUM('categoria', 'subcategoria', 'mecanica', 'tema') NOT NULL,
        nombre       VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
        PRIMARY KEY (id_etiqueta),
        UNIQUE KEY uq_etiqueta (tipo, nombre)""",
    "juego_etiqueta": """
        bgg_id       INT NOT NULL,
        id_etiqueta  INT NOT NULL,
        PRIMARY KEY (bgg_id, id_etiqueta)""",
    "juego_jugadores": """
        bgg_id       INT NOT NULL,
        n_jugadores  SMALLINT NOT NULL,
        PRIMARY KEY (bgg_id, n_jugadores)""",
    "usuario": """
        id_usuario      INT NOT NULL,
        seudonimo       CHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
        n_valoraciones  INT NOT NULL,
        PRIMARY KEY (id_usuario),
        UNIQUE KEY uq_usuario_seudonimo (seudonimo)""",
    "valoracion": """
        id_usuario  INT NOT NULL,
        bgg_id      INT NOT NULL,
        nota        FLOAT NOT NULL,
        PRIMARY KEY (id_usuario, bgg_id),
        CONSTRAINT ck_valoracion_nota CHECK (nota BETWEEN 1 AND 10)""",
    "ranking_2026": """
        bgg_id          INT NOT NULL,
        nombre          VARCHAR(500) NULL,
        anio            SMALLINT NULL,
        puesto          INT NULL COMMENT 'Rank de BGG; rank es palabra reservada en MySQL',
        nota_media      DOUBLE NULL,
        nota_bayes      DOUBLE NULL,
        n_valoraciones  INT NULL,
        url             VARCHAR(1000) NULL,
        miniatura_url   VARCHAR(1000) NULL,
        PRIMARY KEY (bgg_id)""",
}
COMENTARIOS = {
    "juego": "Catálogo completo de juegos (threnjen)",
    "etiqueta": "Categorías, subcategorías, mecánicas y temas",
    "juego_etiqueta": "Etiquetas de cada juego",
    "juego_jugadores": "Números de jugadores recomendados por la comunidad (GoodPlayers)",
    "usuario": "Usuarios de BGG seudonimizados (HMAC-SHA256) que pasan el filtro",
    "valoracion": f"Valoraciones filtradas: usuarios >= {MIN_VAL_USUARIO}, juegos >= {MIN_VAL_JUEGO}",
    "ranking_2026": "Ranking de BGG del 2026-11-12; sin clave foránea: incluye juegos fuera del catálogo",
}
DESPUES_DE_CARGAR = [
    "ALTER TABLE gold.juego_etiqueta ADD KEY ix_je_etiqueta (id_etiqueta), "
    "ADD CONSTRAINT fk_je_juego FOREIGN KEY (bgg_id) REFERENCES gold.juego (bgg_id), "
    "ADD CONSTRAINT fk_je_etiqueta FOREIGN KEY (id_etiqueta) REFERENCES gold.etiqueta (id_etiqueta)",
    "ALTER TABLE gold.juego_jugadores ADD KEY ix_jj_jugadores (n_jugadores), "
    "ADD CONSTRAINT fk_jj_juego FOREIGN KEY (bgg_id) REFERENCES gold.juego (bgg_id)",
    "ALTER TABLE gold.valoracion ADD KEY ix_valoracion_juego (bgg_id), "
    "ADD CONSTRAINT fk_valoracion_usuario FOREIGN KEY (id_usuario) REFERENCES gold.usuario (id_usuario), "
    "ADD CONSTRAINT fk_valoracion_juego FOREIGN KEY (bgg_id) REFERENCES gold.juego (bgg_id)",
    "ALTER TABLE gold.ranking_2026 ADD KEY ix_ranking_puesto (puesto)",
    "ALTER TABLE gold.juego ADD KEY ix_juego_rank (rank_general)",
]
# Hijas antes que padres
ORDEN_BORRADO = ["valoracion", "juego_etiqueta", "juego_jugadores", "usuario", "etiqueta",
                 "ranking_2026", "juego"]


def columnas(tabla: str) -> list[str]:
    """Nombres de columna de la definición (las líneas que no son claves ni restricciones)."""
    nombres = []
    for linea in TABLAS[tabla].strip().splitlines():
        palabra = linea.split()[0]
        if palabra not in ("PRIMARY", "UNIQUE", "KEY", "CONSTRAINT"):
            nombres.append(palabra)
    return nombres


def _miles(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


# --------------------------------------------------------------------------- construcción
def construir(informe: dict) -> dict[str, pd.DataFrame]:
    juegos = pd.read_parquet(SILVER / "juegos.parquet")
    etiquetas = pd.read_parquet(SILVER / "etiquetas.parquet")
    ranking = pd.read_parquet(SILVER / "ranking_2026.parquet")
    # usuario como diccionario: 19 millones de textos repetidos caben como códigos enteros
    val = pq.read_table(SILVER / "valoraciones.parquet", read_dictionary=["usuario"]).to_pandas()

    gold: dict[str, pd.DataFrame] = {}

    # juego: catálogo completo; la lista de jugadores pasa a su propia tabla
    gold["juego"] = juegos[columnas("juego")].copy()
    jj = juegos[["bgg_id", "buenos_n_jugadores"]].explode("buenos_n_jugadores").dropna()
    gold["juego_jugadores"] = pd.DataFrame({"bgg_id": jj["bgg_id"].astype("int32"),
                                            "n_jugadores": jj["buenos_n_jugadores"].astype("int16")})

    # etiqueta con id propio, en orden estable (tipo, nombre)
    cat = (etiquetas[["tipo", "etiqueta"]].drop_duplicates()
           .sort_values(["tipo", "etiqueta"]).reset_index(drop=True))
    cat.insert(0, "id_etiqueta", np.arange(1, len(cat) + 1, dtype="int32"))
    gold["etiqueta"] = cat.rename(columns={"etiqueta": "nombre"})
    gold["juego_etiqueta"] = (etiquetas.merge(cat, on=["tipo", "etiqueta"])[["bgg_id", "id_etiqueta"]]
                              .sort_values(["bgg_id", "id_etiqueta"]).reset_index(drop=True))

    gold["usuario"], gold["valoracion"] = filtrar_valoraciones(val, informe)
    del val

    gold["ranking_2026"] = ranking.rename(columns={"rank": "puesto"})[columnas("ranking_2026")]

    for nombre, df in gold.items():
        log.info("%-16s %12s filas", nombre, _miles(len(df)))
    return gold


def filtrar_valoraciones(val: pd.DataFrame, informe: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtro iterativo (k-core) con arrays de numpy: usuarios >= MIN_VAL_USUARIO y
    juegos >= MIN_VAL_JUEGO."""
    u = val["usuario"].cat.codes.to_numpy()
    bgg = val["bgg_id"].to_numpy()
    nota = val["nota"].to_numpy()
    seudos = val["usuario"].cat.categories.to_numpy()
    entrada = {"valoraciones": len(val), "usuarios": len(seudos), "juegos": int(np.unique(bgg).size)}

    vivo = np.ones(len(u), dtype=bool)
    iteracion = 0
    while True:
        iteracion += 1
        por_usuario = np.bincount(u[vivo], minlength=len(seudos))
        por_juego = np.bincount(bgg[vivo], minlength=int(bgg.max()) + 1)
        nuevo = vivo & (por_usuario[u] >= MIN_VAL_USUARIO) & (por_juego[bgg] >= MIN_VAL_JUEGO)
        quitadas = int(vivo.sum() - nuevo.sum())
        log.info("Filtro, iteración %d: %s valoraciones quitadas", iteracion, _miles(quitadas))
        vivo = nuevo
        if quitadas == 0:
            break

    u, bgg, nota = u[vivo], bgg[vivo], nota[vivo]
    usados, id_nuevo = np.unique(u, return_inverse=True)   # ids 1..n en orden del seudónimo
    id_usuario = (id_nuevo + 1).astype("int32")
    usuario = pd.DataFrame({"id_usuario": np.arange(1, len(usados) + 1, dtype="int32"),
                            "seudonimo": seudos[usados],
                            "n_valoraciones": np.bincount(id_nuevo).astype("int32")})
    valoracion = pd.DataFrame({"id_usuario": id_usuario, "bgg_id": bgg, "nota": nota})

    salida = {"valoraciones": len(valoracion), "usuarios": len(usuario),
              "juegos": int(np.unique(bgg).size)}
    informe["filtro"] = {"min_val_usuario": MIN_VAL_USUARIO, "min_val_juego": MIN_VAL_JUEGO,
                         "iteraciones": iteracion, "entrada": entrada, "salida": salida}
    for clave in entrada:
        log.info("Filtro: %-12s %12s -> %12s (%.1f %%)", clave, _miles(entrada[clave]),
                 _miles(salida[clave]), 100 * salida[clave] / entrada[clave])
    return usuario, valoracion


def validar(gold: dict[str, pd.DataFrame]) -> None:
    errores = []
    ids_juego = gold["juego"]["bgg_id"]
    claves = {"juego": ["bgg_id"], "etiqueta": ["id_etiqueta"], "usuario": ["id_usuario"],
              "juego_etiqueta": ["bgg_id", "id_etiqueta"], "juego_jugadores": ["bgg_id", "n_jugadores"],
              "valoracion": ["id_usuario", "bgg_id"], "ranking_2026": ["bgg_id"]}
    for tabla, clave in claves.items():
        if gold[tabla].duplicated(clave).any():
            errores.append(f"{tabla}: clave primaria {clave} repetida")
    for tabla in ("juego_etiqueta", "juego_jugadores", "valoracion"):
        if not gold[tabla]["bgg_id"].isin(ids_juego).all():
            errores.append(f"{tabla}: bgg_id que no está en juego")
    if not gold["juego_etiqueta"]["id_etiqueta"].isin(gold["etiqueta"]["id_etiqueta"]).all():
        errores.append("juego_etiqueta: id_etiqueta que no está en etiqueta")
    if not gold["valoracion"]["id_usuario"].isin(gold["usuario"]["id_usuario"]).all():
        errores.append("valoracion: id_usuario que no está en usuario")
    if gold["usuario"]["n_valoraciones"].min() < MIN_VAL_USUARIO:
        errores.append("usuario con menos valoraciones que el mínimo")
    if gold["valoracion"]["bgg_id"].value_counts().min() < MIN_VAL_JUEGO:
        errores.append("juego con menos valoraciones que el mínimo")
    for tabla, df in gold.items():
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) in ("string", "str"):
                largo = df[col].dropna().astype(str).str.len().max()
                limite = {"nombre": 500, "familia": 500, "imagen_url": 1000, "url": 1000,
                          "miniatura_url": 1000}.get(col)
                if limite and largo and largo > limite:
                    errores.append(f"{tabla}.{col}: texto de {largo} caracteres (máx. {limite})")
    for e in errores:
        log.error("Validación: %s", e)
    if errores:
        raise ValueError("Gold no supera la validación")
    log.info("Validación: claves primarias únicas, integridad referencial, mínimos del filtro "
             "y longitudes de texto")


# --------------------------------------------------------------------------- CSV
def escribir_csv(gold: dict[str, pd.DataFrame]) -> dict[str, Path]:
    GOLD.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre, df in gold.items():
        df = df[columnas(nombre)]
        # Booleanos como 0/1
        df = df.astype({c: "int8" for c in df.columns if df[c].dtype == bool})
        ruta = GOLD / f"{nombre}.csv"
        pv.write_csv(pa.Table.from_pandas(df, preserve_index=False), ruta)
        rutas[nombre] = ruta
        log.info("CSV %-22s %8.1f MB", ruta.name, ruta.stat().st_size / 1e6)
    return rutas


# --------------------------------------------------------------------------- MySQL
def cargar_mysql(rutas: dict[str, Path], filas: dict[str, int], informe: dict) -> None:
    con = conectar(local_infile=True, autocommit=True)
    try:
        cur = con.cursor()
        cur.execute("SELECT VERSION()")
        log.info("MySQL: conectado (%s)", cur.fetchone()[0])
        cur.execute("SET SESSION foreign_key_checks = 0")
        cur.execute("SET SESSION unique_checks = 0")

        for tabla in ORDEN_BORRADO:
            cur.execute(f"DROP TABLE IF EXISTS gold.{tabla}")
        for tabla, cuerpo in TABLAS.items():
            cur.execute(f"CREATE TABLE gold.{tabla} ({cuerpo}\n) "
                        f"DEFAULT CHARSET = utf8mb4 COMMENT = {con.escape(COMENTARIOS[tabla])}")
        log.info("MySQL: %d tablas creadas en gold", len(TABLAS))

        for tabla, ruta in rutas.items():
            inicio = time.perf_counter()
            cols = columnas(tabla)
            variables = ", ".join(f"@{c}" for c in cols)
            # Campo vacío = NULL. LENGTH y no NULLIF(@c, ''): MySQL compara ignorando los
            # espacios finales y convertiría "   " en NULL
            asignaciones = ", ".join(f"{c} = IF(LENGTH(@{c}) = 0, NULL, @{c})" for c in cols)
            cur.execute(
                f"LOAD DATA LOCAL INFILE {con.escape(ruta.resolve().as_posix())} "
                f"INTO TABLE gold.{tabla} CHARACTER SET utf8mb4 "
                "FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' ESCAPED BY '' "
                "LINES TERMINATED BY '\\n' IGNORE 1 LINES "
                f"({variables}) SET {asignaciones}")
            # LOAD DATA LOCAL convierte los errores en avisos: un aviso es un dato mal cargado
            cur.execute("SHOW COUNT(*) WARNINGS")
            avisos = cur.fetchone()[0]
            if avisos:
                cur.execute("SHOW WARNINGS LIMIT 5")
                for nivel, codigo, mensaje in cur.fetchall():
                    log.error("MySQL %s: %s %s %s", tabla, nivel, codigo, mensaje)
                raise ValueError(f"La carga de gold.{tabla} dio {avisos} avisos")
            cur.execute(f"SELECT COUNT(*) FROM gold.{tabla}")
            en_mysql = cur.fetchone()[0]
            if en_mysql != filas[tabla]:
                raise ValueError(f"gold.{tabla}: {en_mysql} filas en MySQL y {filas[tabla]} en el CSV")
            log.info("MySQL: gold.%-16s %12s filas cargadas (%.0f s)", tabla, _miles(en_mysql),
                     time.perf_counter() - inicio)

        inicio = time.perf_counter()
        for sentencia in DESPUES_DE_CARGAR:
            cur.execute(sentencia)
        log.info("MySQL: índices y claves foráneas creados (%.0f s)", time.perf_counter() - inicio)
        cur.execute("SET SESSION foreign_key_checks = 1")
        cur.execute("SET SESSION unique_checks = 1")

        comprobar_app(cur, informe)
    finally:
        con.close()


def comprobar_app(cur, informe: dict) -> None:
    """Juegos guardados por usuarios de la app que ya no están en el catálogo."""
    cur.execute("SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'app' AND table_name = 'cuenta_juego'")
    if not cur.fetchone()[0]:
        log.warning("No existe app.cuenta_juego: ejecuta docker/mysql/init/02_app.sql")
        return
    cur.execute("SELECT COUNT(*) FROM app.cuenta_juego cj "
                "LEFT JOIN gold.juego j ON j.bgg_id = cj.bgg_id WHERE j.bgg_id IS NULL")
    huerfanos = cur.fetchone()[0]
    informe["app_juegos_fuera_de_catalogo"] = huerfanos
    if huerfanos:
        log.warning("app.cuenta_juego: %d filas con juegos que ya no están en gold.juego", huerfanos)
    else:
        log.info("app.cuenta_juego: todos los juegos guardados siguen en el catálogo")


# --------------------------------------------------------------------------- principal
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sin-mysql", action="store_true", help="solo genera los CSV")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    configurar_logging("gold.log", carpeta=LOGS)
    inicio = time.perf_counter()

    informe_silver = SILVER / "_informe_calidad.json"
    if not informe_silver.exists():
        logging.getLogger("gold").error("No hay Silver: ejecuta python -m pipeline.silver")
        return 1
    origen = json.loads(informe_silver.read_text(encoding="utf-8"))["origen"]
    log.info("Gold: inicio (Silver generado desde %s)", origen)
    informe: dict = {"origen": origen}

    try:
        gold = construir(informe)
        validar(gold)
        filas = {nombre: len(df) for nombre, df in gold.items()}
        rutas = escribir_csv(gold)
        del gold
        if args.sin_mysql:
            log.info("MySQL: no se carga (--sin-mysql)")
        else:
            cargar_mysql(rutas, filas, informe)
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado en Gold")
        return 1

    informe.update({"creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "filas": filas, "cargado_en_mysql": not args.sin_mysql,
                    "columnas_con_fuga": COLUMNAS_CON_FUGA})
    (GOLD / "_informe_gold.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    log.info("Gold: fin (%.0f s)", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())