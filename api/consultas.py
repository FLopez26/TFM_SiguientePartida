"""Consultas de solo lectura al esquema gold (catálogo).

Están separadas de las rutas para poder reutilizarlas: las recomendaciones y el chat también
necesitarán los datos de los juegos.

Lo que llega del usuario va siempre como parámetro (:q, :bgg_id...), nunca pegado al texto del
SQL; las f-strings de abajo solo insertan constantes de este fichero.
"""
from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from api.esquemas import (Etiqueta, EtiquetasJuego, FichaJuego, Filtros, JuegoResumen,
                          TipoEtiqueta)

URL_BGG = "https://boardgamegeek.com/boardgame/{}"
COLUMNAS_FICHA = [
    "bgg_id", "nombre", "anio", "imagen_url", "peso", "min_jugadores", "max_jugadores",
    "mejor_n_jugadores", "duracion_fabricante", "duracion_min_comunidad",
    "duracion_max_comunidad", "edad_min_fabricante", "edad_rec_comunidad", "nota_media_bgg",
    "n_valoraciones_bgg", "rank_general", "familia", "n_expansiones",
]


def _escapar_like(texto: str) -> str:
    """Para que un % o un _ escrito por el usuario se busque tal cual, no como comodín."""
    return texto.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def buscar_juegos(c: Connection, q: str, limite: int) -> list[JuegoResumen]:
    """Juegos cuyo nombre contiene q, sin distinguir mayúsculas ni tildes (intercalación ai_ci).

    Primero el nombre exacto, después los que empiezan por q y después el resto; dentro de cada
    grupo, los más valorados en BGG (así "catan" pone CATAN antes que sus expansiones raras).
    """
    q = q.strip()
    filas = c.execute(text("""
        SELECT bgg_id, nombre, anio, imagen_url
        FROM gold.juego
        WHERE nombre LIKE :contiene ESCAPE '!'
        ORDER BY nombre = :q DESC, nombre LIKE :empieza ESCAPE '!' DESC,
                 n_valoraciones_bgg DESC, bgg_id
        LIMIT :limite"""),
        {"q": q, "contiene": f"%{_escapar_like(q)}%", "empieza": f"{_escapar_like(q)}%",
         "limite": limite})
    return [JuegoResumen(**f._mapping) for f in filas]


def ficha_juego(c: Connection, bgg_id: int) -> FichaJuego | None:
    """Ficha completa de un juego, con sus etiquetas y los números de jugadores recomendados.
    None si el juego no está en el catálogo."""
    fila = c.execute(text(f"SELECT {', '.join(COLUMNAS_FICHA)} FROM gold.juego "
                          "WHERE bgg_id = :bgg_id"), {"bgg_id": bgg_id}).first()
    if fila is None:
        return None
    jugadores = c.execute(text("SELECT n_jugadores FROM gold.juego_jugadores "
                               "WHERE bgg_id = :bgg_id ORDER BY n_jugadores"),
                          {"bgg_id": bgg_id}).scalars().all()
    etiquetas: dict[str, list[str]] = {}
    for tipo, nombre in c.execute(text("""
            SELECT e.tipo, e.nombre
            FROM gold.juego_etiqueta je JOIN gold.etiqueta e USING (id_etiqueta)
            WHERE je.bgg_id = :bgg_id
            ORDER BY e.nombre"""), {"bgg_id": bgg_id}):
        etiquetas.setdefault(tipo, []).append(nombre)
    return FichaJuego(**fila._mapping, url_bgg=URL_BGG.format(bgg_id),
                      jugadores_recomendados=list(jugadores),
                      etiquetas=EtiquetasJuego(**etiquetas))


def listar_etiquetas(c: Connection, tipo: TipoEtiqueta | None = None) -> list[Etiqueta]:
    """Etiquetas con cuántos juegos tiene cada una; dentro de cada tipo, las más usadas primero."""
    filtro = "WHERE e.tipo = :tipo" if tipo else ""
    filas = c.execute(text(f"""
        SELECT e.id_etiqueta, e.tipo, e.nombre, COUNT(je.bgg_id) AS n_juegos
        FROM gold.etiqueta e LEFT JOIN gold.juego_etiqueta je USING (id_etiqueta)
        {filtro}
        GROUP BY e.id_etiqueta, e.tipo, e.nombre
        ORDER BY e.tipo, n_juegos DESC, e.nombre"""), {"tipo": tipo})
    return [Etiqueta(**f._mapping) for f in filas]



def resumenes(c: Connection, ids: Iterable[int]) -> dict[int, JuegoResumen]:
    """Nombre, año e imagen de varios juegos de una vez: {bgg_id: resumen}."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    consulta = text("SELECT bgg_id, nombre, anio, imagen_url FROM gold.juego "
                    "WHERE bgg_id IN :ids").bindparams(bindparam("ids", expanding=True))
    return {f.bgg_id: JuegoResumen(**f._mapping) for f in c.execute(consulta, {"ids": ids})}


class EtiquetasDesconocidas(ValueError):
    """Alguna id_etiqueta de los filtros no está en gold.etiqueta."""


def candidatos(c: Connection, filtros: Filtros) -> set[int]:
    """bgg_id de los juegos del catálogo que cumplen los filtros (ver Filtros en esquemas.py).

    Jugadores: si hay votos de la comunidad (juego_jugadores), el número tiene que ser uno de
    los recomendados; si no los hay, basta con que esté entre min_jugadores y max_jugadores.
    Etiquetas: una condición por tipo; dentro de cada tipo basta con tener una de ellas.
    Un dato vacío no cumple la condición (MySQL: NULL >= 2 no es verdadero).
    """
    condiciones: list[str] = []
    params: dict = {}
    if filtros.jugadores is not None:
        condiciones.append("""(
            EXISTS (SELECT 1 FROM gold.juego_jugadores jj
                    WHERE jj.bgg_id = j.bgg_id AND jj.n_jugadores = :jugadores)
            OR (NOT EXISTS (SELECT 1 FROM gold.juego_jugadores jj WHERE jj.bgg_id = j.bgg_id)
                AND j.min_jugadores <= :jugadores AND j.max_jugadores >= :jugadores))""")
        params["jugadores"] = filtros.jugadores
    for campo, columna, operador in [("duracion_min", "duracion_fabricante", ">="),
                                     ("duracion_max", "duracion_fabricante", "<="),
                                     ("peso_min", "peso", ">="), ("peso_max", "peso", "<=")]:
        if getattr(filtros, campo) is not None:
            condiciones.append(f"j.{columna} {operador} :{campo}")
            params[campo] = getattr(filtros, campo)

    listas = []   # parámetros que son listas (IN): SQLAlchemy los expande en :p_1, :p_2...
    if filtros.etiquetas:
        consulta = text("SELECT id_etiqueta, tipo FROM gold.etiqueta WHERE id_etiqueta IN :ids"
                        ).bindparams(bindparam("ids", expanding=True))
        tipos = dict(c.execute(consulta, {"ids": sorted(set(filtros.etiquetas))}).all())
        faltan = sorted(set(filtros.etiquetas) - tipos.keys())
        if faltan:
            raise EtiquetasDesconocidas(f"No existen las etiquetas {faltan}: mira /etiquetas")
        por_tipo: dict[str, list[int]] = {}
        for id_etiqueta, tipo in tipos.items():
            por_tipo.setdefault(tipo, []).append(id_etiqueta)
        for i, ids in enumerate(por_tipo.values()):
            condiciones.append(f"EXISTS (SELECT 1 FROM gold.juego_etiqueta je WHERE "
                               f"je.bgg_id = j.bgg_id AND je.id_etiqueta IN :etiquetas_{i})")
            params[f"etiquetas_{i}"] = ids
            listas.append(f"etiquetas_{i}")

    sql = "SELECT j.bgg_id FROM gold.juego j"
    if condiciones:
        sql += " WHERE " + " AND ".join(condiciones)
    consulta = text(sql).bindparams(*(bindparam(n, expanding=True) for n in listas))
    return set(c.execute(consulta, params).scalars())