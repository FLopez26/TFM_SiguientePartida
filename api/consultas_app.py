"""Consultas al esquema app: cuentas y juegos guardados (docker/mysql/init/02_app.sql).

El SQL es el común a MySQL y SQLite, para que los tests sigan sin necesitar MySQL. Por eso no
se usa INSERT ... ON DUPLICATE KEY UPDATE: para guardar un juego se
intenta primero el UPDATE y, si no había fila, el INSERT.

Las funciones que escriben no confirman la transacción: las rutas las llaman con escribir(),
que confirma al final y la repite si MySQL la cancela por un interbloqueo. Si algo falla a
mitad, la conexión vuelve al pool sin confirmar y no queda nada a medias.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, OperationalError

from api import consultas
from api.auth import Usuario
from api.esquemas import Cuenta, JuegoGuardado, JuegosPorLista, Lista

log = logging.getLogger("api.cuentas")

T = TypeVar("T")
INTERBLOQUEO = 1213
INTENTOS = 3


def escribir(c: Connection, operacion: Callable[[], T]) -> T:
    """Ejecuta la operación y confirma. Con dos peticiones a la vez sobre el mismo juego (un
    doble clic), MySQL puede cancelar una de ellas por interbloqueo: ya ha deshecho su
    transacción y pide repetirla. Se repite hasta INTENTOS veces; la otra petición ya habrá
    terminado, así que la repetición encuentra su fila."""
    for intento in range(1, INTENTOS + 1):
        try:
            resultado = operacion()
            c.commit()
            return resultado
        except OperationalError as e:
            codigo = getattr(e.orig, "args", (None,))[0]
            if codigo != INTERBLOQUEO or intento == INTENTOS:
                raise
            c.rollback()
            log.warning("Cuentas: interbloqueo en MySQL; se repite la escritura (intento %d "
                        "de %d)", intento + 1, INTENTOS)
    raise AssertionError("no se llega aquí")


def buscar_cuenta(c: Connection, uid: str) -> int | None:
    return c.execute(text("SELECT id_cuenta FROM app.cuenta WHERE firebase_uid = :uid"),
                     {"uid": uid}).scalar_one_or_none()


def obtener_o_crear_cuenta(c: Connection, usuario: Usuario) -> int:
    """id_cuenta del usuario; si es la primera vez que llega, crea la cuenta y la confirma ya,
    para que no se pierda aunque luego falle la petición."""
    id_cuenta = buscar_cuenta(c, usuario.uid)
    if id_cuenta is not None:
        return id_cuenta
    try:
        c.execute(text("""
            INSERT INTO app.cuenta (firebase_uid, nombre_visible, es_invitado, ultimo_acceso)
            VALUES (:uid, :nombre, :invitado, CURRENT_TIMESTAMP)"""),
            {"uid": usuario.uid, "nombre": usuario.nombre_visible,
             "invitado": usuario.es_invitado})
        c.commit()
    except IntegrityError:
        # Otra petición del mismo usuario la ha creado a la vez (dos pestañas). Se deshace esta
        # transacción para leer la fila que ha confirmado la otra
        c.rollback()
        return buscar_cuenta(c, usuario.uid)
    id_cuenta = buscar_cuenta(c, usuario.uid)
    log.info("Cuentas: cuenta %d creada (%s)", id_cuenta,
             "invitado" if usuario.es_invitado else "Google")
    return id_cuenta


def registrar_sesion(c: Connection, id_cuenta: int, usuario: Usuario) -> None:
    """Al iniciar sesión: último acceso, si es invitado (deja de serlo al enlazar con Google) y
    el nombre de Google. Si el token no trae nombre, se conserva el que hubiera."""
    c.execute(text("""
        UPDATE app.cuenta
        SET nombre_visible = COALESCE(:nombre, nombre_visible), es_invitado = :invitado,
            ultimo_acceso = CURRENT_TIMESTAMP
        WHERE id_cuenta = :id"""),
        {"id": id_cuenta, "nombre": usuario.nombre_visible, "invitado": usuario.es_invitado})


def leer_cuenta(c: Connection, id_cuenta: int) -> Cuenta:
    fila = c.execute(text("SELECT nombre_visible, es_invitado, creada_en FROM app.cuenta "
                          "WHERE id_cuenta = :id"), {"id": id_cuenta}).one()
    por_lista = dict(c.execute(text("SELECT lista, COUNT(*) FROM app.cuenta_juego "
                                    "WHERE id_cuenta = :id GROUP BY lista"), {"id": id_cuenta}).all())
    return Cuenta(nombre_visible=fila.nombre_visible, es_invitado=bool(fila.es_invitado),
                  creada_en=fila.creada_en, juegos=JuegosPorLista(**por_lista))


def borrar_cuenta(c: Connection, id_cuenta: int) -> int:
    """Borra la cuenta y sus juegos. Devuelve cuántos juegos tenía. Se borran los juegos de
    forma explícita aunque la clave foránea ya lo haga (ON DELETE CASCADE): SQLite no aplica
    las claves foráneas si no se le pide."""
    n = c.execute(text("DELETE FROM app.cuenta_juego WHERE id_cuenta = :id"),
                  {"id": id_cuenta}).rowcount
    c.execute(text("DELETE FROM app.cuenta WHERE id_cuenta = :id"), {"id": id_cuenta})
    return n


def fusionar(c: Connection, origen: int, destino: int) -> tuple[int, int]:
    """Pasa los juegos de la cuenta origen (invitado) a la destino (Google) y borra la origen.
    Si un juego está en las dos, se queda el de destino. Devuelve (movidos, ya_estaban).

    Es un INSERT ... SELECT y no un UPDATE de id_cuenta porque MySQL no deja que un UPDATE lea
    en una subconsulta la misma tabla que modifica."""
    total = c.execute(text("SELECT COUNT(*) FROM app.cuenta_juego WHERE id_cuenta = :o"),
                      {"o": origen}).scalar_one()
    movidos = c.execute(text("""
        INSERT INTO app.cuenta_juego (id_cuenta, bgg_id, lista, nota, actualizado_en)
        SELECT :d, o.bgg_id, o.lista, o.nota, o.actualizado_en
        FROM app.cuenta_juego o
        WHERE o.id_cuenta = :o
          AND NOT EXISTS (SELECT 1 FROM app.cuenta_juego d
                          WHERE d.id_cuenta = :d AND d.bgg_id = o.bgg_id)"""),
        {"o": origen, "d": destino}).rowcount
    borrar_cuenta(c, origen)
    return movidos, total - movidos


def leer_juegos(c: Connection, id_cuenta: int, lista: Lista | None = None,
                bgg_id: int | None = None) -> list[JuegoGuardado]:
    """Juegos guardados, los últimos cambiados primero, con su resumen del catálogo. Se puede
    pedir solo una lista o solo un juego."""
    filtros = ("AND lista = :lista " if lista else "") + ("AND bgg_id = :bgg_id" if bgg_id else "")
    filas = c.execute(text(f"""
        SELECT bgg_id, lista, nota, actualizado_en FROM app.cuenta_juego
        WHERE id_cuenta = :id {filtros}
        ORDER BY actualizado_en DESC, bgg_id"""),
        {"id": id_cuenta, "lista": lista, "bgg_id": bgg_id}).all()
    resumenes = consultas.resumenes(c, (f.bgg_id for f in filas))
    return [JuegoGuardado(bgg_id=f.bgg_id, lista=f.lista,
                          nota=None if f.nota is None else float(f.nota),
                          actualizado_en=f.actualizado_en, juego=resumenes.get(f.bgg_id))
            for f in filas]


def guardar_juego(c: Connection, id_cuenta: int, bgg_id: int, lista: Lista,
                  nota: float | None) -> None:
    """Guarda el juego en la lista (y lo saca de la que estuviera). Sustituye la nota."""
    params = {"id": id_cuenta, "bgg_id": bgg_id, "lista": lista, "nota": nota}
    actualizar = text("""
        UPDATE app.cuenta_juego
        SET lista = :lista, nota = :nota, actualizado_en = CURRENT_TIMESTAMP
        WHERE id_cuenta = :id AND bgg_id = :bgg_id""")
    # rowcount cuenta las filas encontradas, no solo las cambiadas: SQLAlchemy conecta a MySQL con CLIENT_FOUND_ROWS
    if c.execute(actualizar, params).rowcount:
        return
    try:
        c.execute(text("""
            INSERT INTO app.cuenta_juego (id_cuenta, bgg_id, lista, nota)
            VALUES (:id, :bgg_id, :lista, :nota)"""), params)
    except IntegrityError:
        # Dos peticiones a la vez para el mismo juego (doble clic): la otra ya lo ha insertado.
        # El UPDATE lee siempre la última versión confirmada, así que ahora sí encuentra la fila.
        # Si tampoco la encuentra, el error era otro y se deja subir
        if not c.execute(actualizar, params).rowcount:
            raise


def quitar_juego(c: Connection, id_cuenta: int, bgg_id: int) -> bool:
    return bool(c.execute(text("DELETE FROM app.cuenta_juego WHERE id_cuenta = :id "
                               "AND bgg_id = :bgg_id"), {"id": id_cuenta, "bgg_id": bgg_id}).rowcount)


def existe_en_catalogo(c: Connection, bgg_id: int) -> bool:
    return c.execute(text("SELECT 1 FROM gold.juego WHERE bgg_id = :bgg_id"),
                     {"bgg_id": bgg_id}).first() is not None