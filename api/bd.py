"""Conexión con MySQL: un grupo de conexiones (pool) que comparte toda la API.

Abrir una conexión nueva en cada petición es lento; el pool deja unas cuantas abiertas y las
reutiliza. Las consultas nombran siempre el esquema (gold.juego, app.cuenta), así que la
conexión no elige ninguna base de datos por defecto.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Connection, Engine

from api.config import Ajustes


def crear_motor(ajustes: Ajustes) -> Engine:
    # URL.create escapa la contraseña: puede llevar @ : / # sin romper la dirección
    url = URL.create("mysql+pymysql", username=ajustes.mysql_user,
                     password=ajustes.mysql_password, host=ajustes.mysql_host,
                     port=ajustes.mysql_port, query={"charset": "utf8mb4"})
    return create_engine(
        url,
        pool_size=5, max_overflow=5,
        pool_pre_ping=True,     # comprueba la conexión antes de usarla (MySQL cierra las inactivas)
        pool_recycle=3600,      # y las renueva cada hora
        connect_args={"connect_timeout": 5},
    )


def contar_juegos(motor: Engine) -> int:
    """Comprueba la conexión y que Gold está cargado: número de juegos de gold.juego."""
    with motor.connect() as c:
        return c.execute(text("SELECT COUNT(*) FROM gold.juego")).scalar_one()


def conexion(request: Request) -> Iterator[Connection]:
    """Dependencia de FastAPI: una conexión del pool para la petición, devuelta al terminar."""
    with request.app.state.motor.connect() as c:
        yield c



# Para las rutas: def ruta(c: Conexion) recibe una conexión ya abierta
Conexion = Annotated[Connection, Depends(conexion)]