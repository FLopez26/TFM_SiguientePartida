"""Conexión a MySQL con los datos de .env."""
from __future__ import annotations

import os

import pymysql


def conectar(**extra) -> pymysql.connections.Connection:
    """Conexión con el usuario del proyecto.

    Variables de entorno: MYSQL_HOST (por defecto 127.0.0.1; "mysql" dentro de Docker),
    MYSQL_PORT (3306), MYSQL_USER (tfm) y MYSQL_PASSWORD (obligatoria).
    """
    clave = os.environ.get("MYSQL_PASSWORD")
    if not clave:
        raise ValueError("Falta MYSQL_PASSWORD en .env")
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "tfm"),
        password=clave,
        charset="utf8mb4",
        **extra,
    )