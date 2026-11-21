"""Ajustes de la API, leídos de las variables de entorno.

En local salen de .env (en la raíz). En Docker no hay .env dentro de la imagen: las variables
las pone docker-compose.yml, y load_dotenv no sobrescribe las que ya existen.

    MYSQL_HOST            127.0.0.1 en local; "mysql" dentro de Docker
    MYSQL_PORT            3306
    MYSQL_USER            tfm
    MYSQL_PASSWORD        obligatoria
    MLFLOW_TRACKING_URI   http://localhost:5000 en local; http://mlflow:5000 dentro de Docker
    API_HOST              127.0.0.1: solo se puede llamar desde tu PC (0.0.0.0 dentro de Docker)
    API_PORT              8000
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

# Modelo de producción: se pide siempre por el alias, nunca por una ruta ni por un número de
# versión fijo. Para cambiar de versión se mueve el alias en MLflow y se reinicia la API.
# Son los mismos nombres que en ml/produccion.py (la API no importa ml/).
NOMBRE_REGISTRO = "recomendador_juegos"
ALIAS = "produccion"


@dataclass(frozen=True)
class Ajustes:
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mlflow_uri: str
    host: str
    puerto: int


def leer_ajustes() -> Ajustes:
    load_dotenv(ROOT / ".env")
    clave = os.environ.get("MYSQL_PASSWORD")
    if not clave:
        raise ValueError("Falta MYSQL_PASSWORD en .env")
    return Ajustes(
        mysql_host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        mysql_port=int(os.environ.get("MYSQL_PORT", "3306")),
        mysql_user=os.environ.get("MYSQL_USER", "tfm"),
        mysql_password=clave,
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip("/"),
        host=os.environ.get("API_HOST", "127.0.0.1"),
        puerto=int(os.environ.get("API_PORT", "8000")),
    )