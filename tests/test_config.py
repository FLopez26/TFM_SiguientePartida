"""Pruebas de la lectura de ajustes (api/config.py), sin leer el .env.

    python -m pytest tests/test_config.py -q
"""
from __future__ import annotations

import pytest

from api import config

VARIABLES = ["MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MLFLOW_TRACKING_URI",
             "API_HOST", "API_PORT", "FIREBASE_PROJECT_ID", "GEMINI_API_KEY", "GEMINI_MODELO"]


@pytest.fixture
def entorno(monkeypatch):
    """Entorno vacío: sin las variables del PC y sin cargar el .env."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    for v in VARIABLES:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_valores_por_defecto(entorno):
    entorno.setenv("MYSQL_PASSWORD", "x")
    a = config.leer_ajustes()
    assert (a.mysql_host, a.mysql_port, a.mysql_user, a.mlflow_uri, a.host, a.puerto,
            a.firebase_proyecto, a.gemini_clave, a.gemini_modelo) == (
        "127.0.0.1", 3306, "tfm", "http://localhost:5000", "127.0.0.1", 8000, None, None,
        "gemini-3.5-flash-lite")


def test_lee_todas_las_variables(entorno):
    valores = {"MYSQL_HOST": "mysql", "MYSQL_PORT": "3307", "MYSQL_USER": "u",
               "MYSQL_PASSWORD": "clave", "MLFLOW_TRACKING_URI": "http://mlflow:5000/",
               "API_HOST": "0.0.0.0", "API_PORT": "8001",
               "FIREBASE_PROJECT_ID": "  siguiente-partida-tfm ",
               "GEMINI_API_KEY": " AIza-de-prueba ", "GEMINI_MODELO": "gemini-otro"}
    for v, valor in valores.items():
        entorno.setenv(v, valor)
    a = config.leer_ajustes()
    assert a == config.Ajustes("mysql", 3307, "u", "clave", "http://mlflow:5000", "0.0.0.0",
                               8001, "siguiente-partida-tfm", "AIza-de-prueba", "gemini-otro")


def test_gemini_vacio_es_como_si_no_estuviera(entorno):
    entorno.setenv("MYSQL_PASSWORD", "x")
    entorno.setenv("GEMINI_API_KEY", "  ")
    entorno.setenv("GEMINI_MODELO", "")
    a = config.leer_ajustes()
    assert (a.gemini_clave, a.gemini_modelo) == (None, config.MODELO_GEMINI)


def test_la_clave_de_gemini_no_sale_en_el_repr(entorno):
    entorno.setenv("MYSQL_PASSWORD", "x")
    entorno.setenv("GEMINI_API_KEY", "AIza-secreta")
    a = config.leer_ajustes()
    assert a.gemini_clave == "AIza-secreta" and "AIza-secreta" not in repr(a)


def test_sin_contrasena_de_mysql(entorno):
    with pytest.raises(ValueError, match="MYSQL_PASSWORD"):
        config.leer_ajustes()