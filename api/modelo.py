"""Recomendador de producción, cargado de MLflow por su alias.

Al arrancar, la API pregunta a MLflow qué versión tiene el alias "produccion" y carga esa
versión. Así la versión que muestra /salud es seguro la cargada, aunque se mueva el alias
mientras la API está en marcha. Para usar otra versión: mover el alias y reiniciar la API.

El modelo es el de ml/modelo_mlflow.py: MLflow guarda ese código junto a los vecinos (.npz) y
lo ejecuta al cargarlo. Por eso aquí hace falta la librería siguiente-partida instalada.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

# Antes de importar mlflow: sin estadísticas de uso ni mensajes en cada arranque
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")   # sin barra al descargar

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.pyfunc import PyFuncModel  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from api.config import ALIAS, NOMBRE_REGISTRO, Ajustes  # noqa: E402

log = logging.getLogger("api.modelo")


@dataclass(frozen=True)
class Recomendador:
    modelo: PyFuncModel
    version: str              # versión del registro de MLflow
    variante: str             # p. ej. item_knn-binaria-k10
    juegos: frozenset[int]    # juegos que conoce el modelo (los que tienen notas en Gold)

    def recomendar(self, coleccion: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """coleccion: bgg_id, nota (opcional) y excluir (opcional); ver ml/modelo_mlflow.py.
        Devuelve posicion, bgg_id, puntuacion y porque_bgg_id."""
        return self.modelo.predict(coleccion, params={"k": k})


def cargar(ajustes: Ajustes) -> Recomendador:
    inicio = time.perf_counter()
    uri = ajustes.mlflow_uri
    # Comprobación rápida: si MLflow no está arrancado, su cliente reintentaría durante minutos
    try:
        requests.get(f"{uri}/health", timeout=5).raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"MLflow no responde en {uri} ({e.__class__.__name__}). "
                         "Arráncalo con: docker compose up -d mlflow") from None
    mlflow.set_tracking_uri(uri)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    try:
        version = MlflowClient().get_model_version_by_alias(NOMBRE_REGISTRO, ALIAS)
    except MlflowException:
        raise ValueError(f"MLflow no tiene el modelo {NOMBRE_REGISTRO} con el alias '{ALIAS}': "
                         "ejecuta python -m ml.produccion") from None
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE_REGISTRO}/{version.version}")
    rec = Recomendador(
        modelo=modelo,
        version=version.version,
        variante=version.tags.get("variante", "desconocida"),
        juegos=frozenset(modelo.unwrap_python_model().ids.tolist()),
    )

    # Comprobación: con una colección vacía tiene que devolver los más populares
    prueba = rec.recomendar(pd.DataFrame({"bgg_id": pd.Series([], dtype="int64")}), k=3)
    if len(prueba) != 3:
        raise ValueError(f"El modelo cargado devuelve {len(prueba)} juegos en vez de 3")
    log.info("Modelo: %s versión %s (alias '%s', %s) con %s juegos, cargado en %.1f s",
             NOMBRE_REGISTRO, rec.version, ALIAS, rec.variante,
             f"{len(rec.juegos):,}".replace(",", "."), time.perf_counter() - inicio)
    return rec