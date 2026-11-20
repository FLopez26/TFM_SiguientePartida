"""Conexión con el servidor de MLflow, donde se registra cada entrenamiento y cada medida.

MLflow corre en Docker (servicio `mlflow` de docker-compose.yml). Guarda los parámetros, las
métricas y las etiquetas en el esquema `mlflow` de MySQL, y los ficheros (modelos, tablas) en
el volumen `mlflow_artefactos`. Los ficheros se suben a través del propio servidor, así que
desde aquí solo hace falta su dirección: MLFLOW_TRACKING_URI en .env.

    python -m ml.seguimiento
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# Antes de importar mlflow: sin estadísticas de uso ni mensajes en cada ejecución
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

import mlflow  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from pipeline.rutas import LOGS, ROOT  # noqa: E402
from siguiente_partida import configurar_logging  # noqa: E402

log = logging.getLogger("ml.seguimiento")

URI_POR_DEFECTO = "http://localhost:5000"
EXPERIMENTO_COMPROBACION = "comprobacion"


def conectar() -> str:
    """Comprueba que el servidor responde y deja MLflow apuntando a él. Devuelve la dirección."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", URI_POR_DEFECTO).rstrip("/")
    try:
        requests.get(f"{uri}/health", timeout=5).raise_for_status()
        version_servidor = requests.get(f"{uri}/version", timeout=5).text.strip()
    except requests.RequestException as e:
        raise ValueError(f"MLflow no responde en {uri} ({e.__class__.__name__}). "
                         "Arráncalo con: docker compose up -d mlflow") from None
    mlflow.set_tracking_uri(uri)
    logging.getLogger("mlflow").setLevel(logging.WARNING)   # sin "Creating experiment..."
    log.info("MLflow: conectado a %s (servidor %s, cliente %s)", uri, version_servidor,
             mlflow.__version__)
    if version_servidor != mlflow.__version__:
        log.warning("MLflow: el servidor (%s) y el cliente (%s) tienen versiones distintas; "
                    "usa la misma en docker/mlflow/Dockerfile y en requirements.txt",
                    version_servidor, mlflow.__version__)
    return uri


def main() -> int:
    load_dotenv(ROOT / ".env")
    configurar_logging("ml.log", carpeta=LOGS)
    log.info("Comprobación de MLflow: inicio")
    try:
        conectar()
        mlflow.set_experiment(EXPERIMENTO_COMPROBACION)
        texto = "prueba de subida de ficheros"
        with mlflow.start_run(run_name="prueba") as run, tempfile.TemporaryDirectory() as tmp:
            mlflow.log_param("parametro", 1)
            mlflow.log_metric("metrica", 0.5)
            ruta = Path(tmp) / "prueba.txt"
            ruta.write_text(texto, encoding="utf-8")
            mlflow.log_artifact(str(ruta))
        log.info("MLflow: ejecución de prueba creada con un parámetro, una métrica y un fichero")

        # Leerlo de vuelta: la métrica sale de MySQL y el fichero, del volumen de artefactos
        guardada = mlflow.get_run(run.info.run_id)
        if guardada.data.metrics.get("metrica") != 0.5:
            raise ValueError("MLflow no ha guardado la métrica en MySQL")
        with tempfile.TemporaryDirectory() as tmp:
            bajado = mlflow.artifacts.download_artifacts(run_id=run.info.run_id,
                                                         artifact_path="prueba.txt", dst_path=tmp)
            if Path(bajado).read_text(encoding="utf-8") != texto:
                raise ValueError("El fichero bajado de MLflow no coincide con el subido")
        log.info("MLflow: métrica leída de MySQL y fichero leído del volumen de artefactos")

        MlflowClient().delete_run(run.info.run_id)
        log.info("MLflow: ejecución de prueba borrada")
    except (ValueError, MlflowException) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado comprobando MLflow")
        return 1
    log.info("Comprobación de MLflow: todo correcto")
    return 0


if __name__ == "__main__":
    sys.exit(main())