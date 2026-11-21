"""Modelo de producción: la ganadora de la validación, entrenada con el 100 % y registrada en MLflow.

1. Lee de MLflow la ganadora de la última evaluación completa (con todos los usuarios) y sus
   cifras del test.
2. La entrena con todas las valoraciones: ajuste + validación + test.
3. La guarda como modelo de MLflow (ml/modelo_mlflow.py + los vecinos en un .npz) y la registra
   con el nombre "recomendador_juegos": cada ejecución crea una versión nueva.
4. Le pone el alias "produccion". La API cargará siempre models:/recomendador_juegos@produccion.
5. La vuelve a cargar por el alias y recomienda a una colección de ejemplo, como comprobación.

Las cifras que se reportan son las del test (modelo entrenado con el 80 %): con el 100 % ya no
queda nada sin ver con lo que medir. Se copian al modelo registrado para tenerlas a mano.

Uso (desde la raíz, con MLflow arrancado y después de python -m ml.evaluar):
    python -m ml.produccion
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

# Primero seguimiento: configura MLflow (sin telemetría ni avisos) antes de importarlo
from ml.seguimiento import conectar  # noqa: I001

import mlflow
import numpy as np
import pandas as pd
import scipy
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec, ParamSchema, ParamSpec, Schema

import siguiente_partida
from ml.datos import AJUSTE, TEST, VALIDACION, leer_catalogo, origen_gold, preparar_valoraciones
from ml.evaluar import EXPERIMENTO_TEST, EXPERIMENTO_VALIDACION, Rasgos, construir, nombre
from pipeline.rutas import LOGS, ROOT
from siguiente_partida import configurar_logging

log = logging.getLogger("ml.produccion")

EXPERIMENTO_PRODUCCION = "produccion"
NOMBRE_REGISTRO = "recomendador_juegos"
ALIAS = "produccion"
CODIGO_MODELO = Path(__file__).with_name("modelo_mlflow.py")
# Colección de ejemplo: CATAN, Carcassonne, Ticket to Ride y Pandemic
EJEMPLO_BGG_IDS = [13, 822, 9209, 30549]
METRICAS_TEST = ("precision_10", "recall_10", "ndcg_10", "ndcg_10_ic_inf", "ndcg_10_ic_sup",
                 "dif_ndcg_10", "dif_ndcg_10_ic_inf", "dif_ndcg_10_ic_sup", "cobertura")

FIRMA = ModelSignature(
    inputs=Schema([ColSpec("long", "bgg_id"), ColSpec("double", "nota", required=False),
                   ColSpec("boolean", "excluir", required=False)]),
    outputs=Schema([ColSpec("long", "posicion"), ColSpec("long", "bgg_id"),
                    ColSpec("double", "puntuacion"), ColSpec("long", "porque_bgg_id")]),
    params=ParamSchema([ParamSpec("k", "long", 10)]),
)


def ganadora_ultima_evaluacion() -> tuple[dict, str, dict]:
    """(variante ganadora, lote, métricas del test) de la última evaluación con todos los usuarios."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENTO_VALIDACION],
        filter_string="tags.ganadora = 'si' and params.muestra_usuarios = 'todos'",
        order_by=["attributes.start_time DESC"], max_results=1, output_format="list")
    if not runs:
        raise ValueError("No hay ninguna evaluación completa en MLflow: ejecuta python -m ml.evaluar")
    run = runs[0]
    p, lote = run.data.params, run.data.tags["lote"]
    spec = {c: p[c] for c in ("familia", "criterio", "rasgos", "variante") if c in p}
    if "vecinos" in p:
        spec["vecinos"] = int(p["vecinos"])
    if run.data.tags.get("datos") == "ejemplo":
        log.warning("La evaluación se hizo con la muestra sintética: el modelo no es el real")

    test = mlflow.search_runs(
        experiment_names=[EXPERIMENTO_TEST],
        filter_string=f"tags.lote = '{lote}' and tags.ganadora = 'si'",
        max_results=1, output_format="list")
    metricas = {k: v for k, v in test[0].data.metrics.items() if k in METRICAS_TEST} if test else {}
    if not metricas:
        log.warning("El lote %s no tiene test de la ganadora: ejecuta python -m ml.evaluar "
                    "--fase test", lote)
    log.info("Ganadora del lote %s: %s (NDCG@10 en test %s)", lote, nombre(spec),
             f"{metricas['ndcg_10']:.4f}" if "ndcg_10" in metricas else "sin medir")
    return spec, lote, metricas


def coleccion_ejemplo(juegos: np.ndarray) -> pd.DataFrame:
    """Colección de prueba: los juegos de EJEMPLO_BGG_IDS que estén, o los 4 primeros."""
    ids = [b for b in EJEMPLO_BGG_IDS if b in set(juegos.tolist())] or juegos[:4].tolist()
    return pd.DataFrame({"bgg_id": np.array(ids, dtype=np.int64),
                         "nota": np.linspace(9.0, 7.0, len(ids)),
                         "excluir": np.zeros(len(ids), dtype=bool)})


def main() -> int:
    load_dotenv(ROOT / ".env")
    configurar_logging("ml.log", carpeta=LOGS)
    inicio = time.perf_counter()
    log.info("Producción: inicio")
    try:
        conectar()
        spec, lote, metricas_test = ganadora_ultima_evaluacion()

        datos = preparar_valoraciones("csv")
        R = datos.matriz(AJUSTE, VALIDACION, TEST)
        modelo = construir(spec, Rasgos(datos.juegos, "csv"))
        t = time.perf_counter()
        modelo.ajustar(R)
        log.info("Producción: %s entrenado con el 100 %% (%s valoraciones) en %.0f s",
                 nombre(spec), f"{R.nnz:,}".replace(",", "."), time.perf_counter() - t)

        mlflow.set_experiment(EXPERIMENTO_PRODUCCION)
        with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(
                run_name=f"{nombre(spec)}-100%",
                tags={"fase": "produccion", "lote_seleccion": lote, "datos": origen_gold()}) as run:
            ruta = Path(tmp) / "modelo.npz"
            modelo.guardar(ruta, ids=datos.juegos)
            log.info("Producción: modelo guardado (%.1f MB)", ruta.stat().st_size / 1e6)
            mlflow.log_params({**spec, "entrenado_con": "todas (100 %)", "lote_seleccion": lote,
                               "n_valoraciones": R.nnz})
            mlflow.log_metrics({f"test_{k}": v for k, v in metricas_test.items()})
            info = mlflow.pyfunc.log_model(
                name="modelo", python_model=str(CODIGO_MODELO), artifacts={"modelo": str(ruta)},
                signature=FIRMA, input_example=coleccion_ejemplo(datos.juegos),
                registered_model_name=NOMBRE_REGISTRO,
                pip_requirements=[f"mlflow=={mlflow.__version__}", f"numpy=={np.__version__}",
                                  f"scipy=={scipy.__version__}", f"pandas=={pd.__version__}",
                                  f"siguiente-partida=={siguiente_partida.__version__}"])

        version = info.registered_model_version
        cliente = MlflowClient()
        cliente.update_model_version(
            NOMBRE_REGISTRO, version,
            description=(f"{nombre(spec)} entrenado con el 100 % de las valoraciones. Elegido en "
                         f"la validación del lote {lote}; cifras del test (entrenado con el 80 %) "
                         "en las etiquetas test_*."))
        for k, v in {"variante": nombre(spec), "lote_seleccion": lote, "run_id": run.info.run_id,
                     **{f"test_{m}": f"{x:.4f}" for m, x in metricas_test.items()}}.items():
            cliente.set_model_version_tag(NOMBRE_REGISTRO, version, k, v)
        cliente.set_registered_model_alias(NOMBRE_REGISTRO, ALIAS, version)
        log.info("Producción: registrado %s versión %s con el alias '%s'", NOMBRE_REGISTRO,
                 version, ALIAS)

        # Comprobación: cargarlo como lo hará la API y recomendar a la colección de ejemplo
        cargado = mlflow.pyfunc.load_model(f"models:/{NOMBRE_REGISTRO}@{ALIAS}")
        ejemplo = coleccion_ejemplo(datos.juegos)
        salida = cargado.predict(ejemplo, params={"k": 5})
        nombres = leer_catalogo("csv")[0].set_index("bgg_id")["nombre"]
        log.info("Comprobación: colección de ejemplo: %s",
                 ", ".join(str(nombres.get(b, b)) for b in ejemplo["bgg_id"]))
        for fila in salida.itertuples():
            log.info("Comprobación: %d. %s (porque tienes %s)", fila.posicion,
                     nombres.get(fila.bgg_id, fila.bgg_id),
                     nombres.get(fila.porque_bgg_id, "-") if fila.porque_bgg_id >= 0 else "-")
    except (ValueError, FileNotFoundError, MlflowException) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado creando el modelo de producción")
        return 1
    log.info("Producción: fin (%.0f s)", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())