"""API de Siguiente Partida: catálogo (Gold en MySQL) y recomendaciones (modelo de MLflow).

Uso, desde la raíz y con MySQL y MLflow arrancados (docker compose up -d mysql mlflow):
    python -m api.main
Documentación interactiva: http://127.0.0.1:8000/docs. Registro: logs/api.log.

Si al arrancar falla MySQL o el modelo, la API arranca igualmente y lo registra como ERROR;
/salud responde 503 y dice qué parte falla.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from api import bd, catalogo, modelo, recomendaciones
from api.config import ALIAS, LOGS, NOMBRE_REGISTRO, Ajustes, leer_ajustes
from siguiente_partida import configurar_logging

log = logging.getLogger("api")

sistema = APIRouter(tags=["sistema"])


class SinSaludCorrecta(logging.Filter):
    """Quita del registro las llamadas a /salud que responden 200. En Docker, el healthcheck la
    llama cada 15 s y llenaría el fichero; las que fallan (503) sí se registran."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.access registra con args = (cliente, método, ruta, versión HTTP, estado)
        args = record.args
        return not (isinstance(args, tuple) and len(args) == 5
                    and args[2] == "/salud" and args[4] == 200)


@sistema.get("/", include_in_schema=False)
def raiz() -> RedirectResponse:
    return RedirectResponse("/docs")


@sistema.get("/salud")
def salud(request: Request) -> JSONResponse:
    """Estado de MySQL y del modelo. 200 si los dos están bien; 503 si falla alguno."""
    try:
        mysql = {"estado": "ok", "juegos": bd.contar_juegos(request.app.state.motor)}
    except SQLAlchemyError as e:
        mysql = {"estado": "error", "detalle": str(getattr(e, "orig", e))}
    rec = request.app.state.recomendador
    if rec is None:
        mod = {"estado": "error", "detalle": "no cargado: mira logs/api.log"}
    else:
        mod = {"estado": "ok", "nombre": NOMBRE_REGISTRO, "alias": ALIAS,
               "version": rec.version, "variante": rec.variante, "juegos": len(rec.juegos)}
    ok = mysql["estado"] == "ok" and mod["estado"] == "ok"
    return JSONResponse(status_code=200 if ok else 503,
                        content={"estado": "ok" if ok else "error", "mysql": mysql, "modelo": mod})


def crear_app(ajustes: Ajustes) -> FastAPI:
    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        log.info("API: inicio")
        app.state.motor = bd.crear_motor(ajustes)
        try:
            n = bd.contar_juegos(app.state.motor)
            log.info("MySQL: conectado a %s:%d (gold.juego: %s juegos)", ajustes.mysql_host,
                     ajustes.mysql_port, f"{n:,}".replace(",", "."))
        except SQLAlchemyError as e:
            log.error("MySQL: no se puede leer gold.juego en %s:%d: %s", ajustes.mysql_host,
                      ajustes.mysql_port, getattr(e, "orig", e))
        app.state.recomendador = None
        try:
            app.state.recomendador = modelo.cargar(ajustes)
        except ValueError as e:
            log.error("Modelo: %s", e)
        except Exception:
            log.exception("Modelo: error inesperado al cargarlo de MLflow")
        yield
        app.state.motor.dispose()
        log.info("API: fin")

    app = FastAPI(
        title="Siguiente Partida",
        description="Catálogo de juegos de mesa (BoardGameGeek) y recomendaciones.",
        version="0.1.0",
        lifespan=ciclo_de_vida,
    )

    @app.exception_handler(OperationalError)
    async def mysql_no_responde(request: Request, exc: OperationalError) -> JSONResponse:
        log.error("MySQL: %s %s: %s", request.method, request.url.path, exc.orig)
        return JSONResponse(status_code=503, content={"detail": "La base de datos no responde"})

    app.include_router(sistema)
    app.include_router(catalogo.router)
    app.include_router(recomendaciones.router)
    return app


def main() -> int:
    configurar_logging("api.log", carpeta=LOGS)
    logging.getLogger("uvicorn.access").addFilter(SinSaludCorrecta())
    try:
        ajustes = leer_ajustes()
    except ValueError as e:
        log.error("%s", e)
        return 1
    # log_config=None: uvicorn no configura su propio logging, así que sus mensajes (arranque
    # y una línea por petición) van también a logs/api.log
    uvicorn.run(crear_app(ajustes), host=ajustes.host, port=ajustes.puerto, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())