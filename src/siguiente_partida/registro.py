"""Configuración única del logging para todo el proyecto."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"
_MARCA = "_siguiente_partida"  # distingue los handlers de los de otras librerías


def configurar_logging(
    fichero: str = "siguiente_partida.log",
    carpeta: str | os.PathLike | None = None,
    nivel: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    copias: int = 5,
) -> logging.Logger:
    """Configura el logger raíz con un fichero rotativo y salida por consola.

    La carpeta se toma, por este orden, del argumento, de la variable de entorno
    SIGUIENTE_PARTIDA_LOGS o de ./logs. Usa un fichero distinto por proceso
    (descarga, pipeline, API...): en Windows, dos procesos escribiendo en el mismo
    fichero rotativo hacen fallar la rotación.

    Se puede llamar varias veces: sustituye sus propios handlers en vez de duplicarlos.
    """
    ruta = Path(carpeta or os.environ.get("SIGUIENTE_PARTIDA_LOGS", "logs"))
    ruta.mkdir(parents=True, exist_ok=True)

    raiz = logging.getLogger()
    raiz.setLevel(nivel)
    for h in [h for h in raiz.handlers if getattr(h, _MARCA, False)]:
        raiz.removeHandler(h)
        h.close()

    formato = logging.Formatter(FORMATO, FORMATO_FECHA)
    a_fichero = RotatingFileHandler(
        ruta / fichero, maxBytes=max_bytes, backupCount=copias, encoding="utf-8"
    )
    a_consola = logging.StreamHandler()
    for h in (a_fichero, a_consola):
        h.setFormatter(formato)
        setattr(h, _MARCA, True)
        raiz.addHandler(h)
    return raiz