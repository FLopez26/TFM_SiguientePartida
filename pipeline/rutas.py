"""Rutas de las capas y elección del origen de datos (Bronze completo o muestra de ejemplo)."""
from __future__ import annotations

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
EJEMPLO = DATA / "ejemplo"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
LOGS = ROOT / "logs"

SUB_KAGGLE = "kaggle_threnjen"
SUB_RANKING = "bgg_ranks"
FECHA_RANKING = "2026-11-12"
MANIFIESTO = "_manifiesto.json"

log = logging.getLogger("rutas")


def resolver_origen(pedido: str = "auto") -> tuple[str, Path]:
    """Devuelve ("bronze" | "ejemplo", carpeta).

    De forma predeterminada se usa Bronze, si existe el manifiesto. Si no, se usa el ejemplo.
    """
    hay_bronze = (BRONZE / MANIFIESTO).exists()
    if pedido == "bronze" and not hay_bronze:
        raise FileNotFoundError("Bronze está vacío: ejecuta scripts/descargar_bronze.py")
    if pedido == "bronze" or (pedido == "auto" and hay_bronze):
        return "bronze", BRONZE
    return "ejemplo", EJEMPLO