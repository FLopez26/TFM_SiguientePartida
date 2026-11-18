"""Utilidades genéricas del recomendador de juegos de mesa Siguiente Partida."""
from siguiente_partida.registro import configurar_logging
from siguiente_partida.seudonimizacion import es_seudonimo, seudonimizar, seudonimo

__all__ = ["configurar_logging", "es_seudonimo", "seudonimizar", "seudonimo"]
__version__ = "0.0.1"