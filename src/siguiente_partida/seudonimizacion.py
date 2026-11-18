"""Seudonimización de identificadores con HMAC-SHA256 y una sal secreta.

El mismo valor con la misma sal da siempre el mismo seudónimo, así que se pueden cruzar
tablas sin guardar el valor original. Sin la sal no se puede recalcular el seudónimo a partir
de un nombre conocido.

"""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable

LONGITUD_POR_DEFECTO = 16  


def _bytes(sal: str | bytes) -> bytes:
    sal = sal.encode("utf-8") if isinstance(sal, str) else sal
    if len(sal) < 16:
        raise ValueError("La sal debe tener al menos 16 bytes")
    return sal


def seudonimo(valor: str, sal: str | bytes, longitud: int = LONGITUD_POR_DEFECTO) -> str:
    """Seudónimo hexadecimal de `valor`."""
    return hmac.new(_bytes(sal), valor.encode("utf-8"), hashlib.sha256).hexdigest()[:longitud]


def seudonimizar(valores: Iterable[str], sal: str | bytes,
                 longitud: int = LONGITUD_POR_DEFECTO) -> list[str]:
    """Seudónimos de una colección de valores, en el mismo orden.

    Lanza ValueError si dos valores distintos dan el mismo seudónimo.
    """
    clave = _bytes(sal)
    valores = list(valores)
    resultado = [hmac.new(clave, v.encode("utf-8"), hashlib.sha256).hexdigest()[:longitud]
                 for v in valores]
    if len(set(resultado)) != len(set(valores)):
        raise ValueError("Colisión de seudónimos: aumenta la longitud")
    return resultado


def es_seudonimo(valor: str, longitud: int = LONGITUD_POR_DEFECTO) -> bool:
    """True si `valor` tiene forma de seudónimo (hex en minúsculas de la longitud indicada)."""
    return len(valor) == longitud and all(c in "0123456789abcdef" for c in valor)