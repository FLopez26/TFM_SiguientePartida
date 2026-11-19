"""Partición de valoraciones por usuario: se reserva al azar una parte de las notas de cada uno."""
from __future__ import annotations

import numpy as np


def reservar_por_usuario(usuarios, fraccion: float = 0.2, semilla: int = 0) -> np.ndarray:
    """Máscara booleana de las filas reservadas (True).

    Para cada usuario con n filas se reservan al azar round(fraccion * n) filas (los .5 suben),
    al menos 1, y siempre queda al menos 1 sin reservar. Un usuario con una sola fila no
    reserva nada.

    El resultado depende de la semilla y del orden de las filas. Para que sea reproducible
    aunque cambie el orden de lectura, ordena antes por (usuario, juego).

    >>> reservar_por_usuario([1] * 10, 0.2, semilla=0).sum()
    2
    """
    if not 0 < fraccion < 1:
        raise ValueError("fraccion tiene que estar entre 0 y 1")
    usuarios = np.asarray(usuarios)
    n = len(usuarios)
    if n == 0:
        return np.zeros(0, dtype=bool)

    rng = np.random.default_rng(semilla)
    orden = np.lexsort((rng.random(n), usuarios))       # por usuario y, dentro, al azar
    u = usuarios[orden]
    inicio = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
    tamano = np.diff(np.r_[inicio, n])
    posicion = np.arange(n) - np.repeat(inicio, tamano)  # 0, 1, 2... dentro de cada usuario

    cuantas = np.floor(fraccion * tamano + 0.5).astype(np.int64)
    cuantas = np.minimum(np.maximum(cuantas, 1), tamano - 1)

    reservada = np.empty(n, dtype=bool)
    reservada[orden] = posicion < np.repeat(cuantas, tamano)
    return reservada