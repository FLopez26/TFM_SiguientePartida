"""Recomendadores sobre una matriz dispersa usuarios x juegos con la nota (0 = sin nota).

Interfaz común:
    modelo.ajustar(R)                     aprende de las valoraciones de entrenamiento
    modelo.puntuar(R_usuarios)            puntuación de cada juego para cada fila de R_usuarios
    modelo.recomendar(R_usuarios, k=10)   los k juegos con más puntuación que no estén en R_usuarios

puntuar y recomendar reciben las notas del usuario como entrada. Por eso sirven igual para un
usuario de BGG del conjunto de evaluación que para alguien de la app con su colección.

Las puntuaciones solo sirven para ordenar los juegos de un mismo usuario: no son notas ni
probabilidades, y no se pueden comparar entre usuarios.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

VARIANTES = ("binaria", "centrada")
DESEMPATE = 1e-9  # peso de la popularidad para desempatar


def transformar(R, variante: str) -> sparse.csr_matrix:
    """binaria: 1 donde hay nota. centrada: nota menos la media de ese usuario (fila)."""
    X = sparse.csr_matrix(R, dtype=np.float32, copy=True)
    if variante == "binaria":
        X.data[:] = 1.0
    elif variante == "centrada":
        n = np.diff(X.indptr)
        media = np.divide(np.asarray(X.sum(axis=1)).ravel(), n, out=np.zeros(X.shape[0]),
                          where=n > 0)
        X.data -= np.repeat(media, n).astype(np.float32)
    else:
        raise ValueError(f"variante desconocida: {variante} (usa una de {VARIANTES})")
    return X


def vecinos_coseno(M, k: int, bloque: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Para cada columna de M (filas x columnas), sus k columnas más parecidas por coseno.

    Devuelve (indices, similitudes), de tamaño columnas x k, ordenados de más a menos
    parecido y sin la propia columna. Se calcula por bloques de columnas para no tener nunca
    en memoria la matriz completa de similitudes (columnas x columnas).
    """
    M = sparse.csr_matrix(M, dtype=np.float32)
    n = M.shape[1]
    k = min(k, n - 1)
    normas = np.sqrt(np.asarray(M.multiply(M).sum(axis=0)).ravel())
    inv = np.divide(1.0, normas, out=np.zeros_like(normas), where=normas > 0).astype(np.float32)
    MT = M.T.tocsr()

    indices = np.empty((n, k), dtype=np.int32)
    similitudes = np.empty((n, k), dtype=np.float32)
    for a in range(0, n, bloque):
        b = min(a + bloque, n)
        G = (MT[a:b] @ M).toarray()                    # (b - a) x n, productos escalares
        G *= inv[a:b, None]
        G *= inv[None, :]
        G[np.arange(b - a), np.arange(a, b)] = -np.inf  # fuera la propia columna
        parte = np.argpartition(G, -k, axis=1)[:, -k:]
        s = np.take_along_axis(G, parte, axis=1)
        orden = np.argsort(-s, axis=1, kind="stable")
        indices[a:b] = np.take_along_axis(parte, orden, axis=1)
        similitudes[a:b] = np.take_along_axis(s, orden, axis=1)
    return indices, similitudes


class Recomendador:
    """Base: guarda la popularidad del entrenamiento para desempatar y hace el top-k."""

    def parametros(self) -> dict:
        return {}

    def _preparar(self, R) -> sparse.csr_matrix:
        R = sparse.csr_matrix(R, dtype=np.float32)
        self.n_juegos_ = R.shape[1]
        cuenta = np.diff(R.tocsc().indptr).astype(np.float64)
        self.desempate_ = DESEMPATE * cuenta / max(cuenta.max(), 1.0)
        return R

    def _comprobar(self, R_usuarios) -> sparse.csr_matrix:
        if not hasattr(self, "n_juegos_"):
            raise RuntimeError("el modelo no está ajustado: llama antes a ajustar(R)")
        R = sparse.csr_matrix(R_usuarios, dtype=np.float32)
        if R.shape[1] != self.n_juegos_:
            raise ValueError(f"se esperaban {self.n_juegos_} juegos y llegan {R.shape[1]}")
        return R

    def ajustar(self, R) -> "Recomendador":
        raise NotImplementedError

    def puntuar(self, R_usuarios) -> np.ndarray:
        raise NotImplementedError

    def recomendar(self, R_usuarios, k: int = 10, excluir=None,
                   lote: int = 2048) -> tuple[np.ndarray, np.ndarray]:
        """Top-k de cada usuario: (indices de juego, puntuación), de más a menos puntuación.

        excluir: juegos que no se pueden recomendar (matriz dispersa con las mismas filas).
        Por defecto, los que ya están en R_usuarios. Los empates se deshacen por popularidad.
        """
        R = self._comprobar(R_usuarios)
        E = R if excluir is None else sparse.csr_matrix(excluir)
        n = R.shape[0]
        top = np.empty((n, k), dtype=np.int32)
        punt = np.empty((n, k), dtype=np.float32)
        for a in range(0, n, lote):
            b = min(a + lote, n)
            P = self.puntuar(R[a:b])
            P += self.desempate_
            Eb = E[a:b]
            P[np.repeat(np.arange(b - a), np.diff(Eb.indptr)), Eb.indices] = -np.inf
            parte = np.argpartition(P, -k, axis=1)[:, -k:]
            s = np.take_along_axis(P, parte, axis=1)
            orden = np.argsort(-s, axis=1, kind="stable")
            top[a:b] = np.take_along_axis(parte, orden, axis=1)
            punt[a:b] = np.take_along_axis(s, orden, axis=1)
        return top, punt


class Popularidad(Recomendador):
    """Línea base: los juegos más populares en el entrenamiento, iguales para todos.

    criterio="valoraciones": número de notas; "positivas": número de notas >= umbral.
    """

    def __init__(self, criterio: str = "positivas", umbral: float = 7.0):
        if criterio not in ("valoraciones", "positivas"):
            raise ValueError("criterio tiene que ser 'valoraciones' o 'positivas'")
        self.criterio, self.umbral = criterio, umbral

    def parametros(self) -> dict:
        return {"criterio": self.criterio, "umbral": self.umbral}

    def ajustar(self, R) -> "Popularidad":
        R = self._preparar(R).tocsc()
        if self.criterio == "valoraciones":
            cuenta = np.diff(R.indptr)
        else:
            cuenta = np.asarray((R >= self.umbral).sum(axis=0)).ravel()
        self.popularidad_ = cuenta.astype(np.float64)
        return self

    def puntuar(self, R_usuarios) -> np.ndarray:
        R = self._comprobar(R_usuarios)
        return np.tile(self.popularidad_, (R.shape[0], 1))


class ItemKNN(Recomendador):
    """Vecinos por juego (item-kNN) con similitud coseno entre columnas de juegos.

    variante="binaria": cuenta quién ha valorado cada juego. variante="centrada": usa la nota
    menos la media del usuario, así que lo que puntuó por debajo de su media resta.

    Para cada juego se guardan sus `vecinos` juegos más parecidos (solo similitudes > 0).
    La puntuación de un juego candidato es la suma de sus similitudes con los juegos del
    usuario, ponderadas por el valor de cada uno (1 o la nota centrada).

    max_vecinos permite calcular la similitud una vez y probar después varios valores de
    `vecinos` (<= max_vecinos) cambiando el atributo, sin volver a ajustar.
    """

    def __init__(self, vecinos: int = 100, variante: str = "binaria",
                 max_vecinos: int | None = None, bloque: int = 1024):
        if variante not in VARIANTES:
            raise ValueError(f"variante desconocida: {variante} (usa una de {VARIANTES})")
        self.vecinos, self.variante = vecinos, variante
        self.max_vecinos, self.bloque = max_vecinos, bloque

    def parametros(self) -> dict:
        return {"vecinos": self.vecinos, "variante": self.variante}

    def ajustar(self, R) -> "ItemKNN":
        R = self._preparar(R)
        k = max(self.vecinos, self.max_vecinos or 0)
        self.indices_, self.similitudes_ = self._calcular_vecinos(R, k)
        self._matrices = {}
        return self

    def _calcular_vecinos(self, R, k: int):
        return vecinos_coseno(transformar(R, self.variante), k, self.bloque)

    def matriz_vecinos(self) -> sparse.csr_matrix:
        """Matriz juegos x juegos: fila i = similitudes de i con sus vecinos."""
        k = self.vecinos
        if k > self.indices_.shape[1]:
            raise ValueError(f"vecinos={k} es mayor que los calculados ({self.indices_.shape[1]})")
        if k not in self._matrices:
            n = self.indices_.shape[0]
            sim = self.similitudes_[:, :k]
            positivo = sim > 0
            filas = np.repeat(np.arange(n), k)[positivo.ravel()]
            S = sparse.csr_matrix((sim[positivo], (filas, self.indices_[:, :k][positivo])),
                                  shape=(n, n), dtype=np.float32)
            self._matrices = {k: S}  # solo la última, para no acumular memoria
        return self._matrices[k]

    def puntuar(self, R_usuarios) -> np.ndarray:
        X = transformar(self._comprobar(R_usuarios), self.variante)
        return (X @ self.matriz_vecinos()).toarray().astype(np.float64)


class PorContenido(ItemKNN):
    """Como ItemKNN, pero la similitud entre juegos sale de sus rasgos, no de las notas.

    rasgos: matriz juegos x rasgos (etiquetas, numéricos, texto...), en el mismo orden de
    juegos que las columnas de R. Las notas solo se usan para el perfil de cada usuario.
    """

    def __init__(self, rasgos, vecinos: int = 100, variante: str = "centrada",
                 max_vecinos: int | None = None, bloque: int = 1024):
        super().__init__(vecinos, variante, max_vecinos, bloque)
        self.rasgos = rasgos

    def _calcular_vecinos(self, R, k: int):
        if self.rasgos.shape[0] != R.shape[1]:
            raise ValueError("rasgos tiene que tener una fila por cada juego (columna de R)")
        return vecinos_coseno(sparse.csr_matrix(self.rasgos).T, k, self.bloque)