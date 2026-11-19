"""Métricas de recomendación top-k e intervalos de confianza por bootstrap.

Convenciones:
    top          array (usuarios x k) con los índices de los juegos recomendados, en orden
    relevantes   matriz dispersa (usuarios x juegos); cualquier valor distinto de 0 = relevante
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def aciertos(top: np.ndarray, relevantes) -> np.ndarray:
    """Matriz booleana del tamaño de top: True si ese juego recomendado es relevante."""
    top = np.asarray(top)
    rel = sparse.csr_matrix(relevantes, copy=True)   # copia: no se toca la matriz recibida
    rel.eliminate_zeros()
    rel.sort_indices()
    n, k = top.shape
    if rel.shape[0] != n:
        raise ValueError("top y relevantes tienen que tener las mismas filas")
    if rel.nnz == 0:
        return np.zeros((n, k), dtype=bool)
    n_juegos = np.int64(rel.shape[1])
    # Cada par (usuario, juego) como un entero: las claves de rel ya salen ordenadas
    claves_rel = np.repeat(np.arange(n, dtype=np.int64), np.diff(rel.indptr)) * n_juegos + rel.indices
    claves_top = np.repeat(np.arange(n, dtype=np.int64), k) * n_juegos + top.ravel()
    pos = np.minimum(np.searchsorted(claves_rel, claves_top), len(claves_rel) - 1)
    return (claves_rel[pos] == claves_top).reshape(n, k)


def metricas_por_usuario(top: np.ndarray, relevantes, k: int | None = None) -> dict[str, np.ndarray]:
    """precision@k, recall@k y NDCG@k (ganancia binaria) de cada usuario.

    Los usuarios sin ningún relevante salen con NaN: no se pueden puntuar.
    recall@k = aciertos / relevantes del usuario. El NDCG ideal usa min(relevantes, k).
    """
    top = np.asarray(top)
    k = top.shape[1] if k is None else k
    rel = sparse.csr_matrix(relevantes, copy=True)
    rel.eliminate_zeros()
    hits = aciertos(top[:, :k], rel)
    n_rel = np.diff(rel.indptr).astype(np.float64)
    tp = hits.sum(axis=1).astype(np.float64)
    descuento = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.r_[np.nan, np.cumsum(descuento)][np.minimum(n_rel, k).astype(np.int64)]
    with np.errstate(divide="ignore", invalid="ignore"):
        salida = {
            "precision": np.where(n_rel > 0, tp / k, np.nan),
            "recall": np.where(n_rel > 0, tp / n_rel, np.nan),
            "ndcg": np.where(n_rel > 0, (hits @ descuento) / ideal, np.nan),
        }
    return salida


def cobertura(top: np.ndarray, n_juegos: int) -> float:
    """Proporción del catálogo que aparece en la lista de algún usuario."""
    return float(np.unique(np.asarray(top)).size / n_juegos)


def ic_bootstrap(valores, n_remuestras: int = 1000, nivel: float = 0.95,
                 semilla: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media e intervalo de confianza por percentiles, remuestreando usuarios con reemplazo.

    valores puede ser un vector (un usuario por posición) o una matriz (filas = series sobre
    los mismos usuarios). Con una matriz, todas las filas usan los mismos remuestreos.
    Devuelve (media, inferior, superior); escalares si valores es un vector.
    """
    v = np.asarray(valores, dtype=np.float64)
    vector = v.ndim == 1
    v = np.atleast_2d(v)
    if np.isnan(v).any():
        raise ValueError("hay NaN: quita antes los usuarios que no se pueden puntuar")
    n = v.shape[1]
    rng = np.random.default_rng(semilla)
    medias = np.empty((n_remuestras, v.shape[0]))
    for b in range(n_remuestras):
        medias[b] = v[:, rng.integers(0, n, n)].mean(axis=1)
    alfa = (1 - nivel) / 2
    inf, sup = np.quantile(medias, [alfa, 1 - alfa], axis=0)
    media = v.mean(axis=1)
    if vector:
        return float(media[0]), float(inf[0]), float(sup[0])
    return media, inf, sup


def ic_bootstrap_pareado(a, b, n_remuestras: int = 1000, nivel: float = 0.95,
                         semilla: int = 0) -> tuple[float, float, float]:
    """Diferencia de medias a - b sobre los mismos usuarios, con su IC por bootstrap.

    Se remuestrean usuarios y en cada remuestra se compara a cada usuario consigo mismo.
    Como la variación entre usuarios afecta igual a los dos modelos, el intervalo suele ser
    mucho más estrecho que comparar los dos IC por separado.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("a y b tienen que ser de los mismos usuarios, en el mismo orden")
    return ic_bootstrap(a - b, n_remuestras, nivel, semilla)