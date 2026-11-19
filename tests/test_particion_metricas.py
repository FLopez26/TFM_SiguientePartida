"""Pruebas de la partición por usuario y de las métricas de la librería."""
import numpy as np
import pytest
from scipy import sparse

from siguiente_partida.metricas import (aciertos, cobertura, ic_bootstrap, ic_bootstrap_pareado,
                                        metricas_por_usuario)
from siguiente_partida.particion import reservar_por_usuario


# ----------------------------------------------------------------------------- partición
def test_particion_reserva_el_20_por_ciento_de_cada_usuario():
    usuarios = np.repeat([1, 2, 3, 4], [10, 11, 50, 1])
    r = reservar_por_usuario(usuarios, 0.2, semilla=1)
    por_usuario = {u: int(r[usuarios == u].sum()) for u in (1, 2, 3, 4)}
    assert por_usuario == {1: 2, 2: 2, 3: 10, 4: 0}   # 10*0.2=2; 11*0.2=2.2->2; 1 fila: nada


def test_particion_es_reproducible_y_cambia_con_la_semilla():
    usuarios = np.repeat(np.arange(500), 20)
    a = reservar_por_usuario(usuarios, 0.2, semilla=7)
    assert (a == reservar_por_usuario(usuarios, 0.2, semilla=7)).all()
    assert (a != reservar_por_usuario(usuarios, 0.2, semilla=8)).any()


def test_particion_deja_siempre_algo_para_entrenar():
    usuarios = np.repeat([1, 2], [2, 3])
    r = reservar_por_usuario(usuarios, 0.9, semilla=0)
    assert r[usuarios == 1].sum() == 1 and r[usuarios == 2].sum() == 2


# ----------------------------------------------------------------------------- métricas
def test_metricas_a_mano():
    # usuario 0: relevantes {1, 3}; recomendado [3, 0, 1] -> aciertos en posiciones 1 y 3
    # usuario 1: relevantes {2};    recomendado [0, 1, 4] -> ningún acierto
    # usuario 2: sin relevantes -> NaN
    rel = sparse.csr_matrix(np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]))
    top = np.array([[3, 0, 1], [0, 1, 4], [0, 1, 2]])
    assert aciertos(top, rel).tolist() == [[True, False, True], [False] * 3, [False] * 3]
    m = metricas_por_usuario(top, rel)
    assert m["precision"][:2] == pytest.approx([2 / 3, 0])
    assert m["recall"][:2] == pytest.approx([1, 0])
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg"][:2] == pytest.approx([dcg / idcg, 0])
    assert np.isnan(m["ndcg"][2]) and np.isnan(m["recall"][2])
    assert cobertura(top, 5) == pytest.approx(5 / 5)


def test_ic_bootstrap_contiene_la_media_y_el_pareado_es_mas_estrecho():
    rng = np.random.default_rng(0)
    base = rng.random(2000)
    mejor = base + 0.02 + rng.normal(0, 0.01, 2000)
    media, inf, sup = ic_bootstrap(mejor, 500)
    assert inf < media < sup
    d, d_inf, d_sup = ic_bootstrap_pareado(mejor, base, 500)
    assert d == pytest.approx(0.02, abs=0.002) and d_inf > 0
    assert (d_sup - d_inf) < (sup - inf)
    with pytest.raises(ValueError):
        ic_bootstrap([1.0, np.nan])


def test_metricas_no_modifican_la_entrada_y_ignoran_ceros_explicitos():
    rel = sparse.csr_matrix((np.array([1.0, 0.0]), (np.array([0, 0]), np.array([1, 2]))), shape=(1, 4))
    assert rel.nnz == 2                       # el 0 está guardado de forma explícita
    m = metricas_por_usuario(np.array([[2, 1]]), rel)
    assert rel.nnz == 2                       # la matriz de fuera sigue igual
    assert m["recall"][0] == pytest.approx(1.0)   # solo cuenta el juego 1 como relevante