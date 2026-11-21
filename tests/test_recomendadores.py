"""Pruebas de los recomendadores de la librería."""
import numpy as np
import pytest
from scipy import sparse

from siguiente_partida.recomendadores import ItemKNN, PorContenido, Popularidad, vecinos_coseno

R = sparse.csr_matrix(np.array([
    # juegos: 0  1  2  3  4
    [9, 8, 0, 0, 0],
    [8, 9, 0, 0, 2],
    [9, 0, 0, 0, 0],   # le gusta el 0: el item-kNN debería proponerle el 1
    [0, 0, 7, 8, 0],
    [0, 0, 8, 7, 0],
], dtype=np.float32))


def test_vecinos_coseno_coincide_con_el_calculo_denso():
    M = sparse.random(50, 30, density=0.3, random_state=1, format="csr")
    idx, sim = vecinos_coseno(M, 5, bloque=7)
    D = M.toarray()
    D = D / np.maximum(np.linalg.norm(D, axis=0), 1e-12)
    S = D.T @ D
    np.fill_diagonal(S, -np.inf)
    esperado = np.sort(S, axis=1)[:, ::-1][:, :5]
    assert sim == pytest.approx(esperado, abs=1e-5)


def test_popularidad_cuenta_solo_lo_que_recibe():
    m = Popularidad("valoraciones").ajustar(R)
    assert m.popularidad_.tolist() == [3, 2, 2, 2, 1]
    assert Popularidad("positivas", 7).ajustar(R).popularidad_.tolist() == [3, 2, 2, 2, 0]
    top, _ = m.recomendar(R[2], k=2)
    assert 0 not in top[0]                  # el 0 ya lo tiene: no se recomienda
    assert set(top[0]) <= {1, 2, 3}         # los siguientes, empatados a 2 notas


def test_item_knn_recomienda_lo_parecido_y_no_lo_ya_valorado():
    m = ItemKNN(vecinos=2, variante="binaria").ajustar(R)
    assert m.recomendar(R[2], k=1)[0][0, 0] == 1
    # centrada: con una sola nota no hay señal (nota - media = 0); con dos, lo que puntuó
    # por debajo de su media no empuja a sus vecinos
    m = ItemKNN(vecinos=2, variante="centrada").ajustar(R)
    dos_notas = sparse.csr_matrix(([9.0, 5.0], ([0, 0], [0, 3])), shape=(1, 5))
    assert m.recomendar(dos_notas, k=1)[0][0, 0] == 1
    # un usuario nuevo (como uno de la app) con solo el juego 2
    nuevo = sparse.csr_matrix(([9.0], ([0], [2])), shape=(1, 5))
    assert ItemKNN(vecinos=2).ajustar(R).recomendar(nuevo, k=1)[0][0, 0] == 3


def test_item_knn_cambiar_vecinos_sin_reajustar():
    m = ItemKNN(vecinos=1, max_vecinos=3).ajustar(R)
    a = m.puntuar(R).copy()
    m.vecinos = 3
    b = m.puntuar(R)
    assert m.indices_.shape[1] == 3 and not np.allclose(a, b)
    m.vecinos = 4
    with pytest.raises(ValueError):
        m.puntuar(R)


def test_por_contenido_usa_los_rasgos():
    rasgos = sparse.csr_matrix(np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]], dtype=float))
    m = PorContenido(rasgos, vecinos=2, variante="binaria").ajustar(R)
    top, _ = m.recomendar(R[2], k=1)        # el 0 se parece al 2 por rasgos
    assert top[0, 0] == 2
    with pytest.raises(ValueError):
        PorContenido(rasgos[:3]).ajustar(R)


def test_sin_ajustar_o_con_otro_numero_de_juegos_falla():
    with pytest.raises(RuntimeError):
        ItemKNN().puntuar(R)
    with pytest.raises(ValueError):
        ItemKNN(vecinos=2).ajustar(R).puntuar(R[:, :3])

def test_guardar_y_cargar_recomienda_igual(tmp_path):
    from siguiente_partida.recomendadores import cargar_recomendador
    ids = np.array([101, 102, 103, 104, 105])
    for modelo in (Popularidad("valoraciones"), ItemKNN(vecinos=2, variante="centrada"),
                   PorContenido(sparse.eye(5, format="csr"), vecinos=2, variante="binaria")):
        modelo.ajustar(R)
        ruta = tmp_path / "modelo.npz"
        modelo.guardar(ruta, ids=ids)
        cargado, ids_cargados = cargar_recomendador(ruta)
        assert (ids_cargados == ids).all()
        a, b = modelo.recomendar(R, k=3), cargado.recomendar(R, k=3)
        assert (a[0] == b[0]).all() and np.allclose(a[1], b[1])
    with pytest.raises(RuntimeError):
        ItemKNN().guardar(tmp_path / "vacio.npz")


def test_explicar_da_el_juego_que_mas_aporta():
    m = ItemKNN(vecinos=2, variante="binaria").ajustar(R)
    usuario = sparse.csr_matrix(([9.0, 7.0], ([0, 0], [0, 2])), shape=(1, 5))
    top, _ = m.recomendar(usuario, k=2)
    porque = m.explicar(usuario, top[0])
    # el 1 se recomienda por el 0 (mismos usuarios); el 3, por el 2
    assert dict(zip(top[0].tolist(), porque.tolist())) == {1: 0, 3: 2}