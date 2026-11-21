"""Pruebas de la API con un Gold y un modelo de prueba, sin MySQL ni servidor de MLflow.

- Gold: las tablas que lee la API, en un SQLite con el nombre de esquema "gold" (ATTACH), así
  que las consultas son las mismas que en MySQL. Lo que solo hace MySQL (no distinguir tildes)
  no se prueba aquí.
- Modelo: un item-kNN de la librería entrenado con notas inventadas y servido con la misma
  clase que registra ml/produccion.py (ml/modelo_mlflow.py). Solo se salta la descarga desde
  MLflow, que ya comprueba /salud al arrancar la API de verdad.

    python -m pytest tests/test_api.py -q
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import sparse
from sqlalchemy import create_engine, event

from api import bd, modelo
from api.config import Ajustes
from api.main import crear_app
from api.modelo import Recomendador
from ml.modelo_mlflow import RecomendadorJuegos
from siguiente_partida.recomendadores import ItemKNN

AJUSTES = Ajustes("bd-de-prueba", 3306, "tfm", "x", "http://mlflow-de-prueba", "127.0.0.1", 8000)

# --------------------------------------------------------------------------- datos de juguete
# bgg_id: (nombre, peso, min, max, duración, n_valoraciones_bgg)
JUEGOS = {
    1: ("Catan", 2.3, 3, 4, 90, 1000),
    2: ("Carcassonne", 1.9, 2, 5, 45, 900),
    3: ("Pandemic", 2.4, 2, 4, 45, 800),
    4: ("Azul", 1.8, 2, 4, 40, 700),
    5: ("Twilight Struggle", 3.6, 2, 2, 180, 600),
    6: ("Codenames", 1.3, 2, 8, 15, 500),
    7: ("Gloomhaven", 3.9, 1, 4, 120, 400),
    8: ("Catan: Seafarers", 2.4, 3, 4, 90, 300),
    9: ("Prototipo 2000", 2.0, 2, 4, 30, 10),       # sin valoraciones: el modelo no lo conoce
    10: ("Wildcard 100%", None, 2, 4, None, 20),    # sin peso ni duración
    11: ("Viaje a Catania", 2.0, 2, 4, 60, 5000),   # tampoco está en el modelo
}
# Números de jugadores votados por la comunidad; Codenames, Wildcard y los dos últimos no tienen
JUGADORES = {1: [3, 4], 2: [2, 3, 4], 3: [2, 3, 4], 4: [2, 3, 4], 5: [2], 7: [1, 2, 3, 4],
             8: [3, 4]}
ETIQUETAS = {
    1: ("categoria", "Strategy"), 2: ("categoria", "Family"), 3: ("categoria", "Abstract"),
    4: ("categoria", "War"), 5: ("categoria", "Party"), 6: ("categoria", "Thematic"),
    10: ("mecanica", "Dice Rolling"), 11: ("mecanica", "Tile Placement"),
    12: ("mecanica", "Cooperative Game"), 13: ("mecanica", "Card Driven"),
    14: ("mecanica", "Team-Based Game"),
    20: ("tema", "Economic"), 21: ("tema", "Medieval"), 22: ("tema", "Medical"),
    23: ("tema", "Political"), 24: ("tema", "Spies"), 25: ("tema", "Fantasy"),
    26: ("tema", "Pirates"),                        # sin juegos
}
JUEGO_ETIQUETA = {1: [1, 10, 20], 2: [2, 11, 21], 3: [1, 12, 22], 4: [2, 3, 11], 5: [4, 13, 23],
                  6: [5, 14, 24], 7: [6, 12, 25], 8: [1, 10, 20], 9: [1], 10: [5], 11: [2]}

# Modelo: cada juego lo valoran los n primeros usuarios, así que la popularidad (número de
# valoraciones) los ordena sin empates: Catan, Pandemic, Carcassonne, Azul, Codenames...
VALORACIONES = {1: 9, 3: 8, 2: 7, 4: 6, 6: 5, 7: 4, 8: 3, 5: 2, 10: 1}
IDS_MODELO = np.array(sorted(VALORACIONES), dtype=np.int64)


def cumple_jugadores(bgg_id: int, n: int) -> bool:
    if bgg_id in JUGADORES:
        return n in JUGADORES[bgg_id]
    _, _, minimo, maximo, _, _ = JUEGOS[bgg_id]
    return minimo <= n <= maximo


@pytest.fixture(scope="module")
def ruta_gold(tmp_path_factory) -> str:
    ruta = str(tmp_path_factory.mktemp("gold") / "gold.db")
    juego = pd.DataFrame([
        {"bgg_id": b, "nombre": n, "anio": 2000 + b, "imagen_url": f"https://img/{b}.jpg",
         "peso": p, "min_jugadores": mi, "max_jugadores": ma, "mejor_n_jugadores": None,
         "duracion_fabricante": d, "duracion_min_comunidad": d, "duracion_max_comunidad": d,
         "edad_min_fabricante": 10, "edad_rec_comunidad": 9.5, "nota_media_bgg": 7.0,
         "n_valoraciones_bgg": v, "rank_general": b, "familia": None, "n_expansiones": 0}
        for b, (n, p, mi, ma, d, v) in JUEGOS.items()])
    with closing(sqlite3.connect(ruta)) as con:
        juego.to_sql("juego", con, index=False)
        pd.DataFrame([(i, t, n) for i, (t, n) in ETIQUETAS.items()],
                     columns=["id_etiqueta", "tipo", "nombre"]).to_sql("etiqueta", con, index=False)
        pd.DataFrame([(b, e) for b, es in JUEGO_ETIQUETA.items() for e in es],
                     columns=["bgg_id", "id_etiqueta"]).to_sql("juego_etiqueta", con, index=False)
        pd.DataFrame([(b, n) for b, ns in JUGADORES.items() for n in ns],
                     columns=["bgg_id", "n_jugadores"]).to_sql("juego_jugadores", con, index=False)
        con.commit()
    return ruta


def motor_sqlite(ruta: str):
    motor = create_engine("sqlite://", connect_args={"check_same_thread": False})

    @event.listens_for(motor, "connect")
    def _adjuntar_gold(con, _):
        con.execute(f"ATTACH DATABASE '{ruta}' AS gold")
    return motor


class PyFuncDePrueba:
    """Hace de PyFuncModel: llama a la clase de ml/modelo_mlflow.py sin servidor de MLflow."""

    def __init__(self, python_model: RecomendadorJuegos):
        self._modelo = python_model

    def predict(self, datos, params=None):
        return self._modelo.predict(None, datos, params)

    def unwrap_python_model(self):
        return self._modelo


@pytest.fixture(scope="module")
def recomendador(tmp_path_factory) -> Recomendador:
    filas = [u for b in IDS_MODELO for u in range(VALORACIONES[b])]
    cols = [i for i, b in enumerate(IDS_MODELO) for _ in range(VALORACIONES[b])]
    R = sparse.csr_matrix((np.full(len(filas), 8.0, np.float32), (filas, cols)),
                          shape=(10, len(IDS_MODELO)))
    ruta = tmp_path_factory.mktemp("modelo") / "modelo.npz"
    ItemKNN(vecinos=3, variante="binaria").ajustar(R).guardar(ruta, ids=IDS_MODELO)
    python_model = RecomendadorJuegos()
    python_model.load_context(SimpleNamespace(artifacts={"modelo": str(ruta)}))
    return Recomendador(modelo=PyFuncDePrueba(python_model), version="7",
                        variante="item_knn-binaria-k3", juegos=frozenset(IDS_MODELO.tolist()))


@pytest.fixture
def cliente(ruta_gold, recomendador, monkeypatch):
    monkeypatch.setattr(bd, "crear_motor", lambda ajustes: motor_sqlite(ruta_gold))
    monkeypatch.setattr(modelo, "cargar", lambda ajustes: recomendador)
    with TestClient(crear_app(AJUSTES)) as c:
        yield c


def recomendar(cliente, **peticion) -> dict:
    r = cliente.post("/recomendaciones", json=peticion)
    assert r.status_code == 200, r.text
    return r.json()


def ids(respuesta: dict) -> list[int]:
    return [x["juego"]["bgg_id"] for x in respuesta["recomendaciones"]]


# --------------------------------------------------------------------------- sistema
def test_salud_con_todo_bien(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    d = r.json()
    assert d["mysql"] == {"estado": "ok", "juegos": len(JUEGOS)}
    assert d["modelo"]["version"] == "7" and d["modelo"]["juegos"] == len(IDS_MODELO)


def test_sin_modelo_salud_y_recomendaciones_dan_503(ruta_gold, monkeypatch):
    def falla(ajustes):
        raise ValueError("MLflow no responde")
    monkeypatch.setattr(bd, "crear_motor", lambda ajustes: motor_sqlite(ruta_gold))
    monkeypatch.setattr(modelo, "cargar", falla)
    with TestClient(crear_app(AJUSTES)) as c:
        assert c.get("/salud").status_code == 503
        assert c.post("/recomendaciones", json={}).status_code == 503
        assert c.get("/juegos", params={"q": "catan"}).status_code == 200   # el catálogo sigue


def test_sin_base_de_datos_da_503(tmp_path, recomendador, monkeypatch):
    rota = create_engine(f"sqlite:///{tmp_path}/no_existe/gold.db")
    monkeypatch.setattr(bd, "crear_motor", lambda ajustes: rota)
    monkeypatch.setattr(modelo, "cargar", lambda ajustes: recomendador)
    with TestClient(crear_app(AJUSTES)) as c:
        assert c.get("/salud").json()["mysql"]["estado"] == "error"
        r = c.get("/juegos", params={"q": "catan"})
        assert r.status_code == 503 and r.json()["detail"] == "La base de datos no responde"


# --------------------------------------------------------------------------- catálogo
def test_buscar_ordena_exacto_luego_empieza_y_luego_contiene(cliente):
    r = cliente.get("/juegos", params={"q": "catan"})
    # Viaje a Catania es el más valorado, pero solo contiene el texto: va el último
    assert [j["bgg_id"] for j in r.json()] == [1, 8, 11]


def test_buscar_trata_porcentaje_como_texto(cliente):
    r = cliente.get("/juegos", params={"q": "0%"})
    assert [j["bgg_id"] for j in r.json()] == [10]   # no "Prototipo 2000"


def test_buscar_valida_parametros(cliente):
    assert cliente.get("/juegos", params={"q": "c"}).status_code == 422
    assert cliente.get("/juegos", params={"q": "catan", "limite": 51}).status_code == 422
    assert len(cliente.get("/juegos", params={"q": "an", "limite": 2}).json()) == 2


def test_ficha_de_un_juego(cliente):
    f = cliente.get("/juegos/1").json()
    assert f["nombre"] == "Catan" and f["url_bgg"] == "https://boardgamegeek.com/boardgame/1"
    assert f["jugadores_recomendados"] == [3, 4]
    assert f["etiquetas"] == {"categoria": ["Strategy"], "subcategoria": [],
                              "mecanica": ["Dice Rolling"], "tema": ["Economic"]}
    assert cliente.get("/juegos/6").json()["jugadores_recomendados"] == []
    assert cliente.get("/juegos/999").status_code == 404


def test_etiquetas_por_tipo_con_su_numero_de_juegos(cliente):
    temas = cliente.get("/etiquetas", params={"tipo": "tema"}).json()
    assert temas[0] == {"id_etiqueta": 20, "tipo": "tema", "nombre": "Economic", "n_juegos": 2}
    assert temas[-1]["nombre"] == "Pirates" and temas[-1]["n_juegos"] == 0
    assert {e["tipo"] for e in temas} == {"tema"}
    assert len(cliente.get("/etiquetas").json()) == len(ETIQUETAS)
    assert cliente.get("/etiquetas", params={"tipo": "otro"}).status_code == 422


# --------------------------------------------------------------------------- recomendaciones
def test_sin_coleccion_recomienda_por_popularidad(cliente):
    d = recomendar(cliente, k=5)
    assert ids(d) == [1, 3, 2, 4, 6]
    assert all(x["porque"] is None for x in d["recomendaciones"])
    assert d["modelo_version"] == "7"


def test_no_recomienda_ni_la_coleccion_ni_los_excluidos(cliente):
    d = recomendar(cliente, coleccion=[{"bgg_id": 1}, {"bgg_id": 3, "nota": 9}],
                   excluir=[2, 4], k=50)
    assert set(ids(d)) == set(VALORACIONES) - {1, 2, 3, 4}
    assert d["candidatos"] == len(VALORACIONES) - 4
    assert [x["posicion"] for x in d["recomendaciones"]] == list(range(1, len(ids(d)) + 1))


def test_porque_es_siempre_un_juego_de_la_coleccion(cliente):
    d = recomendar(cliente, coleccion=[{"bgg_id": 1}, {"bgg_id": 5}], k=50)
    porques = {x["porque"]["bgg_id"] for x in d["recomendaciones"] if x["porque"]}
    assert porques and porques <= {1, 5}


def test_juego_en_coleccion_y_en_excluir_cuenta_como_excluido(cliente):
    d = recomendar(cliente, coleccion=[{"bgg_id": 1}, {"bgg_id": 3}], excluir=[3], k=50)
    assert 3 not in ids(d)
    assert all(x["porque"] is None or x["porque"]["bgg_id"] == 1 for x in d["recomendaciones"])


def test_juegos_que_el_modelo_no_conoce_se_ignoran(cliente):
    d = recomendar(cliente, coleccion=[{"bgg_id": 1}, {"bgg_id": 9}, {"bgg_id": 999}], k=3)
    assert d["ignorados"] == [9, 999]
    assert d["candidatos"] == len(VALORACIONES) - 1


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_filtrar_antes_da_lo_mismo_que_ordenar_todo_y_filtrar_despues(cliente, recomendador, n):
    coleccion = [1, 2]
    todo = recomendador.recomendar(pd.DataFrame({"bgg_id": coleccion}),
                                   k=len(IDS_MODELO) - len(coleccion))
    esperado = [b for b in todo["bgg_id"] if cumple_jugadores(b, n)][:3]
    d = recomendar(cliente, coleccion=[{"bgg_id": b} for b in coleccion],
                   filtros={"jugadores": n}, k=3)
    assert ids(d) == esperado


def test_jugadores_sin_votos_usa_el_minimo_y_el_maximo(cliente):
    assert ids(recomendar(cliente, filtros={"jugadores": 7})) == [6]   # Codenames, 2-8
    assert ids(recomendar(cliente, filtros={"jugadores": 1})) == [7]   # Gloomhaven, votado


def test_duracion_y_peso(cliente):
    d = recomendar(cliente, filtros={"duracion_max": 45}, k=50)
    assert set(ids(d)) == {2, 3, 4, 6}
    d = recomendar(cliente, filtros={"duracion_max": 45, "peso_min": 1.5}, k=50)
    assert set(ids(d)) == {2, 3, 4}


def test_un_dato_vacio_no_cumple_el_filtro(cliente):
    assert 10 in ids(recomendar(cliente, k=50))
    assert 10 not in ids(recomendar(cliente, filtros={"peso_max": 5}, k=50))


def test_etiquetas_o_dentro_del_tipo_e_y_entre_tipos(cliente):
    medieval_o_medico = {"etiquetas": [21, 22]}
    assert set(ids(recomendar(cliente, filtros=medieval_o_medico))) == {2, 3}
    con_losetas = {"etiquetas": [21, 22, 11]}      # y además Tile Placement
    assert ids(recomendar(cliente, filtros=con_losetas)) == [2]


def test_la_coleccion_cuenta_aunque_no_cumpla_los_filtros(cliente):
    # Twilight Struggle es solo para 2: con 4 jugadores no se recomienda nada parecido a él
    # por cumplir el filtro, pero sigue explicando recomendaciones
    d = recomendar(cliente, coleccion=[{"bgg_id": 5}], filtros={"jugadores": 4}, k=50)
    assert {x["porque"]["bgg_id"] for x in d["recomendaciones"] if x["porque"]} == {5}


def test_sin_candidatos_devuelve_lista_vacia(cliente):
    d = recomendar(cliente, coleccion=[{"bgg_id": 7}], filtros={"jugadores": 1})
    assert d["recomendaciones"] == [] and d["candidatos"] == 0
    d = recomendar(cliente, filtros={"etiquetas": [26]})                 # Pirates: sin juegos
    assert d["recomendaciones"] == []


@pytest.mark.parametrize("peticion", [
    {"jugadore": 2},                                         # campo mal escrito
    {"filtros": {"jugadore": 2}},
    {"filtros": {"peso_min": 3, "peso_max": 2}},             # rango al revés
    {"filtros": {"duracion_min": 60, "duracion_max": 30}},
    {"coleccion": [{"bgg_id": 1, "nota": 11}]},              # nota fuera de 1-10
    {"coleccion": [{"bgg_id": 0}]},
    {"k": 0},
    {"k": 51},
    {"filtros": {"etiquetas": [999]}},                       # etiqueta que no existe
])
def test_peticiones_invalidas_dan_422(cliente, peticion):
    assert cliente.post("/recomendaciones", json=peticion).status_code == 422