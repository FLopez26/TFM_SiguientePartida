"""Pruebas de las rutas de /cuenta con un Gold y un esquema app de juguete, sin MySQL.

- Gold y app: dos ficheros SQLite adjuntados con los nombres "gold" y "app" (ATTACH), así que
  las consultas son las mismas que en MySQL. El esquema app copia las claves y las
  restricciones de docker/mysql/init/02_app.sql.
- Tokens: un verificador falso que entiende "google:<uid>:<nombre>" e "invitado:<uid>". La
  comprobación de los tokens de verdad está en test_auth.py.
- Interbloqueos: SQLite no los tiene, así que se simula el error 1213 de MySQL.

    python -m pytest tests/test_cuentas.py -q
"""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError

from api import bd, consultas_app, modelo
from api.auth import TokenInvalido, Usuario
from api.config import Ajustes
from api.main import crear_app

AJUSTES = Ajustes("bd-de-prueba", 3306, "tfm", "x", "http://mlflow-de-prueba", "127.0.0.1", 8000)
JUEGOS = {13: "CATAN", 822: "Carcassonne", 9209: "Ticket to Ride", 30549: "Pandemic",
          174430: "Gloomhaven"}
ESQUEMA_APP = """
CREATE TABLE cuenta (
    id_cuenta       INTEGER PRIMARY KEY AUTOINCREMENT,
    firebase_uid    TEXT NOT NULL UNIQUE,
    nombre_visible  TEXT,
    es_invitado     BOOLEAN NOT NULL DEFAULT 1,
    creada_en       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ultimo_acceso   DATETIME
);
CREATE TABLE cuenta_juego (
    id_cuenta       INTEGER NOT NULL REFERENCES cuenta (id_cuenta) ON DELETE CASCADE,
    bgg_id          INTEGER NOT NULL,
    lista           TEXT NOT NULL CHECK (lista IN ('coleccion', 'deseo', 'descarte')),
    nota            NUMERIC CHECK (nota IS NULL OR (lista = 'coleccion' AND nota BETWEEN 1 AND 10)),
    actualizado_en  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_cuenta, bgg_id)
);
"""


class VerificadorFalso:
    proyecto = "proyecto-de-prueba"

    def verificar(self, token: str) -> Usuario:
        tipo, _, resto = token.partition(":")
        uid, _, nombre = resto.partition(":")
        if tipo not in ("google", "invitado") or not uid:
            raise TokenInvalido("token de prueba mal formado")
        invitado = tipo == "invitado"
        return Usuario(uid=uid, es_invitado=invitado,
                       nombre_visible=None if invitado else (nombre or None))


def como(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


GOOGLE = como("google:uidGoogle:Fernando")
INVITADO = como("invitado:uidInvitado")


@pytest.fixture
def motor(tmp_path):
    gold, app = tmp_path / "gold.db", tmp_path / "app.db"
    with closing(sqlite3.connect(gold)) as con:
        con.execute("CREATE TABLE juego (bgg_id INTEGER PRIMARY KEY, nombre TEXT, anio INTEGER, "
                    "imagen_url TEXT, n_valoraciones_bgg INTEGER)")
        con.executemany("INSERT INTO juego VALUES (?, ?, 2000, NULL, 100)", JUEGOS.items())
        con.commit()
    with closing(sqlite3.connect(app)) as con:
        con.executescript(ESQUEMA_APP)
    m = create_engine("sqlite://", connect_args={"check_same_thread": False})

    @event.listens_for(m, "connect")
    def _adjuntar(con, _):
        con.execute(f"ATTACH DATABASE '{gold}' AS gold")
        con.execute(f"ATTACH DATABASE '{app}' AS app")
    return m


@pytest.fixture
def cliente(motor, monkeypatch):
    monkeypatch.setattr(bd, "crear_motor", lambda ajustes: motor)
    monkeypatch.setattr(modelo, "cargar", lambda ajustes: None)   # aquí no hace falta el modelo
    with TestClient(crear_app(AJUSTES)) as c:
        c.app.state.verificador = VerificadorFalso()
        yield c


def consulta(motor, sql: str, **params) -> list[tuple]:
    with motor.connect() as c:
        return [tuple(f) for f in c.execute(text(sql), params)]


def guardar(cliente, bgg_id: int, lista: str, nota=None, quien=GOOGLE) -> dict:
    cuerpo = {"lista": lista} if nota is None else {"lista": lista, "nota": nota}
    r = cliente.put(f"/cuenta/juegos/{bgg_id}", json=cuerpo, headers=quien)
    assert r.status_code == 200, r.text
    return r.json()


def juegos(cliente, quien=GOOGLE, **params) -> dict[int, tuple]:
    r = cliente.get("/cuenta/juegos", params=params, headers=quien)
    assert r.status_code == 200, r.text
    return {j["bgg_id"]: (j["lista"], j["nota"]) for j in r.json()}


# --------------------------------------------------------------------------- token
@pytest.mark.parametrize("metodo, ruta", [
    ("post", "/cuenta/sesion"), ("get", "/cuenta"), ("delete", "/cuenta"),
    ("post", "/cuenta/fusionar"), ("get", "/cuenta/juegos"), ("put", "/cuenta/juegos/13"),
    ("delete", "/cuenta/juegos/13")])
def test_todas_las_rutas_piden_token(cliente, motor, metodo, ruta):
    assert cliente.request(metodo, ruta, json={}).status_code == 401
    assert cliente.request(metodo, ruta, json={}, headers=como("basura")).status_code == 401
    assert consulta(motor, "SELECT COUNT(*) FROM app.cuenta") == [(0,)]


def test_sin_firebase_configurado_da_503(cliente):
    cliente.app.state.verificador = None
    assert cliente.get("/cuenta", headers=GOOGLE).status_code == 503
    assert cliente.get("/juegos", params={"q": "catan"}).status_code == 200   # el catálogo sigue


# --------------------------------------------------------------------------- sesión y perfil
def test_la_primera_sesion_crea_la_cuenta_y_la_segunda_no_la_duplica(cliente, motor):
    for _ in range(2):
        r = cliente.post("/cuenta/sesion", headers=GOOGLE)
        assert r.status_code == 200
    d = r.json()
    assert d["nombre_visible"] == "Fernando" and d["es_invitado"] is False
    assert d["juegos"] == {"coleccion": 0, "deseo": 0, "descarte": 0}
    assert consulta(motor, "SELECT firebase_uid, ultimo_acceso IS NOT NULL FROM app.cuenta") == [
        ("uidGoogle", 1)]


def test_invitado_sin_nombre(cliente):
    d = cliente.post("/cuenta/sesion", headers=INVITADO).json()
    assert d["es_invitado"] is True and d["nombre_visible"] is None


def test_el_uid_distingue_mayusculas(cliente, motor):
    cliente.get("/cuenta", headers=como("invitado:abc"))
    cliente.get("/cuenta", headers=como("invitado:ABC"))
    assert consulta(motor, "SELECT COUNT(*) FROM app.cuenta") == [(2,)]


def test_invitado_enlazado_con_google_deja_de_ser_invitado(cliente):
    guardar(cliente, 13, "coleccion", 9, quien=INVITADO)
    d = cliente.post("/cuenta/sesion", headers=como("google:uidInvitado:Fernando")).json()
    assert d["es_invitado"] is False and d["nombre_visible"] == "Fernando"
    assert d["juegos"]["coleccion"] == 1                     # mismo uid: conserva sus juegos


def test_sin_nombre_en_el_token_se_conserva_el_guardado(cliente):
    cliente.post("/cuenta/sesion", headers=GOOGLE)
    d = cliente.post("/cuenta/sesion", headers=como("google:uidGoogle")).json()
    assert d["nombre_visible"] == "Fernando"


def test_cualquier_ruta_crea_la_cuenta_si_no_existe(cliente):
    assert guardar(cliente, 822, "deseo")["lista"] == "deseo"
    assert cliente.get("/cuenta", headers=GOOGLE).json()["juegos"]["deseo"] == 1


# --------------------------------------------------------------------------- listas
def test_guardar_un_juego_devuelve_su_resumen(cliente):
    d = guardar(cliente, 13, "coleccion", 8.5)
    assert (d["bgg_id"], d["lista"], d["nota"], d["juego"]["nombre"]) == (
        13, "coleccion", 8.5, "CATAN")
    assert d["actualizado_en"]


def test_un_juego_esta_en_una_sola_lista(cliente, motor):
    guardar(cliente, 13, "coleccion", 8)
    guardar(cliente, 13, "descarte")
    assert juegos(cliente) == {13: ("descarte", None)}
    guardar(cliente, 13, "coleccion")
    assert juegos(cliente) == {13: ("coleccion", None)}      # la nota no vuelve
    assert consulta(motor, "SELECT COUNT(*) FROM app.cuenta_juego") == [(1,)]


def test_guardar_lo_mismo_dos_veces_no_falla(cliente):
    for _ in range(2):
        guardar(cliente, 13, "coleccion", 7)
    assert juegos(cliente) == {13: ("coleccion", 7.0)}


def test_la_nota_se_guarda_con_un_decimal(cliente):
    assert guardar(cliente, 13, "coleccion", 7.25)["nota"] == 7.3
    assert guardar(cliente, 822, "coleccion", 7.24)["nota"] == 7.2


def test_filtrar_por_lista(cliente):
    guardar(cliente, 13, "coleccion", 9)
    guardar(cliente, 822, "deseo")
    guardar(cliente, 9209, "descarte")
    assert juegos(cliente, lista="deseo") == {822: ("deseo", None)}
    assert set(juegos(cliente)) == {13, 822, 9209}
    assert cliente.get("/cuenta", headers=GOOGLE).json()["juegos"] == {
        "coleccion": 1, "deseo": 1, "descarte": 1}
    assert cliente.get("/cuenta/juegos", params={"lista": "otra"},
                       headers=GOOGLE).status_code == 422


@pytest.mark.parametrize("cuerpo", [
    {"lista": "deseo", "nota": 7},           # nota fuera de la colección
    {"lista": "coleccion", "nota": 0.5},
    {"lista": "coleccion", "nota": 11},
    {"lista": "favoritos"},
    {"lista": "coleccion", "puntuacion": 7},  # campo desconocido
    {},
])
def test_peticiones_invalidas_dan_422(cliente, motor, cuerpo):
    assert cliente.put("/cuenta/juegos/13", json=cuerpo, headers=GOOGLE).status_code == 422
    assert consulta(motor, "SELECT COUNT(*) FROM app.cuenta_juego") == [(0,)]


def test_juego_que_no_esta_en_el_catalogo_da_404(cliente):
    r = cliente.put("/cuenta/juegos/999999", json={"lista": "deseo"}, headers=GOOGLE)
    assert r.status_code == 404
    assert cliente.put("/cuenta/juegos/0", json={"lista": "deseo"},
                       headers=GOOGLE).status_code == 422


def test_juego_guardado_que_ya_no_esta_en_el_catalogo(cliente, motor):
    guardar(cliente, 13, "coleccion")
    with motor.begin() as c:   # como si se regenerase Gold sin ese juego
        c.execute(text("DELETE FROM gold.juego WHERE bgg_id = 13"))
    r = cliente.get("/cuenta/juegos", headers=GOOGLE).json()
    assert r[0]["bgg_id"] == 13 and r[0]["juego"] is None


def test_quitar_un_juego(cliente):
    guardar(cliente, 13, "coleccion")
    guardar(cliente, 822, "deseo")
    for _ in range(2):    # la segunda vez ya no está: tampoco falla
        assert cliente.delete("/cuenta/juegos/13", headers=GOOGLE).status_code == 204
    assert juegos(cliente) == {822: ("deseo", None)}


def test_cada_cuenta_ve_solo_sus_juegos(cliente):
    guardar(cliente, 13, "coleccion", 9)
    guardar(cliente, 822, "deseo", quien=INVITADO)
    cliente.delete("/cuenta/juegos/822", headers=GOOGLE)      # no es suyo: no lo toca
    assert juegos(cliente) == {13: ("coleccion", 9.0)}
    assert juegos(cliente, quien=INVITADO) == {822: ("deseo", None)}


# --------------------------------------------------------------------------- borrar la cuenta
def test_borrar_la_cuenta_borra_sus_juegos(cliente, motor):
    guardar(cliente, 13, "coleccion")
    guardar(cliente, 822, "deseo")
    guardar(cliente, 9209, "deseo", quien=INVITADO)
    assert cliente.delete("/cuenta", headers=GOOGLE).status_code == 204
    assert consulta(motor, "SELECT c.firebase_uid, cj.bgg_id FROM app.cuenta c "
                           "JOIN app.cuenta_juego cj USING (id_cuenta)") == [("uidInvitado", 9209)]
    assert consulta(motor, "SELECT COUNT(*) FROM app.cuenta") == [(1,)]
    assert cliente.delete("/cuenta", headers=GOOGLE).status_code == 204   # ya no existe


# --------------------------------------------------------------------------- fusionar
def fusionar(cliente, quien=GOOGLE, token_invitado="invitado:uidInvitado"):
    return cliente.post("/cuenta/fusionar", json={"token_invitado": token_invitado},
                        headers=quien)


def test_fusionar_mueve_los_juegos_y_respeta_los_de_google(cliente, motor):
    guardar(cliente, 13, "coleccion", 9)                      # la cuenta de Google ya tenía
    guardar(cliente, 822, "deseo")
    guardar(cliente, 13, "descarte", quien=INVITADO)          # choca con el de Google
    guardar(cliente, 9209, "coleccion", 6, quien=INVITADO)
    guardar(cliente, 30549, "descarte", quien=INVITADO)
    r = fusionar(cliente)
    assert r.status_code == 200, r.text
    d = r.json()
    assert (d["movidos"], d["ya_estaban"]) == (2, 1)
    assert d["cuenta"]["juegos"] == {"coleccion": 2, "deseo": 1, "descarte": 1}
    assert juegos(cliente) == {13: ("coleccion", 9.0), 822: ("deseo", None),
                               9209: ("coleccion", 6.0), 30549: ("descarte", None)}
    assert consulta(motor, "SELECT firebase_uid FROM app.cuenta") == [("uidGoogle",)]


def test_fusionar_dos_veces_no_hace_nada_la_segunda(cliente):
    guardar(cliente, 13, "coleccion", quien=INVITADO)
    assert fusionar(cliente).json()["movidos"] == 1
    d = fusionar(cliente).json()
    assert (d["movidos"], d["ya_estaban"]) == (0, 0) and d["cuenta"]["juegos"]["coleccion"] == 1


def test_fusionar_un_invitado_sin_cuenta(cliente):
    d = fusionar(cliente, token_invitado="invitado:nuncaGuardoNada").json()
    assert (d["movidos"], d["ya_estaban"]) == (0, 0)


@pytest.mark.parametrize("quien, token_invitado, codigo", [
    (INVITADO, "invitado:otroInvitado", 422),             # el destino tiene que ser de Google
    (GOOGLE, "google:otraGoogle:Ana", 422),                # el origen tiene que ser invitado
    (como("google:uidInvitado:Fernando"), "invitado:uidInvitado", 422),   # misma cuenta
    (GOOGLE, "basura", 401),                               # token del invitado no válido
])
def test_fusionar_con_cuentas_que_no_valen(cliente, motor, quien, token_invitado, codigo):
    guardar(cliente, 13, "coleccion", quien=como("invitado:otroInvitado"))
    guardar(cliente, 822, "coleccion", quien=INVITADO)
    assert fusionar(cliente, quien, token_invitado).status_code == codigo
    assert consulta(motor, "SELECT COUNT(*) FROM app.cuenta_juego") == [(2,)]   # nada movido


# --------------------------------------------------------------------------- interbloqueos
def error_mysql(codigo: int) -> OperationalError:
    return OperationalError("UPDATE app.cuenta_juego ...", {}, Exception(codigo, "error de MySQL"))


def test_escribir_repite_si_mysql_cancela_por_interbloqueo(motor, caplog):
    llamadas = []

    def operacion():
        llamadas.append(1)
        if len(llamadas) == 1:
            raise error_mysql(consultas_app.INTERBLOQUEO)
        return "hecho"
    with motor.connect() as c:
        assert consultas_app.escribir(c, operacion) == "hecho"
    assert len(llamadas) == 2 and "interbloqueo" in caplog.text


@pytest.mark.parametrize("codigo, veces", [(2013, 1),       # conexión perdida: no se repite
                                           (consultas_app.INTERBLOQUEO, consultas_app.INTENTOS)])
def test_escribir_no_repite_otros_errores_ni_sin_limite(motor, codigo, veces):
    llamadas = []

    def operacion():
        llamadas.append(1)
        raise error_mysql(codigo)
    with motor.connect() as c, pytest.raises(OperationalError):
        consultas_app.escribir(c, operacion)
    assert len(llamadas) == veces