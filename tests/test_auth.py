"""Pruebas de la comprobación de tokens de Firebase (api/auth.py), sin conexión con Google.

Los tokens se firman aquí con una clave RSA generada en la prueba, que hace de clave pública
de Google: el verificador la recibe en lugar de descargarla.

    python -m pytest tests/test_auth.py -q
"""
from __future__ import annotations

import datetime as dt
import time

import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from api import auth, bd, modelo
from api.auth import (ClavesNoDisponibles, TokenInvalido, Usuario, UsuarioActual,
                      VerificadorFirebase)
from api.config import Ajustes
from api.main import crear_app

PROYECTO = "siguiente-partida-test"
KID = "clave-1"


@pytest.fixture(scope="module")
def clave_privada():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def otra_clave():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class Descargas:
    """Hace de Google: devuelve las claves que se le den y cuenta cuántas veces se le llama."""

    def __init__(self, claves, max_age=3600):
        self.claves, self.max_age, self.veces, self.falla = claves, max_age, 0, False

    def __call__(self):
        self.veces += 1
        if self.falla:
            raise ClavesNoDisponibles("sin conexión")
        return dict(self.claves), self.max_age


class Reloj:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def google(clave_privada):
    return Descargas({KID: clave_privada.public_key()})


@pytest.fixture
def verificador(google):
    return VerificadorFirebase(PROYECTO, descargar=google, reloj=Reloj())


def token(clave, kid=KID, alg="RS256", quitar=(), **cambios) -> str:
    """ID token como los de Firebase. Por defecto, de un usuario de Google recién conectado."""
    ahora = int(time.time())
    datos = {
        "iss": f"https://securetoken.google.com/{PROYECTO}", "aud": PROYECTO,
        "sub": "uidGoogle123", "iat": ahora - 10, "exp": ahora + 3600, "auth_time": ahora - 10,
        "name": "Fernando", "email": "f@example.com", "picture": "https://foto",
        "firebase": {"sign_in_provider": "google.com",
                     "identities": {"google.com": ["123"], "email": ["f@example.com"]}},
    }
    datos.update(cambios)
    for campo in quitar:
        datos.pop(campo)
    return jwt.encode(datos, clave, algorithm=alg, headers={"kid": kid} if kid else None)


INVITADO = {"sub": "uidInvitado456", "firebase": {"sign_in_provider": "anonymous",
                                                  "identities": {}}}


# --------------------------------------------------------------------------- tokens válidos
def test_usuario_de_google(verificador, clave_privada):
    assert verificador.verificar(token(clave_privada)) == Usuario(
        uid="uidGoogle123", es_invitado=False, nombre_visible="Fernando")


def test_invitado_sin_nombre(verificador, clave_privada):
    u = verificador.verificar(token(clave_privada, quitar=["name"], **INVITADO))
    assert u == Usuario(uid="uidInvitado456", es_invitado=True, nombre_visible=None)


def test_invitado_enlazado_con_google_ya_no_es_invitado(verificador, clave_privada):
    # Mismo uid y sign_in_provider sigue en anonymous, pero ya tiene la identidad de Google
    u = verificador.verificar(token(clave_privada, sub="uidInvitado456", firebase={
        "sign_in_provider": "anonymous", "identities": {"google.com": ["123"]}}))
    assert u.uid == "uidInvitado456" and not u.es_invitado and u.nombre_visible == "Fernando"


def test_el_nombre_se_recorta_y_el_vacio_es_none(verificador, clave_privada):
    largo = verificador.verificar(token(clave_privada, name="  " + "x" * 150 + " "))
    assert largo.nombre_visible == "x" * auth.LARGO_NOMBRE
    assert verificador.verificar(token(clave_privada, name="   ")).nombre_visible is None


def test_tolera_unos_segundos_de_diferencia_de_reloj(verificador, clave_privada):
    ahora = int(time.time())
    verificador.verificar(token(clave_privada, iat=ahora + 20, auth_time=ahora + 20))
    verificador.verificar(token(clave_privada, exp=ahora - 20))


# --------------------------------------------------------------------------- tokens rechazados
def rechazado(verificador, tok: str) -> str:
    with pytest.raises(TokenInvalido) as e:
        verificador.verificar(tok)
    return str(e.value)


def test_caducado(verificador, clave_privada):
    assert "caducado" in rechazado(verificador, token(clave_privada, exp=int(time.time()) - 60))


def test_de_otro_proyecto(verificador, clave_privada):
    assert "otro proyecto" in rechazado(verificador, token(clave_privada, aud="otro-proyecto"))
    assert "(iss)" in rechazado(verificador, token(
        clave_privada, iss="https://securetoken.google.com/otro-proyecto"))


def test_firmado_con_otra_clave(verificador, otra_clave):
    assert "firma" in rechazado(verificador, token(otra_clave))


def test_otros_algoritmos_no_valen(verificador, clave_privada):
    assert "RS256" in rechazado(verificador, token("secreto-compartido-de-32-bytes!!", alg="HS256"))
    sin_firma = jwt.encode({"sub": "x"}, None, algorithm="none", headers={"kid": KID})
    assert "RS256" in rechazado(verificador, sin_firma)


def test_no_es_un_jwt(verificador):
    assert "JWT" in rechazado(verificador, "esto-no-es-un-token")


def test_fecha_futura(verificador, clave_privada):
    ahora = int(time.time())
    assert "(iat)" in rechazado(verificador, token(clave_privada, iat=ahora + 600))
    assert "auth_time" in rechazado(verificador, token(clave_privada, auth_time=ahora + 600))


@pytest.mark.parametrize("campo", ["exp", "iat", "aud", "iss", "sub", "auth_time"])
def test_campos_obligatorios(verificador, clave_privada, campo):
    assert campo in rechazado(verificador, token(clave_privada, quitar=[campo]))


@pytest.mark.parametrize("uid", ["", "x" * 129, "uid-con-ñ"])
def test_uid_vacio_o_que_no_cabe_en_la_tabla(verificador, clave_privada, uid):
    assert "uid" in rechazado(verificador, token(clave_privada, sub=uid))


def test_solo_google_e_invitado(verificador, clave_privada):
    otro = token(clave_privada, firebase={"sign_in_provider": "password", "identities": {}})
    assert "password" in rechazado(verificador, otro)
    assert "None" in rechazado(verificador, token(clave_privada, quitar=["firebase"]))


def test_id_de_proyecto_con_formato_raro():
    for malo in ["", "Mayusculas-No", "corto", "apiKey: AIza...", "a" * 31]:
        with pytest.raises(ValueError):
            VerificadorFirebase(malo)


# --------------------------------------------------------------------------- claves de Google
def test_las_claves_se_descargan_una_vez_y_se_renuevan_al_caducar(google, clave_privada):
    reloj = Reloj()
    v = VerificadorFirebase(PROYECTO, descargar=google, reloj=reloj)
    for _ in range(3):
        v.verificar(token(clave_privada))
    assert google.veces == 1
    reloj.t += 3599
    v.verificar(token(clave_privada))
    assert google.veces == 1
    reloj.t += 1
    v.verificar(token(clave_privada))
    assert google.veces == 2


def test_kid_nuevo_vuelve_a_descargar_pero_no_mas_de_una_vez_por_minuto(google, clave_privada,
                                                                       otra_clave):
    reloj = Reloj()
    v = VerificadorFirebase(PROYECTO, descargar=google, reloj=reloj)
    v.verificar(token(clave_privada))
    google.claves["clave-2"] = otra_clave.public_key()      # Google rota las claves
    reloj.t += auth.REINTENTO_CLAVES
    assert v.verificar(token(otra_clave, kid="clave-2")).uid == "uidGoogle123"
    assert google.veces == 2
    for _ in range(5):                                       # kid inventado: no se insiste
        rechazado(v, token(clave_privada, kid="inventado"))
    assert google.veces == 2


def test_sin_conexion_sigue_con_las_claves_anteriores(google, clave_privada):
    reloj = Reloj()
    v = VerificadorFirebase(PROYECTO, descargar=google, reloj=reloj)
    v.verificar(token(clave_privada))
    google.falla = True
    reloj.t += 3600
    assert v.verificar(token(clave_privada)).uid == "uidGoogle123"
    assert google.veces == 2


def test_sin_conexion_y_sin_claves_da_error(google, clave_privada):
    google.falla = True
    with pytest.raises(ClavesNoDisponibles):
        VerificadorFirebase(PROYECTO, descargar=google).verificar(token(clave_privada))


class RespuestaFalsa:
    def __init__(self, contenido, cabeceras):
        self._contenido, self.headers = contenido, cabeceras

    def raise_for_status(self):
        pass

    def json(self):
        return self._contenido


def certificado(clave_privada) -> str:
    """Certificado X.509 autofirmado en PEM, como los que publica Google para securetoken."""
    nombre = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "securetoken.system")])
    ahora = dt.datetime.now(dt.UTC)
    cert = (x509.CertificateBuilder().subject_name(nombre).issuer_name(nombre)
            .public_key(clave_privada.public_key()).serial_number(1)
            .not_valid_before(ahora).not_valid_after(ahora + dt.timedelta(days=5))
            .sign(clave_privada, hashes.SHA256()))
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def test_descargar_claves_lee_los_certificados_y_el_max_age(monkeypatch, clave_privada):
    pedidas = []

    def get(url, timeout):
        pedidas.append(url)
        return RespuestaFalsa({KID: certificado(clave_privada)},
                              {"Cache-Control": "public, max-age=19876, must-revalidate"})
    monkeypatch.setattr(auth.requests, "get", get)
    claves, max_age = auth.descargar_claves()
    assert pedidas == [auth.URL_CLAVES] and max_age == 19876
    v = VerificadorFirebase(PROYECTO, descargar=lambda: (claves, max_age))
    assert v.verificar(token(clave_privada)).uid == "uidGoogle123"


def test_descargar_claves_con_respuesta_rara(monkeypatch):
    monkeypatch.setattr(auth.requests, "get", lambda url, timeout: RespuestaFalsa(
        {KID: "no es un certificado"}, {}))
    with pytest.raises(ClavesNoDisponibles):
        auth.descargar_claves()


# --------------------------------------------------------------------------- en las rutas
@pytest.fixture
def app_con_ruta(verificador):
    app = FastAPI()
    app.state.verificador = verificador

    @app.get("/yo")
    def yo(usuario: UsuarioActual) -> dict:
        return {"uid": usuario.uid, "es_invitado": usuario.es_invitado}
    return app


def test_ruta_con_token_valido(app_con_ruta, clave_privada):
    with TestClient(app_con_ruta) as c:
        r = c.get("/yo", headers={"Authorization": f"Bearer {token(clave_privada, **INVITADO)}"})
    assert r.status_code == 200 and r.json() == {"uid": "uidInvitado456", "es_invitado": True}


@pytest.mark.parametrize("cabecera", [None, "Bearer", "Basic dXNlcjpwYXNz", "Bearer basura"])
def test_ruta_sin_token_o_con_token_malo_da_401(app_con_ruta, cabecera):
    with TestClient(app_con_ruta) as c:
        r = c.get("/yo", headers={"Authorization": cabecera} if cabecera else {})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_ruta_sin_proyecto_configurado_da_503(app_con_ruta, clave_privada):
    app_con_ruta.state.verificador = None
    with TestClient(app_con_ruta) as c:
        r = c.get("/yo", headers={"Authorization": f"Bearer {token(clave_privada)}"})
    assert r.status_code == 503 and "FIREBASE_PROJECT_ID" in r.json()["detail"]


def test_ruta_sin_claves_de_google_da_503(app_con_ruta, google, clave_privada):
    google.falla = True
    with TestClient(app_con_ruta) as c:
        r = c.get("/yo", headers={"Authorization": f"Bearer {token(clave_privada)}"})
    assert r.status_code == 503


def test_el_token_no_aparece_en_el_registro(app_con_ruta, clave_privada, caplog):
    malo = token(clave_privada, exp=int(time.time()) - 60)
    with TestClient(app_con_ruta) as c:
        c.get("/yo", headers={"Authorization": f"Bearer {malo}"})
    assert "caducado" in caplog.text and malo not in caplog.text


# --------------------------------------------------------------------------- al arrancar
@pytest.mark.parametrize("proyecto, estado", [(PROYECTO, "ok"), (None, "sin configurar"),
                                              ("Proyecto Mal Escrito", "sin configurar")])
def test_salud_dice_si_las_cuentas_estan_configuradas(tmp_path, monkeypatch, google, proyecto,
                                                      estado):
    def sin_modelo(ajustes):
        raise ValueError("sin MLflow en esta prueba")
    monkeypatch.setattr(bd, "crear_motor", lambda ajustes: create_engine("sqlite://"))
    monkeypatch.setattr(modelo, "cargar", sin_modelo)
    monkeypatch.setattr(auth, "descargar_claves", google)
    ajustes = Ajustes("bd", 3306, "tfm", "x", "http://mlflow", "127.0.0.1", 8000, proyecto)
    with TestClient(crear_app(ajustes)) as c:
        cuentas = c.get("/salud").json()["cuentas"]
    assert cuentas["estado"] == estado
    assert google.veces == (1 if estado == "ok" else 0)     # las claves se precargan al arrancar