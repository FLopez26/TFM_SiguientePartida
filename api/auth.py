"""Inicio de sesión: comprobación de los tokens de Firebase Auth (Google o invitado anónimo).

El navegador inicia sesión con el SDK de Firebase y manda en cada petición de /cuenta la
cabecera "Authorization: Bearer <ID token>". La API comprueba el token por su cuenta con las
claves públicas de Google, siguiendo la guía de Firebase para librerías JWT de terceros
(https://firebase.google.com/docs/auth/admin/verify-id-tokens):

    cabecera    alg = RS256 y kid = una de las claves públicas de securetoken
    exp         en el futuro            iat y auth_time   en el pasado
    aud         el id del proyecto      iss   https://securetoken.google.com/<proyecto>
    sub         no vacío: es el uid del usuario

No se usa firebase-admin porque pide una cuenta de servicio, que sería un secreto más que
guardar. Así solo hace falta el id del proyecto, que no es secreto (FIREBASE_PROJECT_ID).
Límite: sin cuenta de servicio no se puede saber si un token se ha revocado, así que un token
vale hasta que caduca (Firebase los emite para 1 hora).

Del token solo se usan el uid, si es invitado y el nombre visible. El correo y la foto que trae
el token de Google se ignoran: no se guardan ni se registran.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
import requests
from cryptography import x509
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("api.auth")

URL_CLAVES = ("https://www.googleapis.com/robot/v1/metadata/x509/"
              "securetoken@system.gserviceaccount.com")
MARGEN_RELOJ = 30          # segundos de tolerancia entre el reloj del PC y el de Google
REINTENTO_CLAVES = 60      # con un kid desconocido o un fallo, no se vuelve a descargar antes
PROVEEDORES = {"anonymous", "google.com"}   # los dos inicios de sesión del plan
LARGO_UID = 128            # app.cuenta.firebase_uid
LARGO_NOMBRE = 100         # app.cuenta.nombre_visible
# Id de proyecto de Firebase/Google Cloud: de 6 a 30 caracteres, minúsculas, cifras y guiones
FORMATO_PROYECTO = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]")

Claves = dict[str, Any]    # kid -> clave pública RSA


class TokenInvalido(Exception):
    """El token está mal formado, ha caducado o no es de este proyecto (401)."""


class ClavesNoDisponibles(Exception):
    """No se han podido descargar las claves públicas de Google (503)."""


@dataclass(frozen=True)
class Usuario:
    uid: str
    es_invitado: bool
    nombre_visible: str | None     # None en invitados


def descargar_claves(timeout: float = 5) -> tuple[Claves, int]:
    """Claves públicas de securetoken y cuántos segundos valen (max-age de Cache-Control)."""
    try:
        r = requests.get(URL_CLAVES, timeout=timeout)
        r.raise_for_status()
        claves = {kid: x509.load_pem_x509_certificate(pem.encode()).public_key()
                  for kid, pem in r.json().items()}
    except (requests.RequestException, ValueError, AttributeError) as e:
        raise ClavesNoDisponibles(f"{e.__class__.__name__}: {e}") from None
    if not claves:
        raise ClavesNoDisponibles("Google ha devuelto una lista de claves vacía")
    max_age = re.search(r"max-age=(\d+)", r.headers.get("Cache-Control", ""))
    return claves, int(max_age.group(1)) if max_age else 3600


# Errores de PyJWT -> motivo que se devuelve en el 401
MOTIVOS = [
    (jwt.ExpiredSignatureError, "ha caducado"),
    (jwt.ImmatureSignatureError, "tiene fecha futura (iat): revisa la hora del PC"),
    (jwt.InvalidAudienceError, "es de otro proyecto de Firebase (aud)"),
    (jwt.InvalidIssuerError, "no lo ha emitido Firebase para este proyecto (iss)"),
    (jwt.InvalidSignatureError, "la firma no es válida"),
    (jwt.DecodeError, "no es un JWT válido"),
]


class VerificadorFirebase:
    """Comprueba ID tokens de un proyecto de Firebase. Guarda las claves públicas en memoria
    el tiempo que indica Google y las vuelve a pedir al caducar o si llega un kid nuevo (Google
    las rota cada pocos días). Se comparte entre peticiones simultáneas: usa un cerrojo."""

    def __init__(self, proyecto: str, descargar: Callable[[], tuple[Claves, int]] | None = None,
                 reloj: Callable[[], float] = time.monotonic):
        if not FORMATO_PROYECTO.fullmatch(proyecto):
            raise ValueError(f"FIREBASE_PROJECT_ID no parece un id de proyecto: {proyecto!r} "
                             "(se ve en la consola de Firebase, Configuración del proyecto)")
        self.proyecto = proyecto
        self.emisor = f"https://securetoken.google.com/{proyecto}"
        self._descargar = descargar
        self._reloj = reloj
        self._claves: Claves = {}
        self._caducan = -math.inf          # cuándo hay que renovar las claves
        self._ultimo_intento = -math.inf
        self._cerrojo = threading.Lock()

    def precargar(self) -> int:
        """Descarga las claves al arrancar la API, para ver pronto si hay conexión. Devuelve
        cuántas hay."""
        with self._cerrojo:
            self._renovar(self._reloj())
        return len(self._claves)

    def _renovar(self, ahora: float) -> None:
        self._ultimo_intento = ahora
        descargar = self._descargar or descargar_claves   # se busca aquí para los tests
        self._claves, max_age = descargar()
        self._caducan = ahora + max_age
        log.info("Cuentas: %d claves públicas de Google descargadas (valen %d s)",
                 len(self._claves), max_age)

    def _clave(self, kid: str) -> Any:
        with self._cerrojo:
            ahora = self._reloj()
            puede_reintentar = ahora - self._ultimo_intento >= REINTENTO_CLAVES
            if ahora >= self._caducan or (kid not in self._claves and puede_reintentar):
                try:
                    self._renovar(ahora)
                except ClavesNoDisponibles as e:
                    if not self._claves:
                        raise
                    # Con claves anteriores se sigue: Google las mantiene válidas unos días
                    # después de rotarlas. Se reintenta en REINTENTO_CLAVES segundos
                    self._caducan = ahora + REINTENTO_CLAVES
                    log.warning("Cuentas: no se pueden renovar las claves (%s); se usan las "
                                "anteriores", e)
            return self._claves.get(kid)

    def verificar(self, token: str) -> Usuario:
        try:
            cabecera = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError:
            raise TokenInvalido("no es un JWT válido") from None
        if cabecera.get("alg") != "RS256":
            raise TokenInvalido(f"algoritmo {cabecera.get('alg')!r} en vez de RS256")
        kid = cabecera.get("kid")
        clave = self._clave(kid) if isinstance(kid, str) else None
        if clave is None:
            raise TokenInvalido("lo ha firmado una clave que Google no publica (kid)")
        try:
            datos = jwt.decode(
                token, clave, algorithms=["RS256"], audience=self.proyecto, issuer=self.emisor,
                leeway=MARGEN_RELOJ,
                options={"require": ["exp", "iat", "aud", "iss", "sub", "auth_time"]})
        except jwt.MissingRequiredClaimError as e:
            raise TokenInvalido(f"le falta el campo {e.claim}") from None
        except jwt.InvalidTokenError as e:
            motivo = next((m for tipo, m in MOTIVOS if isinstance(e, tipo)), str(e))
            raise TokenInvalido(motivo) from None

        uid = datos["sub"]
        if not isinstance(uid, str) or not uid or len(uid) > LARGO_UID or not uid.isascii():
            raise TokenInvalido("el uid (sub) está vacío o no tiene un formato válido")
        auth_time = datos["auth_time"]
        if not isinstance(auth_time, int | float) or auth_time > time.time() + MARGEN_RELOJ:
            raise TokenInvalido("auth_time no está en el pasado")

        firebase = datos.get("firebase") or {}
        proveedor = firebase.get("sign_in_provider")
        if proveedor not in PROVEEDORES:
            raise TokenInvalido(f"inicio de sesión con {proveedor!r}: solo se admiten Google e "
                                "invitado")
        # Un invitado que enlaza su cuenta con Google conserva el uid y pasa a tener la
        # identidad google.com, aunque el token siga diciendo sign_in_provider = anonymous
        es_invitado = proveedor == "anonymous" and "google.com" not in (
            firebase.get("identities") or {})
        nombre = datos.get("name")
        if es_invitado or not isinstance(nombre, str) or not nombre.strip():
            nombre = None
        else:
            nombre = nombre.strip()[:LARGO_NOMBRE]
        return Usuario(uid=uid, es_invitado=es_invitado, nombre_visible=nombre)


# --------------------------------------------------------------------------- para las rutas
# auto_error=False: sin cabecera la API responde 401 con un mensaje propio. En /docs aparece el
# botón "Authorize" para pegar un token y probar las rutas
_bearer = HTTPBearer(auto_error=False, scheme_name="Firebase",
                     description="ID token de Firebase: await auth.currentUser.getIdToken()")
NO_AUTORIZADO = {"WWW-Authenticate": "Bearer"}


def comprobar_token(request: Request, token: str) -> Usuario:
    """Comprueba un token (el de la cabecera o, al fusionar cuentas, el del invitado) y traduce
    los fallos a errores HTTP."""
    verificador: VerificadorFirebase | None = request.app.state.verificador
    if verificador is None:
        raise HTTPException(status_code=503, detail="Las cuentas no están configuradas: falta "
                                                    "FIREBASE_PROJECT_ID (mira logs/api.log)")
    try:
        return verificador.verificar(token)
    except TokenInvalido as e:
        log.warning("Cuentas: token rechazado en %s %s: %s", request.method, request.url.path, e)
        raise HTTPException(status_code=401, detail=f"Token no válido: {e}",
                            headers=NO_AUTORIZADO) from None
    except ClavesNoDisponibles as e:
        log.error("Cuentas: no se pueden descargar las claves públicas de Google: %s", e)
        raise HTTPException(status_code=503, detail="No se puede comprobar el token: no se "
                                                    "pueden descargar las claves de Google") from None


def usuario_actual(
    request: Request,
    credenciales: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Usuario:
    """Dependencia de FastAPI: el usuario de la cabecera Authorization, ya comprobado."""
    if credenciales is None:
        raise HTTPException(status_code=401, headers=NO_AUTORIZADO,
                            detail="Falta la cabecera Authorization: Bearer <token de Firebase>")
    return comprobar_token(request, credenciales.credentials)


# Para las rutas: def ruta(usuario: UsuarioActual) recibe el usuario ya comprobado
UsuarioActual = Annotated[Usuario, Depends(usuario_actual)]