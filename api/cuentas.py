"""Rutas de la cuenta: inicio de sesión, perfil, paso de invitado a Google y listas de juegos.

Todas piden la cabecera "Authorization: Bearer <ID token de Firebase>" (ver api/auth.py). La
cuenta se crea sola la primera vez que llega un usuario, así que el orden de las llamadas del
frontal no importa. Aun así, el frontal llama a POST /cuenta/sesion después de cada inicio de
sesión, para guardar el nombre, si es invitado y el último acceso.

    POST   /cuenta/sesion             inicio de sesión: crea o actualiza la cuenta
    GET    /cuenta                    perfil: nombre, invitado y juegos de cada lista
    DELETE /cuenta                    borra la cuenta y sus juegos
    POST   /cuenta/fusionar           invitado -> cuenta de Google que ya existía
    GET    /cuenta/juegos?lista=      juegos guardados
    PUT    /cuenta/juegos/{bgg_id}    guardar un juego en una lista (o moverlo de lista)
    DELETE /cuenta/juegos/{bgg_id}    quitarlo

Paso de invitado a Google (lo decide el frontal según lo que responda Firebase):
    - enlazar funciona (linkWithPopup): el uid no cambia y basta con POST /cuenta/sesion con el
      token nuevo, que ya dice que no es invitado;
    - esa cuenta de Google ya existía (auth/credential-already-in-use): el frontal guarda antes
      el token del invitado, inicia sesión con Google y llama a POST /cuenta/fusionar.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response

from api import consultas_app
from api.auth import UsuarioActual, comprobar_token
from api.bd import Conexion
from api.esquemas import (Cuenta, GuardarJuego, JuegoGuardado, Lista, PeticionFusion,
                          ResultadoFusion)

log = logging.getLogger("api.cuentas")

ERRORES_TOKEN = {401: {"description": "Falta el token o no es válido"},
                 503: {"description": "Las cuentas no están configuradas o no se puede "
                                      "comprobar el token"}}
router = APIRouter(prefix="/cuenta", tags=["cuenta"], responses=ERRORES_TOKEN)


def _cuenta(usuario: UsuarioActual, c: Conexion) -> int:
    return consultas_app.obtener_o_crear_cuenta(c, usuario)


# Para las rutas: id_cuenta del usuario del token (FastAPI resuelve el token una sola vez
# aunque la ruta pida también UsuarioActual, y la conexión es la misma)
IdCuenta = Annotated[int, Depends(_cuenta)]
BggId = Annotated[int, Path(ge=1)]


@router.post("/sesion", response_model=Cuenta)
def iniciar_sesion(usuario: UsuarioActual, id_cuenta: IdCuenta, c: Conexion) -> Cuenta:
    """Llamar después de cada inicio de sesión (o de enlazar el invitado con Google)."""
    consultas_app.escribir(c, lambda: consultas_app.registrar_sesion(c, id_cuenta, usuario))
    return consultas_app.leer_cuenta(c, id_cuenta)


@router.get("", response_model=Cuenta)
def ver_cuenta(id_cuenta: IdCuenta, c: Conexion) -> Cuenta:
    return consultas_app.leer_cuenta(c, id_cuenta)


@router.delete("", status_code=204)
def borrar_cuenta(usuario: UsuarioActual, c: Conexion) -> Response:
    """Borra la cuenta y todos sus juegos. El usuario de Firebase lo borra el frontal."""
    id_cuenta = consultas_app.buscar_cuenta(c, usuario.uid)
    if id_cuenta is not None:
        n = consultas_app.escribir(c, lambda: consultas_app.borrar_cuenta(c, id_cuenta))
        log.info("Cuentas: cuenta %d borrada con sus %d juegos", id_cuenta, n)
    return Response(status_code=204)


@router.post("/fusionar", response_model=ResultadoFusion,
             responses={422: {"description": "Las cuentas no son un invitado y una de Google"}})
def fusionar(peticion: PeticionFusion, usuario: UsuarioActual, id_cuenta: IdCuenta,
             c: Conexion, request: Request) -> ResultadoFusion:
    """Pasa los juegos del invitado (token_invitado) a la cuenta de Google de la cabecera y
    borra la del invitado. Si un juego está en las dos, se queda el de la cuenta de Google.
    Repetirla no hace nada: la segunda vez el invitado ya no tiene cuenta."""
    if usuario.es_invitado:
        raise HTTPException(status_code=422, detail="La cabecera tiene que llevar el token de "
                                                    "la cuenta de Google, no el del invitado")
    invitado = comprobar_token(request, peticion.token_invitado)
    if not invitado.es_invitado:
        raise HTTPException(status_code=422, detail="token_invitado no es de un invitado")
    if invitado.uid == usuario.uid:
        raise HTTPException(status_code=422, detail="Es la misma cuenta: al enlazar el invitado "
                                                    "con Google no hace falta fusionar")
    movidos = ya_estaban = 0
    origen = consultas_app.buscar_cuenta(c, invitado.uid)
    if origen is not None:
        movidos, ya_estaban = consultas_app.escribir(
            c, lambda: consultas_app.fusionar(c, origen, id_cuenta))
        log.info("Cuentas: invitado %d fusionado en la cuenta %d (%d juegos movidos, %d ya "
                 "estaban)", origen, id_cuenta, movidos, ya_estaban)
    return ResultadoFusion(movidos=movidos, ya_estaban=ya_estaban,
                           cuenta=consultas_app.leer_cuenta(c, id_cuenta))


@router.get("/juegos", response_model=list[JuegoGuardado])
def listar_juegos(id_cuenta: IdCuenta, c: Conexion,
                  lista: Lista | None = None) -> list[JuegoGuardado]:
    """Los juegos guardados, los últimos cambiados primero. Sin `lista`, los de las tres."""
    return consultas_app.leer_juegos(c, id_cuenta, lista)


@router.put("/juegos/{bgg_id}", response_model=JuegoGuardado,
            responses={404: {"description": "El juego no está en el catálogo"}})
def guardar_juego(bgg_id: BggId, datos: GuardarJuego, id_cuenta: IdCuenta,
                  c: Conexion) -> JuegoGuardado:
    """Guarda el juego en la lista indicada; si estaba en otra, lo mueve. Sustituye la nota:
    sin `nota`, el juego queda sin nota."""
    if not consultas_app.existe_en_catalogo(c, bgg_id):
        raise HTTPException(status_code=404, detail=f"No hay ningún juego con bgg_id {bgg_id}")
    consultas_app.escribir(
        c, lambda: consultas_app.guardar_juego(c, id_cuenta, bgg_id, datos.lista, datos.nota))
    return consultas_app.leer_juegos(c, id_cuenta, bgg_id=bgg_id)[0]


@router.delete("/juegos/{bgg_id}", status_code=204)
def quitar_juego(bgg_id: BggId, id_cuenta: IdCuenta, c: Conexion) -> Response:
    """Quita el juego de la lista en la que esté. Si no estaba guardado, no pasa nada."""
    consultas_app.escribir(c, lambda: consultas_app.quitar_juego(c, id_cuenta, bgg_id))
    return Response(status_code=204)