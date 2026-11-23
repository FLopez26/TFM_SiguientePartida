"""Chat con Gemini: POST /chat. El grafo de LangGraph está en api/agente.py.

La API no guarda la conversación: el frontal la manda en cada turno. 
Pide el token de Firebase, que también tienen los invitados: así nadie gasta la cuota
de Gemini sin entrar en la web, y si la cuenta tiene colección guardada se usa como perfil, salvo
que la conversación diga que es para otra persona.

Errores de Gemini:
    429  se ha superado la cuota gratuita (el frontal avisa y deja reintentar)
    502  Gemini no ha respondido bien (red, tiempo agotado, respuesta vacía)
    503  falta GEMINI_API_KEY o la clave no vale
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from langchain_google_genai import ChatGoogleGenerativeAI

from api import agente, consultas_app, recomendaciones
from api.auth import UsuarioActual
from api.bd import Conexion
from api.config import Ajustes
from api.esquemas import PeticionChat, RespuestaChat

log = logging.getLogger("api.chat")

router = APIRouter(tags=["chat"])


def preparar_llm(ajustes: Ajustes) -> ChatGoogleGenerativeAI | None:
    """Cliente de Gemini. Sin clave la API arranca igual y POST /chat responde 503. Crear el
    cliente no llama a Google: una clave mal copiada se nota en el primer mensaje."""
    if not ajustes.gemini_clave:
        log.error("Chat: falta GEMINI_API_KEY en .env; POST /chat dará 503")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=ajustes.gemini_modelo, google_api_key=ajustes.gemini_clave,
            max_retries=2,      # 1 reintento: con la cuota agotada
            timeout=30)
    except Exception as e:
        log.error("Chat: no se puede crear el cliente de Gemini (%s)", e.__class__.__name__)
        return None
    log.info("Chat: Gemini con el modelo %s", ajustes.gemini_modelo)
    return llm


@router.post("/chat", response_model=RespuestaChat, responses={
    401: {"description": "Falta el token o no es válido"},
    429: {"description": "Se ha superado la cuota gratuita de Gemini"},
    502: {"description": "Gemini no ha respondido bien"},
    503: {"description": "Falta la clave de Gemini, el modelo o las cuentas"}})
def conversar(peticion: PeticionChat, usuario: UsuarioActual, c: Conexion,
              request: Request) -> RespuestaChat:
    """Un turno del chat. Devuelve el texto de la respuesta, las recomendaciones que menciona
    (en el mismo orden, para pintar sus fichas) y los filtros que se han entendido."""
    llm = request.app.state.llm
    if llm is None:
        raise HTTPException(status_code=503, detail="El chat no está configurado: falta "
                                                    "GEMINI_API_KEY (mira logs/api.log)")
    rec = recomendaciones.recomendador_cargado(request)
    id_cuenta = consultas_app.buscar_cuenta(c, usuario.uid)    # sin crearla: el chat no escribe
    try:
        return agente.conversar(llm, c, rec, id_cuenta, peticion.mensajes)
    except agente.CuotaAgotada:
        log.warning("Chat: cuota de Gemini superada")
        raise HTTPException(status_code=429, detail="Se ha superado el límite de mensajes de la "
                                                    "cuota gratuita de Gemini. Espera un minuto "
                                                    "y vuelve a probar.") from None
    except agente.ClaveNoValida as e:
        log.error("Chat: Gemini rechaza la clave: %s", e)
        raise HTTPException(status_code=503, detail="La clave de Gemini no es válida: mira "
                                                    "logs/api.log") from None
    except agente.GeminiNoResponde as e:
        log.error("Chat: Gemini no ha respondido bien: %s", e)
        raise HTTPException(status_code=502, detail="Gemini no ha podido responder. Prueba "
                                                    "otra vez.") from None