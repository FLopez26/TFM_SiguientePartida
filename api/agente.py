"""Chat con Gemini: grafo de LangGraph que traduce la conversación en filtros del recomendador y
redacta la respuesta solo con juegos que devuelve la API.

    START → interpretar ─┬─ pide juegos o pregunta por alguno ─→ recuperar ─→ responder → END
                         └─ saludo o fuera de tema ────────────────────────→ responder

    interpretar  Gemini con salida estructurada: filtros (jugadores, duración, complejidad y
                 etiquetas por su id), juegos que le gustan, que no y por los que pregunta, y si
                 es para otra persona. Lee toda la conversación, así que entiende "¿y alguno más
                 corto?" sin un paso aparte para reformular la pregunta
    recuperar    sin LLM: busca esos juegos por nombre en gold.juego, lee la colección guardada
                 (salvo si es para otra persona), llama a calcular(), lo mismo que POST
                 /recomendaciones, y lee las fichas
    responder    Gemini con el system prompt, el contexto y los últimos mensajes

Sin memoria en la API: el frontal manda los últimos mensajes en cada turno. A Gemini solo le
llega el texto del chat y datos del catálogo, nunca el token, el uid ni el nombre de la persona.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.exceptions import (ModelAuthenticationError, ModelPermissionDeniedError,
                                       ModelRateLimitError)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from sqlalchemy.engine import Connection

from api import consultas, consultas_app, recomendaciones
from api.esquemas import (EntendidoChat, Etiqueta, FichaJuego, Filtros, JuegoResumen,
                          MensajeChat, RespuestaChat, RespuestaRecomendaciones)
from api.modelo import Recomendador

log = logging.getLogger("api.chat")

K_CHAT = 5                 # recomendaciones por turno, en el orden del modelo
MAX_HISTORIAL = 10         # mensajes que se mandan a Gemini en cada llamada
MAX_NOMBRADOS = 10         # juegos que se buscan por nombre en cada lista
MAX_CONSULTADOS = 3        # fichas de juegos por los que pregunta
MAX_MECANICAS = 6          # mecánicas de cada ficha en el contexto


# --------------------------------------------------------------------------- errores de Gemini
class ErrorGemini(Exception):
    """Fallo al llamar a Gemini; la ruta lo traduce a un código HTTP."""


class CuotaAgotada(ErrorGemini):
    """429: se ha superado el límite de peticiones de la cuota gratuita."""


class ClaveNoValida(ErrorGemini):
    """La clave de Gemini no vale o no tiene permiso para el modelo."""


class GeminiNoResponde(ErrorGemini):
    """Cualquier otro fallo: red, tiempo agotado, respuesta vacía o que no sigue el esquema."""


def _invocar(llm: Any, mensajes: list[BaseMessage], tiempos: list[float]) -> Any:
    """Llama a Gemini y convierte sus errores en los de arriba. Solo envuelve la llamada al LLM:
    los fallos de MySQL siguen su camino (503 desde main.py)."""
    inicio = time.perf_counter()
    try:
        respuesta = llm.invoke(mensajes)
    except ModelRateLimitError as e:
        raise CuotaAgotada(str(e)) from e
    except (ModelAuthenticationError, ModelPermissionDeniedError) as e:
        raise ClaveNoValida(str(e)) from e
    except Exception as e:
        # Solo el tipo: el mensaje de un fallo al leer la salida estructurada repite lo que
        # escribió Gemini, que puede incluir trozos del chat
        raise GeminiNoResponde(e.__class__.__name__) from e
    finally:
        tiempos.append(time.perf_counter() - inicio)
    if respuesta is None:
        raise GeminiNoResponde("respuesta vacía")
    return respuesta


# --------------------------------------------------------------------------- interpretar
class Interpretacion(BaseModel):
    """Salida estructurada de Gemini. Todos los campos son obligatorios (null o lista vacía si no
    aplican) y sin rangos: los valores se revisan en filtros_de(), así un dato raro se descarta
    en vez de hacer fallar el turno."""
    quiere_recomendaciones: bool = Field(
        description="true si pide juegos u otras opciones, o cuenta gustos o restricciones; "
                    "false en saludos, agradecimientos y temas que no son juegos de mesa")
    para_otra_persona: bool = Field(
        description="true si los juegos son para regalar o para otra persona")
    jugadores: int | None = Field(description="Cuántas personas van a jugar")
    duracion_min: int | None = Field(description="Duración mínima en minutos")
    duracion_max: int | None = Field(description="Duración máxima en minutos")
    peso_min: float | None = Field(description="Complejidad mínima, de 1 a 5")
    peso_max: float | None = Field(description="Complejidad máxima, de 1 a 5")
    etiquetas: list[int] = Field(description="ids de la lista de etiquetas que encajan")
    le_gustan: list[str] = Field(description="Juegos que tiene o que le gustan")
    no_le_gustan: list[str] = Field(description="Juegos que no le gustan o no quiere")
    pregunta_por: list[str] = Field(description="Juegos concretos sobre los que pregunta")


PROMPT_INTERPRETAR = """Analizas una conversación entre una persona y el asistente de una web que \
recomienda juegos de mesa. Devuelve lo que la persona busca AHORA, según toda la conversación:
- Acumula lo que ha ido diciendo («somos 4» y después «que dure poco») salvo que lo cambie o lo retire.
- «¿Y alguno más corto / más sencillo / para menos gente?» ajusta ese filtro tomando como \
referencia los juegos que ya recomendó el asistente.
- No pongas restricciones que no haya dicho.

Campos:
- quiere_recomendaciones: true si pide juegos u otras opciones, o cuenta gustos o restricciones; \
false en saludos, agradecimientos, despedidas y temas que no son juegos de mesa.
- para_otra_persona: true si es para regalar o para alguien que no es ella.
- jugadores: cuántos van a jugar (en pareja = 2). null si no lo dice.
- duracion_min / duracion_max: minutos. «Corto» o «rápido» = máximo 30; «menos de una hora» = \
máximo 60; «largo» = mínimo 90.
- peso_min / peso_max: complejidad de 1 a 5. «Sencillo», «familiar» o «para no jugones» = máximo \
2; «intermedio» = entre 2 y 3,5; «complejo» o «para jugones» = mínimo 3,5.
- etiquetas: ids de la LISTA DE ETIQUETAS que encajan con la temática, la ambientación, las \
mecánicas o el tipo de juego que pide («de piratas», «cooperativo», «para niños»). Como mucho 3 y \
solo ids de la lista; si nada encaja, lista vacía. Varias del mismo tipo se combinan con O y de \
tipos distintos con Y, así que cada etiqueta de más deja menos juegos.
- le_gustan: juegos que dice tener o que le gustan. no_le_gustan: los que no le gustan o no \
quiere. pregunta_por: juegos concretos por los que pregunta, también si dice «el segundo» de los \
que recomendó el asistente (escribe su nombre).
- En esas tres listas escribe el nombre con el que aparece en BoardGameGeek, normalmente el \
título original («Aventureros al Tren» → «Ticket to Ride»).

LISTA DE ETIQUETAS (id=nombre, por tipo; los nombres están en inglés):
"""


def _lista_etiquetas(etiquetas: list[Etiqueta]) -> str:
    por_tipo: dict[str, list[str]] = {}
    for e in etiquetas:
        por_tipo.setdefault(e.tipo, []).append(f"{e.id_etiqueta}={e.nombre}")
    return "\n".join(f"{tipo}: {'; '.join(nombres)}" for tipo, nombres in por_tipo.items())


def _historial(mensajes: list[BaseMessage]) -> list[BaseMessage]:
    """Los últimos MAX_HISTORIAL mensajes, empezando por uno del usuario (Gemini lo pide)."""
    recientes = list(mensajes[-MAX_HISTORIAL:])
    while recientes and not isinstance(recientes[0], HumanMessage):
        recientes.pop(0)
    return recientes


def _numero(valor: float | None, minimo: float, maximo: float) -> float | None:
    return valor if valor is not None and minimo <= valor <= maximo else None


def filtros_de(i: Interpretacion, validas: set[int]) -> Filtros:
    """Filtros del recomendador a partir de lo que entiende Gemini: descarta valores fuera de
    rango y etiquetas que no están en el catálogo, y da la vuelta a un rango al revés."""
    entero = lambda v, a, b: None if (n := _numero(v, a, b)) is None else int(n)  # noqa: E731
    d: dict[str, Any] = {
        "jugadores": entero(i.jugadores, 1, 100),
        "duracion_min": entero(i.duracion_min, 1, 100_000),
        "duracion_max": entero(i.duracion_max, 1, 100_000),
        "peso_min": _numero(i.peso_min, 1, 5),
        "peso_max": _numero(i.peso_max, 1, 5),
        "etiquetas": sorted({e for e in i.etiquetas if e in validas})[:50],
    }
    for campo in ("duracion", "peso"):
        minimo, maximo = d[f"{campo}_min"], d[f"{campo}_max"]
        if minimo is not None and maximo is not None and minimo > maximo:
            d[f"{campo}_min"], d[f"{campo}_max"] = maximo, minimo
    return Filtros(**d)


# --------------------------------------------------------------------------- grafo
@dataclass
class Contexto:
    """Lo que necesita el grafo y no va en el estado (LangGraph lo pasa a cada nodo)."""
    llm: Any                          # ChatGoogleGenerativeAI, o uno falso en los tests
    conexion: Connection
    recomendador: Recomendador
    id_cuenta: int | None             # None si la cuenta aún no existe en app.cuenta
    tiempos_llm: list[float] = field(default_factory=list)


class EstadoChat(MessagesState):
    """'messages' lo aporta MessagesState: la conversación y, al final, la respuesta."""
    interpretacion: Interpretacion
    etiquetas_catalogo: dict[int, Etiqueta]
    entendido: EntendidoChat
    resultado: RespuestaRecomendaciones | None
    contexto: str


def nodo_interpretar(estado: EstadoChat, runtime: Runtime[Contexto]) -> dict:
    ctx = runtime.context
    etiquetas = {e.id_etiqueta: e for e in consultas.listar_etiquetas(ctx.conexion)
                 if e.n_juegos > 0}
    sistema = SystemMessage(PROMPT_INTERPRETAR + _lista_etiquetas(list(etiquetas.values())))
    interpretacion = _invocar(ctx.llm.with_structured_output(Interpretacion),
                              [sistema, *_historial(estado["messages"])], ctx.tiempos_llm)
    return {"interpretacion": interpretacion, "etiquetas_catalogo": etiquetas}


def hay_que_consultar(estado: EstadoChat) -> str:
    i = estado["interpretacion"]
    return "recuperar" if i.quiere_recomendaciones or i.pregunta_por else "responder"


def _buscar(c: Connection, nombres: list[str]) -> tuple[list[JuegoResumen], list[str]]:
    """Busca cada nombre en el catálogo y se queda con el primero (mismo orden que GET /juegos:
    nombre exacto, después los que empiezan así y después el resto)."""
    encontrados: dict[int, JuegoResumen] = {}
    no_encontrados: list[str] = []
    for nombre in dict.fromkeys(n.strip() for n in nombres[:MAX_NOMBRADOS]):
        if len(nombre) < 2:
            continue
        resultado = consultas.buscar_juegos(c, nombre[:100], 1)
        if resultado:
            encontrados.setdefault(resultado[0].bgg_id, resultado[0])
        else:
            no_encontrados.append(nombre)
    return list(encontrados.values()), no_encontrados


def nodo_recuperar(estado: EstadoChat, runtime: Runtime[Contexto]) -> dict:
    ctx, c, i = runtime.context, runtime.context.conexion, estado["interpretacion"]
    etiquetas = estado["etiquetas_catalogo"]
    filtros = filtros_de(i, set(etiquetas))
    le_gustan, faltan_1 = _buscar(c, i.le_gustan)
    no_le_gustan, faltan_2 = _buscar(c, i.no_le_gustan)
    consultados, faltan_3 = _buscar(c, i.pregunta_por[:MAX_CONSULTADOS])

    # Colección guardada (2b): cuenta para el perfil salvo que sea para otra persona, como en
    # el modo regalo. Sus deseos y descartes no se recomiendan, como en "para mí"
    coleccion: dict[int, float | None] = {}
    fuera: set[int] = set()
    if ctx.id_cuenta is not None and not i.para_otra_persona:
        coleccion, fuera = consultas_app.leer_perfil(c, ctx.id_cuenta)
    con_coleccion = bool(coleccion)
    del_chat = {j.bgg_id for j in le_gustan} - coleccion.keys()
    for j in le_gustan:
        coleccion.setdefault(j.bgg_id, None)
    fuera |= {j.bgg_id for j in no_le_gustan}

    resultado = None
    if i.quiere_recomendaciones:
        resultado = recomendaciones.calcular(ctx.recomendador, c, coleccion, fuera, filtros,
                                             K_CHAT, modo="chat")
    ids = [r.juego.bgg_id for r in resultado.recomendaciones] if resultado else []
    fichas = {b: consultas.ficha_juego(c, b) for b in dict.fromkeys(
        ids + [j.bgg_id for j in consultados])}

    entendido = EntendidoChat(
        filtros=filtros, etiquetas=[etiquetas[e] for e in filtros.etiquetas],
        le_gustan=le_gustan, no_le_gustan=no_le_gustan,
        no_encontrados=list(dict.fromkeys(faltan_1 + faltan_2 + faltan_3)),
        con_coleccion=con_coleccion)
    return {"entendido": entendido, "resultado": resultado,
            "contexto": construir_contexto(entendido, resultado, fichas, consultados,
                                           i.para_otra_persona, del_chat)}


SYSTEM_PROMPT = """Eres el asistente de Siguiente Partida, una web que recomienda juegos de mesa \
con datos de BoardGameGeek. Ayudas a la persona a encontrar juegos que encajen con lo que busca.

## Lo que sabes
Solo sabes lo que aparece en el CONTEXTO de este turno y lo que ya habéis hablado. El contexto \
lo prepara la web: los filtros que ha entendido, los juegos que la persona ha nombrado y las \
recomendaciones del recomendador, con sus fichas. No uses otros conocimientos sobre juegos de mesa.

## Reglas
1. Menciona y recomienda solo juegos del contexto o que ya hayas recomendado antes en esta \
conversación, con su nombre tal cual. Nunca inventes juegos ni datos (jugadores, duración, \
reglas, premios, precios).
2. Presenta todas las recomendaciones del contexto, en el mismo orden y sin añadir otras: el \
orden lo decide el recomendador.
3. Justifica cada una con lo que ha pedido (jugadores, duración, complejidad, temática) y con \
lo que la distingue según su ficha. Si el contexto la relaciona con un juego suyo, puedes \
decirlo; si varias se relacionan con el mismo, dilo una sola vez al presentarlas, no en cada línea.
4. Si el contexto dice que algún juego no está en el catálogo, díselo ("No lo tengo en el catálogo").
5. Si ningún juego cumple lo pedido, dilo y propón relajar alguna condición.
6. Si sabes poco de lo que busca, recomienda igualmente y termina con UNA pregunta corta sobre lo \
que aún no sepas: cuántos jugarán, cuánto tiempo, qué complejidad o qué temática.
7. Si la conversación no trata de juegos de mesa, di amablemente que solo puedes ayudar con \
juegos de mesa.
8. No tienes precios ni tiendas: si te preguntan por el precio, dilo.
9. La complejidad va de 1 (muy sencillo) a 5 (muy complejo); la duración es la que indica la caja.

## Formato
- En español, con un tono cercano y conciso. No empieces con un saludo salvo que la persona \
salude.
- Texto plano, sin Markdown (ni asteriscos ni almohadillas): la web lo muestra tal cual y enseña \
debajo las fichas de los juegos recomendados.
- Una línea por recomendación: "1. Nombre (año): por qué encaja", en una o dos frases.
- Las etiquetas del catálogo están en inglés: tradúcelas.
"""


def _titulo(j: JuegoResumen | FichaJuego) -> str:
    return f"{j.nombre} ({j.anio})" if j.anio else str(j.nombre)


def describir_ficha(f: FichaJuego) -> str:
    """Una ficha en pocas líneas, solo con datos de Gold."""
    datos = []
    if f.min_jugadores and f.max_jugadores:
        caja = (str(f.min_jugadores) if f.min_jugadores == f.max_jugadores
                else f"{f.min_jugadores}-{f.max_jugadores}")
        votos = (f" (la comunidad recomienda {', '.join(map(str, f.jugadores_recomendados))})"
                 if f.jugadores_recomendados else "")
        datos.append(f"jugadores {caja}{votos}")
    if f.duracion_fabricante:
        datos.append(f"{f.duracion_fabricante} min")
    if f.peso:
        datos.append(f"complejidad {f.peso:.1f}/5")
    if f.edad_min_fabricante:
        datos.append(f"desde {f.edad_min_fabricante} años")
    e = f.etiquetas
    etiquetas = [f"{nombre}: {', '.join(lista)}" for nombre, lista in [
        ("categorías", e.categoria), ("mecánicas", e.mecanica[:MAX_MECANICAS]),
        ("temas", e.tema)] if lista]
    return "\n".join([" · ".join(datos) or "sin datos de jugadores, duración ni complejidad",
                      *etiquetas])


def _texto_filtros(entendido: EntendidoChat) -> str:
    f, partes = entendido.filtros, []
    if f.jugadores:
        partes.append(f"{f.jugadores} jugadores")
    for campo, unidad, nombre in [("duracion", " min", "duración"), ("peso", "", "complejidad")]:
        minimo, maximo = getattr(f, f"{campo}_min"), getattr(f, f"{campo}_max")
        if minimo is not None or maximo is not None:
            partes.append(f"{nombre} " + (f"de {minimo} a {maximo}{unidad}" if minimo and maximo
                          else f"desde {minimo}{unidad}" if minimo else f"hasta {maximo}{unidad}"))
    if entendido.etiquetas:
        partes.append("etiquetas " + ", ".join(f"{e.nombre} ({e.tipo})"
                                               for e in entendido.etiquetas))
    return "; ".join(partes) or "ninguno"


def construir_contexto(entendido: EntendidoChat, resultado: RespuestaRecomendaciones | None,
                       fichas: dict[int, FichaJuego | None], consultados: list[JuegoResumen],
                       para_otra_persona: bool, del_chat: set[int]) -> str:
    """Texto que recibe Gemini. del_chat: juegos del perfil que ha nombrado en la conversación
    (el resto del perfil sale de su colección guardada)."""
    lineas = [f"FILTROS APLICADOS: {_texto_filtros(entendido)}"]
    if entendido.le_gustan:
        lineas.append("JUEGOS QUE LE GUSTAN (en el catálogo): "
                      + ", ".join(map(_titulo, entendido.le_gustan)))
    if entendido.no_le_gustan:
        lineas.append("JUEGOS QUE NO QUIERE (no se recomiendan): "
                      + ", ".join(map(_titulo, entendido.no_le_gustan)))
    if entendido.no_encontrados:
        lineas.append("NO ESTÁN EN EL CATÁLOGO: " + ", ".join(entendido.no_encontrados))
    if para_otra_persona:
        lineas.append("ES PARA OTRA PERSONA: no se ha usado la colección guardada.")
    elif entendido.con_coleccion:
        lineas.append("COLECCIÓN GUARDADA: se ha tenido en cuenta.")

    if resultado is not None:
        if not resultado.recomendaciones:
            lineas.append("\nRECOMENDACIONES: ninguna. Ningún juego del catálogo cumple todos "
                          "los filtros a la vez.")
        else:
            lineas.append("\nRECOMENDACIONES (preséntalas todas, en este orden):")
        for r in resultado.recomendaciones:
            ficha = fichas.get(r.juego.bgg_id)
            if r.porque is None:
                porque = "sale por popularidad entre los que cumplen los filtros"
            else:
                origen = ("que ha dicho que le gusta" if r.porque.bgg_id in del_chat
                          else "que está en su colección")
                porque = f"relacionado con {_titulo(r.porque)}, {origen}"
            lineas.append(f"[{r.posicion}] {_titulo(r.juego)}\n"
                          + (describir_ficha(ficha) + "\n" if ficha else "")
                          + f"Por qué: {porque}")
    if consultados:
        lineas.append("\nJUEGOS POR LOS QUE PREGUNTA:")
        for j in consultados:
            ficha = fichas.get(j.bgg_id)
            lineas.append(f"- {_titulo(j)}\n" + (describir_ficha(ficha) if ficha else "sin ficha"))
    return "\n".join(lineas)


SIN_CONSULTA = ("En este turno no se ha consultado el catálogo: la conversación no pide juegos "
                "ni pregunta por ninguno.")


def nodo_responder(estado: EstadoChat, runtime: Runtime[Contexto]) -> dict:
    ctx = runtime.context
    sistema = SystemMessage(f"{SYSTEM_PROMPT}\n<contexto>\n"
                            f"{estado.get('contexto') or SIN_CONSULTA}\n</contexto>")
    respuesta = _invocar(ctx.llm, [sistema, *_historial(estado["messages"])], ctx.tiempos_llm)
    texto = (respuesta.text or "").strip()
    if not texto:
        raise GeminiNoResponde("respuesta sin texto")
    return {"messages": [AIMessage(content=texto)]}


def construir_grafo():
    grafo = StateGraph(EstadoChat, context_schema=Contexto)
    grafo.add_node("interpretar", nodo_interpretar)
    grafo.add_node("recuperar", nodo_recuperar)
    grafo.add_node("responder", nodo_responder)
    grafo.add_edge(START, "interpretar")
    grafo.add_conditional_edges("interpretar", hay_que_consultar, ["recuperar", "responder"])
    grafo.add_edge("recuperar", "responder")
    grafo.add_edge("responder", END)
    return grafo.compile()      # sin checkpointer: la conversación la guarda el frontal


GRAFO = construir_grafo()


def conversar(llm: Any, c: Connection, rec: Recomendador, id_cuenta: int | None,
              mensajes: list[MensajeChat]) -> RespuestaChat:
    """Un turno del chat: la conversación entra, la respuesta sale. Lanza ErrorGemini."""
    inicio = time.perf_counter()
    ctx = Contexto(llm=llm, conexion=c, recomendador=rec, id_cuenta=id_cuenta)
    historial = [HumanMessage(m.texto) if m.rol == "usuario" else AIMessage(m.texto)
                 for m in mensajes]
    estado = GRAFO.invoke({"messages": historial}, context=ctx)

    entendido = estado.get("entendido") or EntendidoChat(
        filtros=Filtros(), etiquetas=[], le_gustan=[], no_le_gustan=[], no_encontrados=[],
        con_coleccion=False)
    resultado = estado.get("resultado")
    recs = resultado.recomendaciones if resultado else []
    # Sin el texto del chat: alguien puede escribir datos personales
    log.info("Chat: %d mensajes, %s, filtros %s, %d juegos nombrados (%d no están), "
             "colección %s, %d recomendados en %.0f ms (Gemini: %d llamadas, %.0f ms)",
             len(mensajes), "consulta" if "contexto" in estado else "sin consulta",
             entendido.filtros.model_dump(exclude_defaults=True) or "ninguno",
             len(entendido.le_gustan) + len(entendido.no_le_gustan)
             + len(entendido.no_encontrados), len(entendido.no_encontrados),
             "sí" if entendido.con_coleccion else "no", len(recs),
             (time.perf_counter() - inicio) * 1000, len(ctx.tiempos_llm),
             sum(ctx.tiempos_llm) * 1000)
    return RespuestaChat(texto=estado["messages"][-1].text, recomendaciones=recs,
                         entendido=entendido, modelo_version=rec.version)