"""Formato de las respuestas de la API. Es lo que aparece en /docs y lo que recibirá el frontal.

Los campos se llaman igual que las columnas de Gold, para poder seguir un dato desde la base de
datos hasta la pantalla sin traducciones. Las notas y el puesto de BGG tienen fuga de
información para los modelos (COLUMNAS_CON_FUGA), pero mostrarlos en la ficha no es un problema:
ningún modelo los usa.
"""
from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TipoEtiqueta = Literal["categoria", "subcategoria", "mecanica", "tema"]


class JuegoResumen(BaseModel):
    """Lo mínimo para enseñar un juego en una lista (búsqueda, recomendaciones, colección)."""
    bgg_id: int
    nombre: str | None
    anio: int | None = Field(description="Año de publicación; negativo = a. C.")
    imagen_url: str | None


class EtiquetasJuego(BaseModel):
    categoria: list[str] = Field(default_factory=list)
    subcategoria: list[str] = Field(default_factory=list)
    mecanica: list[str] = Field(default_factory=list)
    tema: list[str] = Field(default_factory=list)


class FichaJuego(JuegoResumen):
    url_bgg: str = Field(description="Página del juego en BoardGameGeek")
    peso: float | None = Field(description="Complejidad de 1 a 5 votada por la comunidad de BGG")
    min_jugadores: int | None
    max_jugadores: int | None
    jugadores_recomendados: list[int] = Field(
        description="Números de jugadores que recomienda la comunidad de BGG (vacío si no hay votos)")
    mejor_n_jugadores: int | None = Field(description="Vacío en el 91 % de los juegos")
    duracion_fabricante: int | None = Field(description="Minutos")
    duracion_min_comunidad: int | None = Field(description="Minutos")
    duracion_max_comunidad: int | None = Field(description="Minutos")
    edad_min_fabricante: int | None
    edad_rec_comunidad: float | None
    nota_media_bgg: float | None
    n_valoraciones_bgg: int | None
    rank_general: int | None = Field(description="Puesto en el ranking de BGG de los datos de Kaggle")
    familia: str | None
    n_expansiones: int | None
    etiquetas: EtiquetasJuego


class Etiqueta(BaseModel):
    id_etiqueta: int
    tipo: TipoEtiqueta
    nombre: str
    n_juegos: int = Field(description="Juegos del catálogo con esta etiqueta")



# --------------------------------------------------------------------------- recomendaciones
# En las peticiones, extra="forbid": un campo mal escrito desde el frontal ("jugadore") da un
# error 422 en vez de ignorarse en silencio.

class JuegoColeccion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bgg_id: int = Field(ge=1)
    nota: float | None = Field(default=None, ge=1, le=10, description="Opcional")


class Filtros(BaseModel):
    """Filtros del cuestionario del modo regalo (y, en la fase 5, del chat). Todos opcionales.
    Con un filtro puesto, los juegos que no tienen ese dato en Gold no pasan."""
    model_config = ConfigDict(extra="forbid")
    jugadores: int | None = Field(
        default=None, ge=1, le=100,
        description="Si la comunidad de BGG votó los números de jugadores del juego, tiene que "
                    "ser uno de los recomendados; si no, basta con que esté entre el mínimo y "
                    "el máximo de la caja")
    duracion_min: int | None = Field(default=None, ge=1, description="Minutos (duracion_fabricante)")
    duracion_max: int | None = Field(default=None, ge=1, description="Minutos (duracion_fabricante)")
    peso_min: float | None = Field(default=None, ge=1, le=5, description="Complejidad de 1 a 5")
    peso_max: float | None = Field(default=None, ge=1, le=5, description="Complejidad de 1 a 5")
    etiquetas: list[int] = Field(
        default_factory=list, max_length=50,
        description="id_etiqueta de /etiquetas. Dentro de un mismo tipo basta con una (Fantasy o "
                    "Science Fiction); de tipos distintos hacen falta todas (un tema y una mecánica)")

    @model_validator(mode="after")
    def _rangos(self) -> "Filtros":
        for campo in ("duracion", "peso"):
            minimo, maximo = getattr(self, f"{campo}_min"), getattr(self, f"{campo}_max")
            if minimo is not None and maximo is not None and minimo > maximo:
                raise ValueError(f"{campo}_min no puede ser mayor que {campo}_max")
        return self

    def hay_filtros(self) -> bool:
        return any(v not in (None, []) for v in self.model_dump().values())


class PeticionRecomendacion(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "coleccion": [{"bgg_id": 13, "nota": 9}, {"bgg_id": 822, "nota": 8},
                      {"bgg_id": 9209}, {"bgg_id": 30549}],
        "excluir": [],
        "filtros": {"jugadores": 2},
        "k": 5,
    }]})
    coleccion: list[JuegoColeccion] = Field(
        default_factory=list, max_length=5000,
        description="Juegos de la persona. Vacía en el modo regalo: se recomiendan los más "
                    "populares que cumplan los filtros")
    excluir: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=list, max_length=5000,
        description="Juegos que no se recomiendan y no cuentan para el perfil (p. ej. los "
                    "descartes). Si un juego está también en la colección, cuenta como excluido")
    filtros: Filtros = Field(default_factory=Filtros)
    k: int = Field(default=10, ge=1, le=50, description="Cuántos juegos recomendar")


class Recomendacion(BaseModel):
    posicion: int
    juego: JuegoResumen
    porque: JuegoResumen | None = Field(
        description="El juego de la colección que más pesa en esta recomendación (\"porque "
                    "tienes X\"); vacío si la recomendación sale de la popularidad")
    puntuacion: float = Field(description="Solo sirve para ordenar: no es una nota")


class RespuestaRecomendaciones(BaseModel):
    recomendaciones: list[Recomendacion]
    candidatos: int = Field(description="Juegos que cumplían los filtros y no estaban ni en la "
                                        "colección ni en excluir")
    ignorados: list[int] = Field(description="Juegos de la colección que el modelo no conoce (sin "
                                             "valoraciones en Gold): no cuentan para el perfil")
    modelo_version: str

# --------------------------------------------------------------------------- cuentas
# Los nombres siguen el esquema app (docker/mysql/init/02_app.sql).

Lista = Literal["coleccion", "deseo", "descarte"]


class JuegosPorLista(BaseModel):
    coleccion: int = 0
    deseo: int = 0
    descarte: int = 0


class Cuenta(BaseModel):
    nombre_visible: str | None = Field(description="El de Google; vacío en invitados")
    es_invitado: bool
    creada_en: datetime = Field(description="Hora del servidor de MySQL (UTC en Docker)")
    juegos: JuegosPorLista = Field(description="Cuántos juegos hay en cada lista")


class JuegoGuardado(BaseModel):
    bgg_id: int
    lista: Lista
    nota: float | None = Field(description="Solo en la colección")
    actualizado_en: datetime
    juego: JuegoResumen | None = Field(
        description="Vacío si el juego ya no está en el catálogo (Gold se ha regenerado)")


class GuardarJuego(BaseModel):
    """Cuerpo de PUT /cuenta/juegos/{bgg_id}: sustituye lo que hubiera guardado de ese juego.
    Un juego está en una sola lista: guardarlo en otra lo mueve."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"lista": "coleccion", "nota": 8.5}, {"lista": "deseo"}]})
    lista: Lista
    nota: float | None = Field(default=None, ge=1, le=10,
                               description="Opcional y solo en la colección; se guarda con un "
                                           "decimal (7,25 pasa a 7,3)")

    @field_validator("nota")
    @classmethod
    def _un_decimal(cls, nota: float | None) -> float | None:
        # Como DECIMAL(3,1) de MySQL (redondeo hacia arriba en el 5), pero igual en los tests
        if nota is None:
            return None
        return float(Decimal(str(nota)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

    @model_validator(mode="after")
    def _nota_solo_en_coleccion(self) -> "GuardarJuego":
        if self.nota is not None and self.lista != "coleccion":
            raise ValueError("la nota solo se guarda en los juegos de la colección")
        return self


class PeticionFusion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token_invitado: str = Field(min_length=1, max_length=4096,
                                description="ID token del invitado, pedido antes de iniciar "
                                            "sesión con Google")


class ResultadoFusion(BaseModel):
    movidos: int = Field(description="Juegos del invitado que pasan a la cuenta de Google")
    ya_estaban: int = Field(description="Juegos que la cuenta de Google ya tenía: se queda su "
                                        "dato y se descarta el del invitado")
    cuenta: Cuenta

class PeticionParaMi(BaseModel):
    """Cuerpo de POST /cuenta/recomendaciones. La colección (con sus notas) y los juegos que no
    se recomiendan (deseos y descartes) los pone la API con lo que hay guardado en la cuenta."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"filtros": {"jugadores": 2}, "k": 10}]})
    filtros: Filtros = Field(default_factory=Filtros)
    k: int = Field(default=10, ge=1, le=50, description="Cuántos juegos recomendar")



# --------------------------------------------------------------------------- chat
# El frontal guarda la conversación y la manda entera en cada turno: la API no la guarda.

class MensajeChat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rol: Literal["usuario", "asistente"]
    texto: str = Field(min_length=1, max_length=2000)


class PeticionChat(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"mensajes": [
        {"rol": "usuario", "texto": "Busco un juego para jugar en pareja que no dure más de "
                                    "una hora"}]}]})
    mensajes: list[MensajeChat] = Field(
        min_length=1, max_length=20,
        description="La conversación, de la más antigua a la última, que tiene que ser del "
                    "usuario. A Gemini solo le llegan los últimos 10 mensajes")

    @model_validator(mode="after")
    def _ultimo_del_usuario(self) -> "PeticionChat":
        if self.mensajes[-1].rol != "usuario":
            raise ValueError("el último mensaje tiene que ser del usuario")
        return self


class EntendidoChat(BaseModel):
    """Lo que la API ha sacado de la conversación, para enseñarlo ("He entendido: ...")."""
    filtros: Filtros = Field(description="Los que se han aplicado al recomendador")
    etiquetas: list[Etiqueta] = Field(description="Las de filtros.etiquetas, con su nombre")
    le_gustan: list[JuegoResumen] = Field(description="Juegos nombrados que cuentan para el perfil")
    no_le_gustan: list[JuegoResumen] = Field(description="Juegos nombrados que no se recomiendan")
    no_encontrados: list[str] = Field(description="Juegos nombrados que no están en el catálogo")
    con_coleccion: bool = Field(description="Si se ha usado la colección guardada de la cuenta")


class RespuestaChat(BaseModel):
    texto: str = Field(description="Respuesta del asistente, en texto plano")
    recomendaciones: list[Recomendacion] = Field(
        description="Las que presenta el texto, en el mismo orden; vacía si en este turno no se "
                    "piden juegos")
    entendido: EntendidoChat
    modelo_version: str