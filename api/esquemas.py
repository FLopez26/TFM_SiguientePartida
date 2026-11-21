"""Formato de las respuestas de la API. Es lo que aparece en /docs y lo que recibirá el frontal.

Los campos se llaman igual que las columnas de Gold, para poder seguir un dato desde la base de
datos hasta la pantalla sin traducciones. Las notas y el puesto de BGG tienen fuga de
información para los modelos (COLUMNAS_CON_FUGA), pero mostrarlos en la ficha no es un problema:
ningún modelo los usa.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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