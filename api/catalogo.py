"""Rutas del catálogo: buscar juegos, ficha de un juego y lista de etiquetas.

    GET /juegos?q=catan          buscar por nombre (para añadir juegos a la colección)
    GET /juegos/{bgg_id}         ficha de un juego
    GET /etiquetas?tipo=tema     etiquetas para el cuestionario del modo regalo y para explorar
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from api import consultas
from api.bd import Conexion
from api.esquemas import Etiqueta, FichaJuego, JuegoResumen, TipoEtiqueta

router = APIRouter(tags=["catálogo"])


@router.get("/juegos", response_model=list[JuegoResumen])
def buscar_juegos(
    c: Conexion,
    q: Annotated[str, Query(min_length=2, max_length=100, description="Parte del nombre")],
    limite: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[JuegoResumen]:
    """Juegos cuyo nombre contiene `q` (sin distinguir mayúsculas ni tildes)."""
    return consultas.buscar_juegos(c, q, limite)


@router.get("/juegos/{bgg_id}", response_model=FichaJuego,
            responses={404: {"description": "El juego no está en el catálogo"}})
def ficha_juego(c: Conexion, bgg_id: Annotated[int, Path(ge=1)]) -> FichaJuego:
    ficha = consultas.ficha_juego(c, bgg_id)
    if ficha is None:
        raise HTTPException(status_code=404, detail=f"No hay ningún juego con bgg_id {bgg_id}")
    return ficha


@router.get("/etiquetas", response_model=list[Etiqueta])
def listar_etiquetas(c: Conexion, tipo: TipoEtiqueta | None = None) -> list[Etiqueta]:
    """Categorías, subcategorías, mecánicas y temas, con cuántos juegos tiene cada una."""
    return consultas.listar_etiquetas(c, tipo)