"""Recomendaciones: POST /recomendaciones.

Sirve para los usos del plan; solo cambia lo que manda el frontal:
    para mí   la colección de la persona, y sus descartes en `excluir`. Con cuenta, mejor
              POST /cuenta/recomendaciones (api/cuentas.py), que lee lo guardado y llama a
              calcular() con lo mismo
    regalo    colección vacía y los filtros del cuestionario: salen los más populares que los
              cumplen. También se pueden mandar juegos que le gustan a quien recibe el regalo;
              la API no guarda nada
    chat      el LLM traduce la conversación a estos mismos filtros

Los filtros se aplican antes del modelo: los juegos que no los cumplen se le pasan con
excluir=True (no cuentan para el perfil y no se recomiendan), así que el top-k sale ya entre los
que los cumplen y no hay que pedir más juegos de la cuenta y recortar después.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.engine import Connection

from api import consultas
from api.bd import Conexion
from api.esquemas import (Filtros, PeticionRecomendacion, Recomendacion,
                          RespuestaRecomendaciones)
from api.modelo import Recomendador

log = logging.getLogger("api.recomendaciones")

router = APIRouter(tags=["recomendaciones"])


def entrada_modelo(perfil: dict[int, float | None], no_recomendables: set[int]) -> pd.DataFrame:
    """Filas para el modelo (ver ml/modelo_mlflow.py): los juegos del perfil con su nota, y los
    que no se pueden recomendar con excluir=True."""
    ids_perfil, ids_fuera = list(perfil), sorted(no_recomendables)
    notas = [np.nan if perfil[b] is None else perfil[b] for b in ids_perfil]
    return pd.DataFrame({
        "bgg_id": np.array(ids_perfil + ids_fuera, dtype=np.int64),
        "nota": np.array(notas + [np.nan] * len(ids_fuera), dtype=np.float64),
        "excluir": np.array([False] * len(ids_perfil) + [True] * len(ids_fuera), dtype=bool),
    })


def recomendador_cargado(request: Request) -> Recomendador:
    rec = request.app.state.recomendador
    if rec is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado: mira /salud")
    return rec


def calcular(rec: Recomendador, c: Connection, coleccion: dict[int, float | None],
             excluir: set[int], filtros: Filtros, k: int,
             modo: str = "petición") -> RespuestaRecomendaciones:
    """Recomendaciones para una colección (bgg_id -> nota o None). Los juegos de `excluir` no
    se recomiendan y no cuentan para el perfil, aunque estén también en la colección."""
    inicio = time.perf_counter()
    ignorados = sorted(b for b in coleccion if b not in rec.juegos)
    perfil = {b: n for b, n in coleccion.items() if b in rec.juegos and b not in excluir}

    if filtros.hay_filtros():
        try:
            permitidos = consultas.candidatos(c, filtros) & rec.juegos
        except consultas.EtiquetasDesconocidas as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
    else:
        permitidos = rec.juegos
    candidatos = permitidos.difference(coleccion, excluir)

    recomendaciones: list[Recomendacion] = []
    if candidatos:
        # Los juegos del perfil ya los evita el modelo; aquí van los que no cumplen los
        # filtros (sin los del perfil, que deben seguir contando) y los excluidos
        no_recomendables = (rec.juegos - permitidos).difference(perfil) | (excluir & rec.juegos)
        salida = rec.recomendar(entrada_modelo(perfil, no_recomendables),
                                k=min(k, len(candidatos)))
        porques = salida.loc[salida["porque_bgg_id"] >= 0, "porque_bgg_id"]
        juegos = consultas.resumenes(c, [*salida["bgg_id"], *porques])
        for f in salida.itertuples():
            juego = juegos.get(int(f.bgg_id))
            if juego is None:   # el modelo es de un Gold anterior y el juego ya no está
                log.warning("Recomendación: el juego %d no está en gold.juego", f.bgg_id)
                continue
            recomendaciones.append(Recomendacion(
                posicion=len(recomendaciones) + 1, juego=juego,
                porque=juegos.get(int(f.porque_bgg_id)), puntuacion=float(f.puntuacion)))

    log.info("Recomendación (%s): perfil de %d juegos (%d ignorados, %d excluidos), filtros %s, "
             "%d candidatos, %d recomendados en %.0f ms", modo, len(perfil), len(ignorados),
             len(excluir), filtros.model_dump(exclude_defaults=True) or "ninguno",
             len(candidatos), len(recomendaciones), (time.perf_counter() - inicio) * 1000)
    return RespuestaRecomendaciones(recomendaciones=recomendaciones, candidatos=len(candidatos),
                                    ignorados=ignorados, modelo_version=rec.version)


@router.post("/recomendaciones", response_model=RespuestaRecomendaciones,
             responses={503: {"description": "El modelo no está cargado"}})
def recomendar(peticion: PeticionRecomendacion, c: Conexion,
               request: Request) -> RespuestaRecomendaciones:
    rec = recomendador_cargado(request)
    coleccion = {j.bgg_id: j.nota for j in peticion.coleccion}   # repetido: vale la última nota
    return calcular(rec, c, coleccion, set(peticion.excluir), peticion.filtros, peticion.k)