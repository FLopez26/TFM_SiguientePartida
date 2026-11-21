"""Modelo de producción en MLflow: recibe la colección de una persona y devuelve sus recomendaciones.

MLflow guarda este fichero junto al modelo y lo ejecuta al cargarlo.
Los vecinos van aparte, en el .npz que genera guardar() de la librería.

Entrada: un DataFrame con una fila por juego de la persona
    bgg_id   int     juego (id de BGG)
    nota     float   opcional: su nota, o vacío si no la ha puesto
    excluir  bool    opcional: True = no se usa para el perfil, solo se evita recomendarlo
                     (por ejemplo, sus descartes)
Parámetros: k = cuántos juegos recomendar (10 por defecto)
Salida: un DataFrame con posicion, bgg_id, puntuacion y porque_bgg_id (el juego suyo que más
    pesa en esa recomendación; -1 si no hay ninguno, por ejemplo con una colección vacía, en
    la que se recomiendan los más populares)
"""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel
from scipy import sparse

from siguiente_partida.recomendadores import ItemKNN, cargar_recomendador

NOTA_POR_DEFECTO = 7.0   # si la persona no ha puesto ninguna nota


class RecomendadorJuegos(PythonModel):
    def load_context(self, context):
        self.modelo, self.ids = cargar_recomendador(context.artifacts["modelo"])
        self.posicion = pd.Series(np.arange(len(self.ids)), index=self.ids)

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        k = int((params or {}).get("k", 10))
        entrada = pd.DataFrame(model_input).drop_duplicates("bgg_id", keep="last")
        excluir = (entrada["excluir"].fillna(False).astype(bool) if "excluir" in entrada
                   else pd.Series(False, index=entrada.index))
        nota = (entrada["nota"].astype(float) if "nota" in entrada
                else pd.Series(np.nan, index=entrada.index))
        nota = nota.fillna(nota.mean() if nota.notna().any() else NOTA_POR_DEFECTO)

        # Juegos sin notas en BGG no están en el modelo: no cuentan para el perfil
        conocido = entrada["bgg_id"].isin(self.posicion.index)
        en_perfil = conocido & ~excluir
        n = len(self.ids)
        cols = self.posicion[entrada.loc[en_perfil, "bgg_id"]].to_numpy()
        R = sparse.csr_matrix((nota[en_perfil].to_numpy(np.float32), (np.zeros(len(cols)), cols)),
                              shape=(1, n))
        evitar = self.posicion[entrada.loc[conocido, "bgg_id"]].to_numpy()
        E = sparse.csr_matrix((np.ones(len(evitar)), (np.zeros(len(evitar)), evitar)), shape=(1, n))

        top, punt = self.modelo.recomendar(R, k=min(k, n - len(evitar)), excluir=E)
        if isinstance(self.modelo, ItemKNN):
            porque = self.modelo.explicar(R, top[0])
        else:
            porque = np.full(top.shape[1], -1)
        return pd.DataFrame({
            "posicion": np.arange(1, top.shape[1] + 1, dtype=np.int64),
            "bgg_id": self.ids[top[0]].astype(np.int64),
            "puntuacion": punt[0].astype(np.float64),
            "porque_bgg_id": np.where(porque >= 0, self.ids[np.maximum(porque, 0)], -1).astype(np.int64),
        })


mlflow.models.set_model(RecomendadorJuegos())