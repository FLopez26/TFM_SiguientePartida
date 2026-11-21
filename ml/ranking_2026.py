"""Prueba aparte con el ranking de 2026: ¿el modelo recomienda juegos que después subieron?

El modelo de producción se entrenó solo con la foto antigua (Kaggle). El ranking de BGG del
12/11/2026 no lo ha visto nunca. Aquí se mira si sus recomendaciones incluyen juegos que
subieron en ese ranking más de lo que lo haría una lista al azar.

Qué cuenta como "subir" (decidido antes de ver ningún resultado, 21/11/2026):
    A (principal)  fuera del top 1.000 en la foto antigua y dentro en 2026
    B (robustez)   su puesto en 2026 es la mitad del antiguo o mejor
Solo pueden contar juegos que estén en el modelo y en los dos rankings. El ranking antiguo
(rank_general) está en COLUMNAS_CON_FUGA porque ningún modelo puede usarlo como rasgo; aquí
no entra en el modelo: solo sirve para decidir qué juegos subieron.

Qué se mide, con todos los usuarios de Gold y todas sus notas como colección:
    - qué parte de las recomendaciones (10 por usuario) son juegos que subieron, con el modelo
      de producción y con la popularidad (referencia)
    - la comparación justa: el azar dentro de la misma zona del ranking. Un juego del puesto
      1.500 se compara con lo que suben, de media, los juegos de esa zona (1.001-2.000). Un azar
      sobre todo el catálogo no sirve: elige casi siempre juegos del fondo, que no suben nunca,
      y cualquier modelo que recomiende juegos algo conocidos le ganaría sin anticipar nada
    - cuántas veces lo esperado por zona y la diferencia, con IC 95 % por bootstrap de usuarios
    - qué juegos de los que subieron se recomiendan, y a cuántos usuarios: así se ve si el
      resultado se apoya en muchos juegos o en un puñado

Uso (desde la raíz, con MLflow arrancado y después de python -m ml.produccion):
    python -m ml.ranking_2026
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

# Primero seguimiento: configura MLflow (sin telemetría ni avisos) antes de importarlo
from ml.seguimiento import conectar  # noqa: I001

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ml.datos import AJUSTE, SEMILLA, TEST, VALIDACION, origen_gold, preparar_valoraciones
from ml.produccion import ALIAS, NOMBRE_REGISTRO
from pipeline.rutas import GOLD, LOGS, ROOT
from siguiente_partida import configurar_logging
from siguiente_partida.metricas import ic_bootstrap
from siguiente_partida.recomendadores import Popularidad, cargar_recomendador

log = logging.getLogger("ml.ranking_2026")

EXPERIMENTO = "ranking_2026"
K = 10
TOP = 1000
# Zonas del ranking antiguo para el azar justo; los juegos sin puesto forman su propia zona
ZONAS = [0, 1000, 2000, 4000, 8000, np.inf]
DEFINICIONES = {
    "A_entra_top_1000": "fuera del top 1.000 en la foto antigua y dentro en 2026 (principal)",
    "B_mitad_o_mejor": "puesto de 2026 igual o mejor que la mitad del antiguo (robustez)",
}


def cargar_produccion() -> tuple[object, np.ndarray, str]:
    """(modelo, bgg_id de cada columna, versión) del modelo con el alias de producción."""
    version = MlflowClient().get_model_version_by_alias(NOMBRE_REGISTRO, ALIAS).version
    with tempfile.TemporaryDirectory() as tmp:
        carpeta = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{NOMBRE_REGISTRO}@{ALIAS}", dst_path=tmp)
        modelo, ids = cargar_recomendador(Path(carpeta) / "artifacts" / "modelo.npz")
    log.info("Modelo: %s versión %s (alias '%s')", NOMBRE_REGISTRO, version, ALIAS)
    return modelo, ids, version


def leer_rankings(juegos: np.ndarray) -> pd.DataFrame:
    """Nombre, puesto antiguo y puesto de 2026 de cada juego del modelo (NaN si no tiene)."""
    opciones = pv.ParseOptions(newlines_in_values=True)
    rutas = {t: GOLD / f"{t}.csv" for t in ("juego", "ranking_2026")}
    for ruta in rutas.values():
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {ruta}: ejecuta python -m pipeline.gold")
    antiguo = pv.read_csv(rutas["juego"], parse_options=opciones, convert_options=pv.ConvertOptions(
        include_columns=["bgg_id", "nombre", "rank_general"],
        column_types={"bgg_id": pa.int32(), "nombre": pa.string(), "rank_general": pa.float64()})
    ).to_pandas().set_index("bgg_id")
    nuevo = pv.read_csv(rutas["ranking_2026"], parse_options=opciones, convert_options=pv.ConvertOptions(
        include_columns=["bgg_id", "puesto"],
        column_types={"bgg_id": pa.int32(), "puesto": pa.float64()})
    ).to_pandas().set_index("bgg_id")
    tabla = pd.DataFrame(index=pd.Index(juegos, name="bgg_id"))
    tabla["nombre"] = antiguo["nombre"].reindex(juegos).to_numpy()
    tabla["rank_antiguo"] = antiguo["rank_general"].reindex(juegos).to_numpy()
    tabla["puesto_2026"] = nuevo["puesto"].reindex(juegos).to_numpy()
    en_las_dos = tabla["rank_antiguo"].notna() & tabla["puesto_2026"].notna()
    log.info("Rankings: %d de %d juegos del modelo están en las dos fotos", en_las_dos.sum(),
             len(juegos))
    return tabla


def juegos_que_subieron(r: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscara (una posición por juego del modelo) de cada definición de "subir"."""
    en_las_dos = r["rank_antiguo"].notna() & r["puesto_2026"].notna()
    return {
        "A_entra_top_1000": (en_las_dos & (r["rank_antiguo"] > TOP) & (r["puesto_2026"] <= TOP)).to_numpy(),
        "B_mitad_o_mejor": (en_las_dos & (r["puesto_2026"] <= r["rank_antiguo"] / 2)).to_numpy(),
    }


def main() -> int:
    load_dotenv(ROOT / ".env")
    configurar_logging("ml.log", carpeta=LOGS)
    inicio = time.perf_counter()
    log.info("Ranking 2026: inicio")
    try:
        conectar()
        modelo, ids, version = cargar_produccion()
        datos = preparar_valoraciones("csv")
        if len(ids) != len(datos.juegos) or (ids != datos.juegos).any():
            raise ValueError("El modelo de producción se entrenó con otro Gold: ejecuta python -m "
                             "ml.produccion")
        R = datos.matriz(AJUSTE, VALIDACION, TEST)   # todas las notas: la colección de cada usuario
        rankings = leer_rankings(datos.juegos)

        t = time.perf_counter()
        top_modelo, _ = modelo.recomendar(R, k=K)
        top_popular, _ = Popularidad("valoraciones").ajustar(R).recomendar(R, k=K)
        log.info("Recomendaciones: %d usuarios x %d juegos, modelo y popularidad (%.0f s)",
                 R.shape[0], K, time.perf_counter() - t)
        veces_modelo = np.bincount(top_modelo.ravel(), minlength=len(ids))
        veces_popular = np.bincount(top_popular.ravel(), minlength=len(ids))
        candidatos = len(ids) - np.diff(R.indptr)   # juegos que cada usuario no tiene
        zona = pd.cut(rankings["rank_antiguo"], ZONAS, labels=False).fillna(len(ZONAS) - 1)
        zona = zona.to_numpy().astype(int)

        mlflow.set_experiment(EXPERIMENTO)
        for definicion, subio in juegos_que_subieron(rankings).items():
            n_subieron = int(subio.sum())
            if n_subieron == 0:
                log.warning("%s: ningún juego cumple la definición", definicion)
                continue
            # Aciertos por usuario: cuántos de sus 10 juegos recomendados subieron
            aciertos_modelo = subio[top_modelo].sum(axis=1)
            aciertos_popular = subio[top_popular].sum(axis=1)
            # Azar en todo el catálogo (solo como referencia): valor esperado exacto
            subieron_que_no_tiene = n_subieron - np.asarray(R @ subio.astype(np.float32)).ravel()
            esperado_azar = K * subieron_que_no_tiene / candidatos
            # Azar justo: cada juego recomendado vale lo que suben, de media, los de su zona
            tasa_por_zona = (np.bincount(zona, weights=subio, minlength=len(ZONAS))
                             / np.maximum(np.bincount(zona, minlength=len(ZONAS)), 1))
            esperado_zona = tasa_por_zona[zona][top_modelo].sum(axis=1)

            tasa = {"modelo": aciertos_modelo.mean() / K, "popularidad": aciertos_popular.mean() / K,
                    "azar": esperado_azar.mean() / K, "misma_zona": esperado_zona.mean() / K}
            dif, inf, sup = ic_bootstrap((aciertos_modelo - esperado_zona) / K, semilla=SEMILLA)
            veces = tasa["modelo"] / tasa["misma_zona"] if tasa["misma_zona"] > 0 else float("nan")

            # Juego a juego: a cuántos usuarios se recomienda cada uno de los que subieron
            juegos = rankings[subio].assign(usuarios_modelo=veces_modelo[subio],
                                            usuarios_popularidad=veces_popular[subio])
            juegos = juegos.sort_values("usuarios_modelo", ascending=False)
            recomendados = int((juegos["usuarios_modelo"] > 0).sum())
            total = juegos["usuarios_modelo"].sum()
            peso_top5 = juegos["usuarios_modelo"].head(5).sum() / total if total else float("nan")

            log.info("%s (%s): %d juegos subieron", definicion, DEFINICIONES[definicion], n_subieron)
            log.info("%s: de cada 1.000 recomendaciones subieron %.1f con el modelo y %.1f con la "
                     "popularidad; al azar en la misma zona del ranking, %.1f (en todo el "
                     "catálogo, %.1f)", definicion, 1000 * tasa["modelo"],
                     1000 * tasa["popularidad"], 1000 * tasa["misma_zona"], 1000 * tasa["azar"])
            log.info("%s: el modelo saca %.2f veces lo esperado por zona; diferencia %+.4f por "
                     "recomendación (IC 95 %% %.4f a %.4f)", definicion, veces, dif, inf, sup)
            log.info("%s: el modelo recomienda %d de los %d juegos que subieron; los 5 más "
                     "recomendados suman el %.0f %% de esos aciertos", definicion, recomendados,
                     n_subieron, 100 * peso_top5)
            for fila in juegos.head(5).itertuples():
                log.info("%s:   %s (puesto %d -> %d), recomendado a %d usuarios", definicion,
                         fila.nombre, fila.rank_antiguo, fila.puesto_2026, fila.usuarios_modelo)

            with mlflow.start_run(run_name=definicion, tags={
                    "fase": "ranking_2026", "datos": origen_gold(),
                    "modelo": f"{NOMBRE_REGISTRO}@{ALIAS} v{version}"}), \
                    tempfile.TemporaryDirectory() as tmp:
                mlflow.log_params({"definicion": DEFINICIONES[definicion], "k": K,
                                   "version_modelo": version, "usuarios": R.shape[0]})
                mlflow.log_metrics({
                    "juegos_que_subieron": n_subieron, "subieron_recomendados": recomendados,
                    "tasa_modelo": tasa["modelo"], "tasa_popularidad": tasa["popularidad"],
                    "tasa_azar_catalogo": tasa["azar"], "tasa_azar_misma_zona": tasa["misma_zona"],
                    "veces_lo_esperado_por_zona": veces, "dif_modelo_zona": dif,
                    "dif_modelo_zona_ic_inf": inf, "dif_modelo_zona_ic_sup": sup,
                    "peso_5_mas_recomendados": peso_top5})
                ruta = Path(tmp) / f"juegos_que_subieron_{definicion}.csv"
                juegos.reset_index().to_csv(ruta, index=False)
                mlflow.log_artifact(str(ruta))
    except (ValueError, FileNotFoundError, MlflowException) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado en la prueba del ranking de 2026")
        return 1
    log.info("Ranking 2026: fin (%.0f s)", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())