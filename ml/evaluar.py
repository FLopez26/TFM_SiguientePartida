"""Entrenar, comparar y elegir: validación y test, con cada ejecución registrada en MLflow.

Fase 1, validación (experimento de MLflow "seleccion_en_validacion"):
    las 20 variantes se entrenan con la parte de ajuste (64 %) y se miden en la de
    validación (16 %). De cada familia (popularidad, contenido, item-kNN) se elige la de
    mayor NDCG@10 (etiqueta elegida=si), y la mejor de las tres es la ganadora
    (etiqueta ganadora=si): la que irá a producción.
Fase 2, test (experimento "evaluacion_en_test"):
    las tres elegidas se vuelven a entrenar con ajuste + validación (80 %) y se miden una sola
    vez en el test (20 %): IC 95 % por bootstrap y diferencia pareada con la popularidad.
    El test solo mide: no cambia ninguna elección.

Por qué hay validación: notebooks/01_por_que_validacion.ipynb.

Uso (desde la raíz, con MLflow arrancado: docker compose up -d):
    python -m ml.evaluar                       # las dos fases (unos 20 minutos)
    python -m ml.evaluar --usuarios 20000      # prueba rápida: solo 20.000 usuarios evaluados
    python -m ml.evaluar --fase validacion
    python -m ml.evaluar --fase test           # repite el test con la última selección
    python -m ml.evaluar --origen mysql        # Gold desde MySQL en vez de los CSV
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Primero seguimiento: configura MLflow (sin telemetría ni avisos) antes de importarlo
from ml.seguimiento import conectar  # noqa: I001

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ml.datos import (AJUSTE, CONJUNTOS_RASGOS, FRACCION_TEST, FRACCION_VALIDACION, SEMILLA,
                      TEST, UMBRAL_RELEVANTE, VALIDACION, Valoraciones, leer_catalogo,
                      origen_gold, preparar_valoraciones, rasgos_contenido)
from pipeline.rutas import LOGS, ROOT
from siguiente_partida import configurar_logging
from siguiente_partida.metricas import cobertura, ic_bootstrap, metricas_por_usuario
from siguiente_partida.recomendadores import VARIANTES, ItemKNN, PorContenido, Popularidad

log = logging.getLogger("ml.evaluar")

K = 10                    # tamaño de la lista recomendada
N_REMUESTRAS = 1000       # bootstrap
EXPERIMENTO_VALIDACION = "seleccion_en_validacion"
EXPERIMENTO_TEST = "evaluacion_en_test"
FAMILIAS = ("popularidad", "contenido", "item_knn")
VECINOS_KNN = (10, 25, 50, 100, 200, 400)
VECINOS_CONTENIDO = 100
METRICAS = ("precision", "recall", "ndcg")


# --------------------------------------------------------------------------- variantes
def rejilla() -> list[dict]:
    """Las 20 variantes que se comparan en validación."""
    variantes = [{"familia": "popularidad", "criterio": c} for c in ("valoraciones", "positivas")]
    variantes += [{"familia": "contenido", "rasgos": r, "variante": v, "vecinos": VECINOS_CONTENIDO}
                  for r in CONJUNTOS_RASGOS for v in VARIANTES]
    variantes += [{"familia": "item_knn", "variante": v, "vecinos": k}
                  for v in VARIANTES for k in VECINOS_KNN]
    return variantes


def nombre(spec: dict) -> str:
    """Nombre legible de una variante, p. ej. item_knn-binaria-k100."""
    partes = [spec["familia"]] + [str(spec[c]) for c in ("criterio", "rasgos", "variante") if c in spec]
    if "vecinos" in spec:
        partes.append(f"k{spec['vecinos']}")
    return "-".join(partes)


def clave_ajuste(spec: dict) -> tuple:
    """Variantes con la misma clave comparten el entrenamiento: solo cambia k o el perfil."""
    f = spec["familia"]
    return f, spec.get("criterio"), spec.get("rasgos"), spec.get("variante") if f == "item_knn" else None


class Rasgos:
    """Rasgos de contenido, calculados la primera vez que se piden."""

    def __init__(self, juegos: np.ndarray, origen: str):
        self.juegos, self.origen, self._cache, self._catalogo = juegos, origen, {}, None

    def __getitem__(self, conjunto: str):
        if conjunto not in self._cache:
            if self._catalogo is None:
                self._catalogo = leer_catalogo(self.origen)
            self._cache[conjunto] = rasgos_contenido(*self._catalogo, self.juegos, conjunto)
        return self._cache[conjunto]


def construir(spec: dict, rasgos: Rasgos, max_vecinos: int | None = None):
    """El modelo de la librería que corresponde a una variante, sin entrenar."""
    f = spec["familia"]
    if f == "popularidad":
        return Popularidad(spec["criterio"], UMBRAL_RELEVANTE)
    if f == "contenido":
        return PorContenido(rasgos[spec["rasgos"]], spec["vecinos"], spec["variante"], max_vecinos)
    if f == "item_knn":
        return ItemKNN(spec["vecinos"], spec["variante"], max_vecinos)
    raise ValueError(f"familia desconocida: {f}")


# --------------------------------------------------------------------------- medida
def usuarios_a_evaluar(relevantes, n_max: int | None) -> np.ndarray:
    """Filas con al menos un relevante en la parte reservada (las demás no se pueden puntuar)."""
    filas = np.flatnonzero(np.diff(relevantes.indptr) > 0)
    if n_max and n_max < len(filas):
        filas = np.sort(np.random.default_rng(SEMILLA).choice(filas, n_max, replace=False))
    return filas


def medir(modelo, R_entrada, relevantes, filas: np.ndarray) -> tuple[dict, np.ndarray, float]:
    """Recomienda 10 juegos a cada usuario y calcula sus métricas frente a los relevantes."""
    inicio = time.perf_counter()
    top, _ = modelo.recomendar(R_entrada[filas], k=K)
    return metricas_por_usuario(top, relevantes[filas]), top, time.perf_counter() - inicio


def _f(x: float) -> str:
    return f"{x:.4f}"


# --------------------------------------------------------------------------- fase 1: validación
def fase_validacion(datos: Valoraciones, rasgos: Rasgos, args,
                    contexto: dict) -> tuple[list[dict], dict]:
    """Entrena y mide las 20 variantes. Devuelve (elegida de cada familia, ganadora)."""
    inicio_fase = time.perf_counter()
    R = datos.matriz(AJUSTE)
    rel = datos.relevantes(VALIDACION)
    filas = usuarios_a_evaluar(rel, args.usuarios)
    log.info("Validación: inicio, %d variantes, %d usuarios evaluados", len(rejilla()), len(filas))
    mlflow.set_experiment(EXPERIMENTO_VALIDACION)

    grupos: dict[tuple, list[dict]] = {}
    for spec in rejilla():
        grupos.setdefault(clave_ajuste(spec), []).append(spec)

    resultados = []
    for specs in grupos.values():
        kmax = max(s.get("vecinos", 0) for s in specs) or None
        modelo = construir(specs[0], rasgos, max_vecinos=kmax)
        inicio = time.perf_counter()
        modelo.ajustar(R)
        t_ajuste = time.perf_counter() - inicio
        for spec in specs:
            if "vecinos" in spec:
                modelo.vecinos = spec["vecinos"]
            if spec["familia"] == "contenido":
                modelo.variante = spec["variante"]   # solo cambia el perfil del usuario
            m, top, t_medida = medir(modelo, R, rel, filas)
            ndcg, inf, sup = ic_bootstrap(m["ndcg"], N_REMUESTRAS, semilla=SEMILLA)
            metricas = {"precision_10": float(np.mean(m["precision"])),
                        "recall_10": float(np.mean(m["recall"])),
                        "ndcg_10": ndcg, "ndcg_10_ic_inf": inf, "ndcg_10_ic_sup": sup,
                        "cobertura": cobertura(top, datos.forma[1]),
                        "usuarios_evaluados": len(filas),
                        "segundos_ajuste": t_ajuste, "segundos_medida": t_medida}
            with mlflow.start_run(run_name=nombre(spec), tags={
                    "fase": "validacion", "lote": contexto["lote"], "familia": spec["familia"],
                    "datos": contexto["datos"]}) as run:
                mlflow.log_params({**spec, **contexto["params"], "entrenado_con": "ajuste (64 %)",
                                   "medido_en": "validacion"})
                mlflow.log_metrics(metricas)
            resultados.append({"spec": spec, "run_id": run.info.run_id, **metricas})
            log.info("Validación %-56s NDCG@10 %s (IC %s-%s)  cobertura %.3f  (%.0f s)",
                     nombre(spec), _f(ndcg), _f(inf), _f(sup), metricas["cobertura"], t_medida)
        del modelo

    cliente = MlflowClient()
    mejores = []
    for familia in FAMILIAS:
        mejor = max((r for r in resultados if r["spec"]["familia"] == familia),
                    key=lambda r: r["ndcg_10"])
        cliente.set_tag(mejor["run_id"], "elegida", "si")
        mejores.append(mejor)
        log.info("Validación: elegida de %-12s %s (NDCG@10 %s)", familia, nombre(mejor["spec"]),
                 _f(mejor["ndcg_10"]))
        if familia == "item_knn" and mejor["spec"]["vecinos"] == max(VECINOS_KNN):
            log.warning("Validación: el mejor item-kNN usa el máximo de vecinos probado (%d). "
                        "Puede que más vecinos sea aún mejor: amplía VECINOS_KNN",
                        max(VECINOS_KNN))
        elif familia == "item_knn" and mejor["spec"]["vecinos"] == min(VECINOS_KNN):
            log.info("Validación: el mejor item-kNN usa el mínimo probado (%d vecinos). Es un "
                     "límite decidido: por debajo el NDCG apenas cambia y los usuarios con pocas "
                     "notas se quedan sin candidatos (ver ml/README.md)", min(VECINOS_KNN))
    ganadora = max(mejores, key=lambda r: r["ndcg_10"])
    cliente.set_tag(ganadora["run_id"], "ganadora", "si")
    log.info("Validación: GANADORA %s (NDCG@10 %s). Es la que irá a producción",
             nombre(ganadora["spec"]), _f(ganadora["ndcg_10"]))
    log.info("Validación: fin (%.0f s)", time.perf_counter() - inicio_fase)
    return [r["spec"] for r in mejores], ganadora["spec"]


def ultima_seleccion() -> tuple[list[dict], dict]:
    """Las elegidas y la ganadora de la última validación registrada en MLflow."""
    exp = mlflow.get_experiment_by_name(EXPERIMENTO_VALIDACION)
    if exp is None:
        raise ValueError("No hay ninguna validación en MLflow: ejecuta antes --fase validacion")
    runs = mlflow.search_runs([exp.experiment_id], filter_string="tags.elegida = 'si'",
                              order_by=["attributes.start_time DESC"], output_format="list")
    if not runs:
        raise ValueError("La validación de MLflow no tiene variantes elegidas")
    lote = runs[0].data.tags["lote"]
    elegidas, ganadora = [], None
    for run in runs:
        if run.data.tags.get("lote") != lote:
            continue
        p = run.data.params
        spec = {c: p[c] for c in ("familia", "criterio", "rasgos", "variante") if c in p}
        if "vecinos" in p:
            spec["vecinos"] = int(p["vecinos"])
        elegidas.append(spec)
        if run.data.tags.get("ganadora") == "si":
            ganadora = spec
    if ganadora is None:
        raise ValueError(f"El lote {lote} de validación no tiene ganadora")
    elegidas.sort(key=lambda s: FAMILIAS.index(s["familia"]))
    log.info("Test: selección del lote %s: %s; ganadora %s", lote,
             ", ".join(nombre(s) for s in elegidas), nombre(ganadora))
    return elegidas, ganadora


# --------------------------------------------------------------------------- fase 2: test
def fase_test(datos: Valoraciones, rasgos: Rasgos, elegidas: list[dict], ganadora: dict, args,
              contexto: dict) -> pd.DataFrame:
    """Reentrena las elegidas con el 80 % y las mide una vez en el test."""
    if elegidas[0]["familia"] != "popularidad":
        raise ValueError("La primera elegida tiene que ser la popularidad (es la referencia)")
    inicio_fase = time.perf_counter()
    R = datos.matriz(AJUSTE, VALIDACION)
    rel = datos.relevantes(TEST)
    filas = usuarios_a_evaluar(rel, args.usuarios)
    log.info("Test: inicio, %d modelos, %d usuarios evaluados", len(elegidas), len(filas))
    # Techo: lo que sacaría una lista perfecta. Quien tiene menos de 10 relevantes reservados
    # no puede llegar a precision@10 = 1 ni con la lista ideal
    n_rel = np.diff(rel.indptr)[filas]
    techo = {"precision_10_maxima": float(np.mean(np.minimum(n_rel, K) / K)),
             "recall_10_maximo": float(np.mean(np.minimum(n_rel, K) / n_rel))}
    log.info("Test: relevantes reservados por usuario: mediana %d; con lista perfecta, "
             "precision@10 = %.3f y recall@10 = %.3f", np.median(n_rel),
             techo["precision_10_maxima"], techo["recall_10_maximo"])
    mlflow.set_experiment(EXPERIMENTO_TEST)
    por_usuario_base = None
    resumen = []
    for spec in elegidas:
        modelo = construir(spec, rasgos)
        inicio = time.perf_counter()
        modelo.ajustar(R)
        t_ajuste = time.perf_counter() - inicio
        m, top, t_medida = medir(modelo, R, rel, filas)
        M = np.vstack([m[c] for c in METRICAS])
        media, inf, sup = ic_bootstrap(M, N_REMUESTRAS, semilla=SEMILLA)
        metricas = {"cobertura": cobertura(top, datos.forma[1]), "usuarios_evaluados": len(filas),
                    "segundos_ajuste": t_ajuste, "segundos_medida": t_medida, **techo}
        for i, c in enumerate(METRICAS):
            metricas |= {f"{c}_10": media[i], f"{c}_10_ic_inf": inf[i], f"{c}_10_ic_sup": sup[i]}
        if por_usuario_base is None:
            por_usuario_base = M
        else:   # mismos usuarios y mismas remuestras: diferencia pareada
            d, d_inf, d_sup = ic_bootstrap(M - por_usuario_base, N_REMUESTRAS, semilla=SEMILLA)
            for i, c in enumerate(METRICAS):
                metricas |= {f"dif_{c}_10": d[i], f"dif_{c}_10_ic_inf": d_inf[i],
                             f"dif_{c}_10_ic_sup": d_sup[i]}

        es_ganadora = spec == ganadora
        etiquetas = {"fase": "test", "lote": contexto["lote"], "familia": spec["familia"],
                     "datos": contexto["datos"]}
        if es_ganadora:
            etiquetas["ganadora"] = "si"
        with mlflow.start_run(run_name=nombre(spec), tags=etiquetas):
            mlflow.log_params({**spec, **contexto["params"],
                               "entrenado_con": "ajuste + validacion (80 %)", "medido_en": "test"})
            mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
            # Métricas por usuario, sin identificadores, para el análisis de errores
            tabla = pd.DataFrame({c: m[c] for c in METRICAS})
            tabla["n_valoraciones_entrada"] = np.diff(R.indptr)[filas]
            tabla["n_relevantes"] = np.diff(rel.indptr)[filas]
            with tempfile.TemporaryDirectory() as tmp:
                ruta = Path(tmp) / "metricas_por_usuario.parquet"
                tabla.to_parquet(ruta, index=False)
                mlflow.log_artifact(str(ruta))

        texto_dif = ""
        if "dif_ndcg_10" in metricas:
            texto_dif = (f"  dif. con popularidad {metricas['dif_ndcg_10']:+.4f} "
                         f"(IC {_f(metricas['dif_ndcg_10_ic_inf'])} a "
                         f"{_f(metricas['dif_ndcg_10_ic_sup'])})")
        log.info("Test %-62s NDCG@10 %s (IC %s-%s)%s", nombre(spec), _f(metricas["ndcg_10"]),
                 _f(metricas["ndcg_10_ic_inf"]), _f(metricas["ndcg_10_ic_sup"]), texto_dif)
        resumen.append({"modelo": nombre(spec), "ganadora": "si" if es_ganadora else "",
                        **{k: metricas.get(k) for k in (
                            "precision_10", "recall_10", "ndcg_10", "dif_ndcg_10",
                            "dif_ndcg_10_ic_inf", "dif_ndcg_10_ic_sup", "cobertura")}})
        del modelo

    tabla = pd.DataFrame(resumen)
    gan = tabla[tabla["ganadora"] == "si"].iloc[0]
    if gan["modelo"] != tabla.loc[tabla["ndcg_10"].idxmax(), "modelo"]:
        log.warning("Test: la ganadora en validación (%s) no es la de mayor NDCG@10 en test. "
                    "No se cambia la elección: el test solo mide", gan["modelo"])
    if not pd.isna(gan["dif_ndcg_10_ic_inf"]) and gan["dif_ndcg_10_ic_inf"] <= 0:
        log.warning("Test: la ganadora no supera a la popularidad con un IC que excluya el 0")
    log.info("Test: fin (%.0f s)", time.perf_counter() - inicio_fase)
    return tabla


# --------------------------------------------------------------------------- principal
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fase", choices=["todo", "validacion", "test"], default="todo")
    parser.add_argument("--usuarios", type=int, default=None,
                        help="evalúa solo una muestra fija de N usuarios (para pruebas rápidas)")
    parser.add_argument("--origen", choices=["csv", "mysql"], default="csv",
                        help="de dónde leer Gold (por defecto, data/gold/*.csv)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    configurar_logging("ml.log", carpeta=LOGS)
    inicio = time.perf_counter()
    log.info("Evaluación: inicio (fase %s, Gold desde %s, usuarios evaluados: %s)", args.fase,
             args.origen, args.usuarios or "todos")

    try:
        conectar()
        datos = preparar_valoraciones(args.origen)
        tam = datos.resumen()
        contexto = {
            "lote": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "datos": origen_gold(),
            "params": {"semilla": SEMILLA, "fraccion_test": FRACCION_TEST,
                       "fraccion_validacion": FRACCION_VALIDACION,
                       "umbral_relevante": UMBRAL_RELEVANTE, "k": K, "lectura_gold": args.origen,
                       "muestra_usuarios": args.usuarios or "todos",
                       "n_valoraciones": tam["valoraciones"], "n_usuarios": tam["usuarios"],
                       "n_juegos": tam["juegos"]},
        }
        log.info("Evaluación: lote %s", contexto["lote"])
        if contexto["datos"] == "ejemplo":
            log.warning("Gold se generó con la muestra sintética: las cifras no son las reales")
        if args.usuarios:
            log.warning("Evaluación con una muestra de %d usuarios: es una prueba, no el resultado "
                        "final", args.usuarios)
        rasgos = Rasgos(datos.juegos, args.origen)

        if args.fase in ("todo", "validacion"):
            elegidas, ganadora = fase_validacion(datos, rasgos, args, contexto)
        else:
            elegidas, ganadora = ultima_seleccion()
        if args.fase in ("todo", "test"):
            tabla = fase_test(datos, rasgos, elegidas, ganadora, args, contexto)
            log.info("Resultado en test (lote %s):\n%s", contexto["lote"],
                     tabla.round(4).to_string(index=False))
    except (ValueError, FileNotFoundError, MlflowException) as e:
        log.error("%s", e)
        return 1
    except Exception:
        log.exception("Error inesperado en la evaluación")
        return 1
    log.info("Evaluación: fin (%.0f s)", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())