# ml

Entrenamiento, evaluación y elección del recomendador, modelo de producción y prueba con el
ranking de 2026. Cada entrenamiento y cada medida quedan registrados en MLflow.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `datos.py` | Lee Gold (CSV o MySQL), reparte las notas en ajuste, validación y test, y construye los rasgos del modelo por contenido. No guarda nada: la partición sale igual en cada ejecución porque usa una semilla fija |
| `seguimiento.py` | Conexión con el servidor de MLflow y comprobación de que guarda y lee bien |
| `evaluar.py` | Entrena y mide las 20 variantes en validación, elige la mejor de cada familia y la ganadora, y mide las elegidas en test |
| `produccion.py` | Entrena la ganadora con el 100 % de las notas, la registra en MLflow como `recomendador_juegos` y le pone el alias `produccion` |
| `modelo_mlflow.py` | El modelo tal como lo carga la API: recibe la colección de una persona y devuelve sus recomendaciones con el motivo de cada una |
| `ranking_2026.py` | Prueba aparte: ¿el modelo recomienda juegos que después subieron en el ranking de BGG? |

El código genérico (partición, métricas y recomendadores) está en la librería
`siguiente_partida` (`src/siguiente_partida/`), con sus pruebas en `tests/`. Las decisiones
se explican con los datos reales en `notebooks/`:

- `00_como_funciona_item_knn.ipynb`: el item-kNN con un ejemplo hecho a mano y con los vecinos reales de CATAN.
- `01_por_que_validacion.ipynb`: por qué se elige en validación y no en el test.
- `02_eleccion_de_variantes.ipynb`: qué variante gana, el resultado en test y el acierto según cuántas notas tiene el usuario.

## Uso

Con MySQL y MLflow arrancados (`docker compose up -d`), desde la raíz y en este orden:

```bash
python -m ml.datos              # comprobar que Gold se lee y se parte bien
python -m ml.seguimiento        # comprobar MLflow
python -m ml.evaluar            # validación y test (unos 19 minutos)
python -m ml.produccion         # entrenar la ganadora con el 100 % y registrarla
python -m ml.ranking_2026       # prueba con el ranking de 2026
```

Otras opciones de `evaluar.py`: `--usuarios 20000` (prueba rápida con una muestra; no cuenta
para producción), `--fase test` (repite solo el test con la última selección) y
`--origen mysql` (lee Gold de MySQL en vez de los CSV; mismos datos, más lento).

Todo deja su registro en `logs/ml.log` y en la interfaz de MLflow, `http://localhost:5000`.

## Cómo se evalúa

**Datos.** 18.340.325 valoraciones de 224.558 usuarios sobre 21.924 juegos (Gold). Las
columnas de `COLUMNAS_CON_FUGA` no llegan a leerse.

**Partición.** Por usuario, al azar y con semilla fija (`SEMILLA = 2026`):

| Parte | Cuánto | Valoraciones | Para qué |
|---|---|---|---|
| Ajuste | 64 % | 11.734.722 | Entrenar las 20 variantes |
| Validación | 16 % | 2.938.867 | Elegir la mejor de cada familia y la ganadora |
| Test | 20 % | 3.666.736 | Medir una sola vez a las elegidas, entrenadas con ajuste + validación (80 %) |

Todas las decisiones se toman en validación; el test solo mide.

**Acierto.** Un juego recomendado es un acierto si el usuario le puso un 7 o más en la parte
reservada. Los usuarios sin ningún 7 en esa parte no se pueden puntuar.

**Métricas** (10 recomendaciones por usuario, entre los juegos que no tenía):
- precision@10: aciertos / 10.
- recall@10: aciertos / juegos relevantes del usuario.
- NDCG@10: como el recall, pero un acierto vale más cuanto más arriba esté en la lista.
- Cobertura: proporción del catálogo que se recomienda a algún usuario.
- IC 95 % por bootstrap de usuarios (1.000 remuestras) y, en el test, diferencia pareada con
  la popularidad: cada usuario se compara consigo mismo.

**Variantes** (20 en validación):

| Familia | Variantes |
|---|---|
| Popularidad | número de notas, o número de notas ≥ 7; contadas solo con la parte de entrenamiento |
| Por contenido | rasgos: etiquetas · + numéricos · + descripción; perfil del usuario binario o centrado; 100 vecinos |
| Item-kNN | matriz binaria o de notas centradas; 10, 25, 50, 100, 200 o 400 vecinos |

**Número de vecinos.** La rejilla empezó en 25. Ganó el extremo, así que se amplió a 10 y
volvió a ganar el extremo. Se decidió no bajar más (21/11/2026): entre 10 y 25 vecinos el
NDCG@10 cambia menos de 0,001, y con menos vecinos los usuarios con pocas notas apenas
tendrían juegos candidatos. Si el mejor usa 400 vecinos (el máximo), el log avisa para
ampliar la rejilla.

## Resultados (lote 20261120-190950)

**Validación** (221.726 usuarios):

| Variante | NDCG@10 (IC 95 %) |
|---|---|
| **Item-kNN binario, 10 vecinos (ganadora)** | **0,1416 (0,1408-0,1423)** |
| Item-kNN binario, 25 / 50 / 100 / 200 / 400 vecinos | 0,1407 / 0,1386 / 0,1375 / 0,1379 / 0,1401 |
| Item-kNN centrado (mejor: 10 vecinos) | 0,0948 |
| Popularidad (número de notas / notas ≥ 7) | 0,1024 / 0,1023 |
| Por contenido (mejor: etiquetas + numéricos + descripción, binario) | 0,0204 |

**Test** (222.771 usuarios):

| Modelo | precision@10 | recall@10 | NDCG@10 (IC 95 %) | Diferencia con popularidad (IC 95 %) | Cobertura |
|---|---|---|---|---|---|
| Popularidad | 0,092 | 0,110 | 0,132 (0,131-0,133) | — | 0,3 % |
| Por contenido | 0,018 | 0,022 | 0,025 (0,025-0,025) | −0,107 (−0,108 a −0,106) | 64 % |
| **Item-kNN binario, 10 vecinos** | **0,129** | **0,158** | **0,189 (0,188-0,189)** | **+0,057 (0,056 a 0,057)** | **27 %** |

- El item-kNN supera a la popularidad con un intervalo lejos del 0 y recomienda 80 veces más catálogo.
- **El techo:** la mediana es de 6 juegos relevantes reservados por usuario, así que una lista perfecta sacaría precision@10 = 0,617 y recall@10 = 0,859, no 1. Además, un juego que el usuario no ha valorado cuenta como fallo aunque le pudiera gustar.
- **La variante binaria gana a la centrada:** en BGG, haber valorado un juego dice más de los gustos que la nota concreta.
- **El modelo por contenido queda por debajo de la popularidad:** las mecánicas y los temas, por sí solos, no predicen qué juegos valorará alguien.

**Según cuántas notas ve el modelo de cada usuario** (NDCG@10 en test):

| Notas vistas | Usuarios | Item-kNN | Popularidad | Diferencia |
|---|---|---|---|---|
| ≤ 15 | 54.125 | 0,138 | 0,091 | +0,047 |
| 16-30 | 55.483 | 0,150 | 0,103 | +0,047 |
| 31-60 | 48.857 | 0,164 | 0,114 | +0,050 |
| 61-150 | 42.352 | 0,227 | 0,160 | +0,066 |
| > 150 | 21.954 | 0,390 | 0,291 | +0,100 |

El item-kNN gana en todos los grupos, también con pocas notas, que es el caso de alguien nuevo
en la app. Las cifras absolutas suben con la actividad porque esos usuarios tienen más juegos
escondidos que acertar; lo comparable es la diferencia dentro de cada grupo.

## Modelo de producción

`produccion.py` lee de MLflow la ganadora de la última evaluación completa (las pruebas con
`--usuarios` no cuentan), la entrena con el 100 % de las notas y la registra:

- **Nombre:** `recomendador_juegos`. Cada ejecución crea una versión nueva (1, 2, 3...).
- **Alias `produccion`:** la API carga siempre `models:/recomendador_juegos@produccion`. Para volver a una versión anterior basta con mover el alias desde la interfaz de MLflow.
- **Cifras:** las del test de su lote, copiadas en las etiquetas `test_*` de la versión. Con el 100 % ya no queda nada sin ver con lo que medir.
- **Versión 1:** item-kNN binario de 10 vecinos, entrenado con las 18.340.325 notas en 18 s. Ocupa 1,3 MB. Se guarda sin pickle: el código (`modelo_mlflow.py`) y un `.npz` con los vecinos.

| Entrada (una fila por juego de la persona) | |
|---|---|
| `bgg_id` | id del juego en BGG |
| `nota` | opcional: su nota |
| `excluir` | opcional: `True` = no cuenta para su perfil, solo se evita recomendarlo (p. ej., descartes) |

| Salida (una fila por recomendación; parámetro `k`, 10 por defecto) | |
|---|---|
| `posicion`, `bgg_id` | orden y juego recomendado |
| `puntuacion` | solo sirve para ordenar: no es una nota ni una probabilidad |
| `porque_bgg_id` | el juego de la persona que más pesa en esa recomendación ("porque tienes X"); -1 si no hay ninguno |

Con una colección vacía, o solo con juegos que no están en el modelo, recomienda los más
populares. Ejemplo de la comprobación: a quien tiene Catan, Carcassonne, Ticket to Ride y
Pandemic le recomienda Dominion, 7 Wonders, Small World, Puerto Rico y Power Grid.

## Prueba con el ranking de 2026

El modelo de producción solo ha visto la foto de Kaggle. `ranking_2026.py` le pide 10
recomendaciones para cada usuario y mira cuántas son juegos que **subieron** en el ranking
de BGG del 12/11/2026. La definición se fijó antes de ver resultados:

- **A, principal:** fuera del top 1.000 en la foto antigua y dentro en 2026 (73 juegos).
- **B, robustez:** puesto de 2026 igual o mejor que la mitad del antiguo (207 juegos).

La comparación es con el **azar dentro de la misma zona del ranking antiguo**. Un azar sobre
todo el catálogo elige casi siempre juegos del fondo, que no suben nunca, y cualquier modelo le
ganaría sin anticipar nada. Se comprobó con datos sintéticos sin señal: contra el azar del
catálogo el modelo salía 3,2 veces mejor; contra el de su zona, 1,0 veces, que es lo correcto.

| | Subieron, de cada 1.000 recomendaciones | | |
|---|---|---|---|
| | Modelo | Azar en su zona | Modelo frente a lo esperado |
| A (entran en el top 1.000) | 0,2 | 0,6 | 0,27 veces |
| B (puesto a la mitad o mejor) | 0,7 | 16,7 | 0,04 veces |

**El modelo no anticipa qué juegos van a subir.** Los recomienda menos que el azar de su zona,
y los pocos aciertos se concentran en unos pocos juegos: recomienda 19 de los 73 de la A, y
5 de ellos, casi todos de la serie *Unmatched*, suman el 89 %. Es "más de lo mismo", no
anticipación.

La comparación por zona no descuenta los juegos que cada usuario ya tiene, que el modelo no
puede recomendarle. Eso juega algo en contra del modelo, aunque no basta para explicar
diferencias tan grandes.

## Conclusión

Estoy conforme con el modelo obtenido. El item-kNN binario de 10 vecinos supera a la
popularidad con claridad (+0,057 de NDCG@10 en test, con un IC 95 % de 0,056 a 0,057), en
todos los grupos de usuarios, incluidos los que tienen pocas notas, y recomienda una parte
mucho mayor del catálogo. Además, cada recomendación se puede explicar ("porque tienes X"), y
el modelo es pequeño y rápido para la API.

Tiene limitaciones conocidas:

- **No predice juegos emergentes.** Recomienda juegos que mucha gente ya ha valorado junto con
  los tuyos. Los juegos que luego suben solían ser recientes o tener pocas notas en la foto
  antigua, y casi nunca aparecen como vecinos. La prueba con el ranking de 2026 lo confirma.
- **No conoce los juegos posteriores a la foto de Kaggle,** ni los gustos de ahora: solo puede
  recomendar los 21.924 juegos que tienen notas en Gold.
- **Las notas no tienen fecha.** La prueba principal mide si el modelo recupera gustos
  conocidos, no cómo acertaría con lo que alguien jugará en el futuro.

La causa de fondo es el acceso a los datos. Sin la API de BGG, el modelo depende de una foto
fija del catálogo y de las valoraciones. Con acceso al catálogo y a las notas actualizadas,
se podría reentrenar cada cierto tiempo: conocería los juegos nuevos y los gustos recientes,
y esta limitación se reduciría mucho. Aun así, con los datos disponibles, el resultado
obtenido me parece satisfactorio.

**Trabajo futuro:**
1. Descontar en la prueba del ranking los juegos que cada usuario ya tiene.
2. Dar más peso a los juegos menos conocidos, mezclando el item-kNN con el modelo por contenido
   o penalizando la popularidad, y elegir ese peso en validación midiendo cuánto NDCG cuesta.
3. Con la API de BGG, reentrenar periódicamente con valoraciones actuales.

## Qué se guarda en MLflow

| Experimento | Una ejecución por | Contenido |
|---|---|---|
| `comprobacion` | prueba de conexión | Se borra sola al terminar |
| `seleccion_en_validacion` | variante (20) | Parámetros, métricas con el IC del NDCG@10 y tiempos. Etiquetas `elegida=si` (mejor de cada familia) y `ganadora=si` (mejor de todas) |
| `evaluacion_en_test` | elegida (3) | Métricas con IC, diferencia pareada con la popularidad, techo de las métricas y `metricas_por_usuario.parquet` (sin identificadores). La ganadora lleva `ganadora=si` |
| `produccion` | modelo de producción | El modelo guardado, con las cifras del test de su lote copiadas como `test_*` |
| `ranking_2026` | definición de "subir" (A y B) | Tasas del modelo, la popularidad y el azar, IC de la diferencia y la tabla de juegos que subieron |

Las ejecuciones de una misma evaluación comparten la etiqueta `lote` (fecha y hora), y la
etiqueta `datos` dice si Gold salió de Bronze o de la muestra sintética. Los metadatos van al
esquema `mlflow` de MySQL y los ficheros, al volumen `mlflow_artefactos`.

## Tiempos

Con los datos reales, en local:

| Paso | Tiempo |
|---|---|
| `python -m ml.datos` (leer Gold, partir y rasgos) | 34 s (desde MySQL, 78 s) |
| `python -m ml.evaluar` | 19 min: validación 16 min, test 3 min |
| Medir una variante con todos los usuarios | 25-65 s |
| `python -m ml.produccion` | 53 s |
| `python -m ml.ranking_2026` | 104 s |

## Decisiones técnicas

- **Similitud por bloques:** la matriz completa de similitudes entre juegos (22.000 × 22.000)
  ocuparía casi 2 GB. Se calcula por bloques de 1.024 juegos y de cada uno se guardan solo sus
  vecinos. Con el máximo de vecinos se calcula una vez y los valores menores se sacan de ahí.
- **El usuario entra como vector de notas:** los modelos reciben las notas de una persona, no
  su id. El mismo modelo sirve para los usuarios de BGG en la evaluación y para la colección
  de alguien de la app.
- **Empates por popularidad:** se suma la popularidad del entrenamiento multiplicada por 1e-9.
  Solo decide entre juegos con la misma puntuación (por ejemplo, un usuario sin señal).
- **Reproducibilidad:** con la misma semilla y Gold, dos ejecuciones de `evaluar.py` dan
  exactamente las mismas cifras.
- **Sin pickle:** el modelo de producción se guarda como código y arrays de numpy, así que se
  puede cargar con otras versiones de Python sin riesgo.
- **CSV por defecto:** los CSV de Gold contienen lo mismo que MySQL y se leen más rápido.