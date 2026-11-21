# API de Siguiente Partida (FastAPI). Se construye desde la raíz del repositorio:
#   docker compose build api
# Solo entra la carpeta api/ (ver .dockerignore): la librería se instala desde PyPI y el modelo
# se descarga de MLflow al arrancar, así que la imagen no lleva datos ni modelos.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Sin privilegios de root: la API solo necesita escribir en logs/
RUN useradd --create-home --uid 1000 api \
    && mkdir -p /app/logs && chown api:api /app/logs
WORKDIR /app
COPY api/ api/
USER api

# Dentro del contenedor escucha en todas sus interfaces; docker-compose.yml solo publica el
# puerto en el 127.0.0.1 del PC
ENV API_HOST=0.0.0.0 \
    API_PORT=8000
EXPOSE 8000
CMD ["python", "-m", "api.main"]