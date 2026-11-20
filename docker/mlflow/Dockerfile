FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.18.0 pymysql==1.2.4 "cryptography>=44"

ENV MLFLOW_DISABLE_TELEMETRY=true
EXPOSE 5000