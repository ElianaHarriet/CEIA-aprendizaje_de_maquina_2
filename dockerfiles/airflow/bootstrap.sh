#!/bin/bash
set -e

# Si el comando es un servicio principal de Airflow, importar variables/conexiones al metadata DB y luego arrancar el servicio
case "$1" in
  api-server|scheduler|worker|triggerer|dag-processor)
    if [ -f /opt/airflow/secrets/variables.json ]; then
      airflow variables import /opt/airflow/secrets/variables.json || true
    fi
    if [ -f /opt/airflow/secrets/connections.json ]; then
      airflow connections import /opt/airflow/secrets/connections.json || true
    fi
    exec /entrypoint "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
