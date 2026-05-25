#!/bin/bash
set -e

# Si el comando es un servicio principal de Airflow, importar variables/conexiones al metadata DB y luego arrancar el servicio
case "$1" in
  api-server|scheduler|worker|triggerer)
    if [ -f /opt/airflow/secrets/variables.yaml ]; then
      airflow variables import /opt/airflow/secrets/variables.yaml || true
    fi
    if [ -f /opt/airflow/secrets/connections.yaml ]; then
      airflow connections import /opt/airflow/secrets/connections.yaml || true
    fi
    exec /entrypoint "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
