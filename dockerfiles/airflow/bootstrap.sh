#!/bin/bash
set -e

# Si el comando es un servicio principal de Airflow, importar variables/conexiones al metadata DB y luego arrancar el servicio
case "$1" in
  api-server|scheduler|worker|triggerer|dag-processor)
    if [ -f /opt/secrets/variables.json ]; then
      airflow variables import /opt/secrets/variables.json || true
    fi
    if [ -f /opt/secrets/connections.json ]; then
      airflow connections import /opt/secrets/connections.json || true
    fi
    exec /entrypoint "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
