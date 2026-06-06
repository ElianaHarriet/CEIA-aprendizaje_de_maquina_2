# Ejemplo de ambiente productivo
### MLOps1 - CEIA - FIUBA
Estructura de servicios para la implementación del proyecto final de MLOps1 - CEIA - FIUBA

El modelo a implementar fue desarrollado como trabajo práctico final de **Aprendizaje de Máquina I**. Se trata de un clasificador binario que predice si un usuario calificará una película con 4 o más estrellas, usando el dataset [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/).

## Integrantes del grupo

| Padrón | Nombre | GitHub |
|--------|--------|--------|
| a2217 | Eliana Harriet | [@ElianaHarriet](https://github.com/ElianaHarriet) |
| a2219 | Alejandro López Bayona | - |
| a2329 | Pablo Santiago Rodríguez Castro | - |
| a2416 | Damian Nicolas Smilovich | - |

El notebook original y la documentación de autoría se encuentran en la carpeta [`modelo/`](./modelo/).

## Instalación

La estructura de servicios se basa en Docker compose, el único prerrequisito es instalar [Docker](https://docs.docker.com/engine/install/). Luego,

1. Cloná este repositorio.
2. Creá las carpetas `airflow/config`, `airflow/dags`, `airflow/logs`, `airflow/plugins`.
3. Si estás en Linux o MacOS, en el archivo `.env`, reemplazá `AIRFLOW_UID` por el de tu usuario o alguno que consideres oportuno (para tu UID, usá el comando `id -u <username>`). De lo contrario, Airflow deja sus carpetas internas como root y no vas a poder subir DAGs (en `airflow/dags`) o plugins, etc.
4. En la carpeta raíz de este repositorio, ejecutá:

```bash
docker compose build airflow-apiserver postgres mlflow fastapi
```

## Ejecución

1. En la carpeta raíz de este repositorio, ejecuta:

```bash
docker compose --profile all up
```

Opcionalmente agregar `-d` para correr en background.

2. Una vez que todos los servicios estén funcionando (verifica con el comando `docker ps -a` que todos los servicios estén `healthy` o revisa en Docker Desktop), podrás acceder a los diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080 (user: airflow, pass: airflow)
   - MLflow: http://localhost:5001
   - MinIO: http://localhost:9001 (user: minio, pass: minio123)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

Si estás usando un servidor externo a tu computadora de trabajo, reemplaza `localhost` por su IP, todos los puertos u otras configuraciones se pueden modificar en el archivo `.env`.

## Apagar los servicios

Al finalizar, se detienen los servicios mediante

```bash
docker compose --profile all down
```

Para además eliminar toda la infraestructura relacionada

```bash
docker compose down --rmi all --volumes
```

Nota: si hacés esto, se pierde todo el contenido de los buckets y bases de datos.