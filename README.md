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

## Uso de la API

Una vez que los servicios estén corriendo y el modelo esté entrenado:

```bash
# Predecir rating (enviar 50 features en el orden de entrenamiento)
curl -X POST http://localhost:8800/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.5,2.5,1.2,1995,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,3.8,0.3,4.1,1.0,0.5,0.0]}'

# Health check
curl http://localhost:8800/health

# Documentación interactiva
open http://localhost:8800/docs
```

> **Nota:** la API carga el modelo `champion` **una sola vez, al arrancar**. Cuando el
> retrain promueve un nuevo champion (o lo cambiás manualmente), reiniciá el contenedor
> para que la API sirva el modelo actualizado:
>
> ```bash
> docker compose restart fastapi
> ```

## Tests

Los tests unitarios se ejecutan localmente sin Docker:

```bash
# Instalar dependencias
uv sync

# Ejecutar todos los tests
uv run pytest tests/ -v

# Tests específicos
uv run pytest tests/test_fastapi_predict.py -v
uv run pytest tests/test_etl_dag.py -v
```

Actualmente hay **36 tests** que cubren: API endpoints, lógica de champion/challenger, y estructura de los 3 DAGs de Airflow.

## Pipeline automático

El pipeline de 3 DAGs se encadena automáticamente:

1. Disparás `etl_movielens` (único trigger manual)
2. Al terminar, dispara `train_movielens` automáticamente
3. Al terminar, dispara `retrain_movielens` automáticamente

Cada DAG también se puede disparar manualmente desde la UI de Airflow.

> **Importante:** las DAGs se crean **pausadas**. Antes de disparar `etl_movielens`,
> despausá las 3 (`etl_movielens`, `train_movielens`, `retrain_movielens`) con el
> toggle de cada DAG en la UI, o por CLI:
>
> ```bash
> docker compose exec airflow-scheduler airflow dags unpause etl_movielens
> docker compose exec airflow-scheduler airflow dags unpause train_movielens
> docker compose exec airflow-scheduler airflow dags unpause retrain_movielens
> ```
>
> Si una DAG queda pausada, su run disparado queda en cola y no se ejecuta hasta despausarla.