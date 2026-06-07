"""
train_movielens.py - DAG de entrenamiento para el modelo MovieLens.

Lee los splits train/test generados por `etl_movielens` desde MinIO, ejecuta
la busqueda de hiperparametros con Optuna, entrena el modelo final, lo calibra
y lo registra en MLflow.

Flujo de datos:
    s3://data/final/{X_train,X_test,y_train,y_test}.npy
    s3://data/final/feature_names.txt
            v
    train_and_register_model()
        |- MLflow experiment: movielens-rating-prediction
           Registered model: movielens-rating-classifier
"""

import datetime

from airflow.decorators import dag, task

markdown_text = """
### Training Pipeline - MovieLens

Entrena el clasificador binario de MovieLens usando los artefactos generados por
`etl_movielens` en MinIO y persiste el modelo en MLflow exactamente con la misma
logica de registro que usa `modelo/src/train.py`.

**Etapas:**
1. `train_and_register_model` - Lee `s3://data/final/`, optimiza hiperparametros,
   entrena, calibra y registra el modelo en MLflow.

**Inputs:** `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`, `feature_names.txt`

**Output final:** modelo logueado en MLflow y registrado como
`movielens-rating-classifier`
"""

default_args = {
    "owner": "CEIA - FIUBA",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="train_movielens",
    description="Entrenamiento y registro en MLflow del modelo MovieLens.",
    doc_md=markdown_text,
    tags=["train", "MovieLens", "MLflow"],
    default_args=default_args,
    schedule=None,
    dagrun_timeout=datetime.timedelta(minutes=180),
    catchup=False,
)
def train_movielens():

    @task.virtualenv(
        task_id="train_and_register_model",
        requirements=[
            "boto3~=1.34",
            "mlflow[extras]~=2.10",
            "numpy>=1.26",
            "optuna>=3.5",
            "scikit-learn>=1.4",
            "xgboost>=2.0",
        ],
        system_site_packages=True,
    )
    def train_and_register_model():
        """
        Entrena y registra el modelo final usando los splits persistidos en MinIO.

        Pasos:
            1. Descarga X_train, X_test, y_train, y_test y feature_names.txt desde
               s3://data/final/ usando boto3, ya que los splits estan guardados en
               formato .npy binario y no como CSV.
            2. Ejecuta busqueda de hiperparametros con Optuna sobre XGBoost usando
               F1 promedio en validacion cruzada de 3 folds.
            3. Entrena el modelo final con los mejores parametros.
            4. Calibra probabilidades con Platt scaling (`CalibratedClassifierCV`).
            5. Evalua sobre test y loguea metricas/params en MLflow.
            6. Registra el modelo calibrado en MLflow con el mismo nombre que usa
               hoy `modelo/src/train.py`: `movielens-rating-classifier`.

        Por que leemos desde S3 en vez de disco local:
            Con CeleryExecutor la task puede correr en cualquier worker. MinIO es el
            storage compartido entre el DAG de ETL y el DAG de entrenamiento.

        Output: run en MLflow + modelo registrado como
                `movielens-rating-classifier`
        """
        import io
        import logging
        import os

        import boto3
        import mlflow
        import mlflow.sklearn
        import numpy as np
        import optuna
        import xgboost as xgb
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import cross_val_score

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        log = logging.getLogger(__name__)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        RANDOM_STATE = 42
        N_CORES = max(1, (os.cpu_count() or 2) - 1)
        N_TRIALS = 50
        endpoint_url = (
            os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("AWS_ENDPOINT_URL_S3")
            or "http://s3:9000"
        )
        os.environ["AWS_ENDPOINT_URL"] = endpoint_url


        from airflow.models import Variable
        DATA_BUCKET = Variable.get("DATA_BUCKET", default_var="data")
        FINAL_PREFIX = Variable.get("FINAL_PREFIX", default_var="final")
        MLFLOW_URI = Variable.get("MLFLOW_URI", default_var="http://mlflow:5000")
        EXPERIMENT_NAME = Variable.get("EXPERIMENT_NAME", default_var="movielens-rating-prediction")
        REGISTERED_MODEL_NAME = Variable.get("REGISTERED_MODEL_NAME", default_var="movielens-rating-classifier")

        s3_client = boto3.client("s3", endpoint_url=endpoint_url)

        def load_numpy_from_s3(bucket: str, key: str) -> np.ndarray:
            """
            Descarga un archivo .npy desde MinIO y lo deserializa con numpy.

            boto3 devuelve un stream binario; usamos BytesIO como buffer en RAM
            para que `np.load()` pueda leerlo como file-like object.
            """
            response = s3_client.get_object(Bucket=bucket, Key=key)
            buffer = io.BytesIO(response["Body"].read())
            buffer.seek(0)
            return np.load(buffer, allow_pickle=False)

        def load_text_lines_from_s3(bucket: str, key: str) -> list[str]:
            """Lee un archivo de texto desde MinIO y devuelve sus lineas."""
            try:
                response = s3_client.get_object(Bucket=bucket, Key=key)
            except s3_client.exceptions.NoSuchKey:
                return []
            body = response["Body"].read().decode("utf-8")
            return body.splitlines()

        def load_splits_from_s3() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """Carga los splits train/test persistidos por `etl_movielens`."""
            x_train = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/X_train.npy")
            x_test = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/X_test.npy")
            y_train = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/y_train.npy")
            y_test = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/y_test.npy")
            log.info("Splits cargados desde s3://%s/%s/", DATA_BUCKET, FINAL_PREFIX)
            log.info("  X_train=%s  X_test=%s", x_train.shape, x_test.shape)
            return x_train, x_test, y_train, y_test

        def optimize_hyperparams(
            x_train: np.ndarray,
            y_train: np.ndarray,
            n_trials: int,
        ) -> dict:
            """Busqueda de hiperparametros con Optuna (TPE sampler)."""

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.6, 1.0
                    ),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                }
                clf = xgb.XGBClassifier(
                    **params,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=N_CORES,
                )
                scores = cross_val_score(
                    clf,
                    x_train,
                    y_train,
                    cv=3,
                    scoring="f1",
                    n_jobs=N_CORES,
                )
                return scores.mean()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            log.info("Iniciando busqueda de hiperparametros (%d trials)...", n_trials)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            log.info("Mejor F1 (CV): %.4f | Params: %s", study.best_value, study.best_params)
            return study.best_params

        def compute_metrics(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_proba: np.ndarray,
        ) -> dict:
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "auc": roc_auc_score(y_true, y_proba),
            }

        x_train, x_test, y_train, y_test = load_splits_from_s3()
        feature_names = load_text_lines_from_s3(
            DATA_BUCKET,
            f"{FINAL_PREFIX}/feature_names.txt",
        )

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="xgboost-optuna"):
            best_params = optimize_hyperparams(x_train, y_train, N_TRIALS)
            mlflow.log_params(best_params)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("random_state", RANDOM_STATE)

            log.info("Entrenando modelo final con mejores parametros...")
            model = xgb.XGBClassifier(
                **best_params,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=N_CORES,
            )
            model.fit(x_train, y_train)

            log.info("Calibrando modelo (Platt scaling)...")
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calibrated.fit(x_train, y_train)

            y_pred = calibrated.predict(x_test)
            y_proba = calibrated.predict_proba(x_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)
            log.info("Metricas en test: %s", metrics)

            signature = mlflow.models.infer_signature(
                x_train,
                calibrated.predict(x_train),
            )
            mlflow.sklearn.log_model(
                calibrated,
                artifact_path="model",
                signature=signature,
                registered_model_name=REGISTERED_MODEL_NAME,
                input_example=x_train[:5],
            )

            if feature_names:
                mlflow.log_text("\n".join(feature_names), "feature_names.txt")

            log.info("Run completado. F1=%.4f  AUC=%.4f", metrics["f1"], metrics["auc"])

    train_and_register_model()


dag = train_movielens()
