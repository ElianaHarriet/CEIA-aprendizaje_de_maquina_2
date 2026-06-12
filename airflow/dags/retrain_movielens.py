"""
retrain_movielens.py - DAG de reentreno: challenger vs champion.

Carga el modelo champion desde MLflow, entrena un challenger con Optuna,
compara F1 en test set, y promueve al ganador via aliases de MLflow.

Flujo:
    train_challenger()
        └─ Registra nuevo modelo con alias "challenger" en MLflow
                v
    evaluate_and_promote()
        ├─ Carga champion y challenger desde MLflow
        ├─ Calcula F1 en test para ambos
        └─ Promueve challenger si supera a champion
"""

import datetime

from airflow.decorators import dag, task

markdown_text = """
### Retrain Pipeline - MovieLens (Challenger vs Champion)

Re-entrena el clasificador de MovieLens usando los splits de `etl_movielens`,
compara su F1 contra el modelo champion actual, y promueve al mejor.

**Etapas:**
1. `train_challenger` - Clona config del champion, Optuna rápida, entrena y
   registra con alias `challenger` en MLflow.
2. `evaluate_and_promote` - Compara F1 en test set. Si challenger supera
   a champion, promueve challenger a `champion`; si no, descarta challenger.
"""

default_args = {
    "owner": "CEIA - FIUBA",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="retrain_movielens",
    description="Re-entrena el modelo MovieLens y promueve si supera al champion.",
    doc_md=markdown_text,
    tags=["retrain", "MovieLens", "MLflow"],
    default_args=default_args,
    schedule=None,
    dagrun_timeout=datetime.timedelta(minutes=180),
    catchup=False,
)
def retrain_movielens():

    @task.virtualenv(
        task_id="train_challenger",
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
    def train_challenger():
        """
        Entrena un modelo challenger y lo registra en MLflow con alias 'challenger'.

        Pasos:
            1. Carga el champion actual desde MLflow por alias.
            2. Carga los splits train/test desde MinIO.
            3. Usa los hiperparámetros del champion como punto de partida y
               ejecuta Optuna (5 trials) para refinarlos.
            4. Entrena XGBoost final + Platt calibration.
            5. Calcula F1 en test.
            6. Registra en MLflow con alias 'challenger'.
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
        from sklearn.metrics import f1_score
        from sklearn.model_selection import cross_val_score

        from airflow.models import Variable

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        log = logging.getLogger(__name__)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        RANDOM_STATE = 42
        N_TRIALS = 5
        endpoint_url = (
            os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("AWS_ENDPOINT_URL_S3")
            or "http://s3:9000"
        )
        os.environ["AWS_ENDPOINT_URL"] = endpoint_url

        DATA_BUCKET = Variable.get("DATA_BUCKET", default_var="data")
        FINAL_PREFIX = Variable.get("FINAL_PREFIX", default_var="final")
        MLFLOW_URI = Variable.get("MLFLOW_URI", default_var="http://mlflow:5000")
        EXPERIMENT_NAME = Variable.get("EXPERIMENT_NAME", default_var="movielens-rating-prediction")
        REGISTERED_MODEL_NAME = Variable.get("REGISTERED_MODEL_NAME", default_var="movielens-rating-classifier")

        s3_client = boto3.client("s3", endpoint_url=endpoint_url)

        def load_numpy_from_s3(bucket: str, key: str) -> np.ndarray:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            buffer = io.BytesIO(response["Body"].read())
            buffer.seek(0)
            return np.load(buffer, allow_pickle=False)

        def load_splits():
            x_train = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/X_train.npy")
            x_test = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/X_test.npy")
            y_train = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/y_train.npy")
            y_test = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/y_test.npy")
            return x_train, x_test, y_train, y_test

        def load_champion():
            client = mlflow.MlflowClient()
            try:
                model_data = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
            except Exception as e:
                raise RuntimeError(
                    f"No champion model found for '{REGISTERED_MODEL_NAME}'. "
                    "Run train_movielens DAG first."
                ) from e
            return mlflow.sklearn.load_model(model_data.source)

        def optimize_params(x_train, y_train, champion_params):
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        max(100, champion_params.get("n_estimators", 300) - 100),
                        champion_params.get("n_estimators", 300) + 100,
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        max(3, champion_params.get("max_depth", 6) - 2),
                        champion_params.get("max_depth", 6) + 2,
                    ),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
                }
                clf = xgb.XGBClassifier(
                    **params,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=1,
                )
                scores = cross_val_score(clf, x_train, y_train, cv=3, scoring="f1", n_jobs=1)
                return scores.mean()

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            )
            log.info("Iniciando Optuna para challenger (%d trials)...", N_TRIALS)
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
            log.info("Mejor F1 (CV): %.4f", study.best_value)
            return study.best_params

        x_train, x_test, y_train, y_test = load_splits()
        log.info("Splits cargados: X_train=%s, X_test=%s", x_train.shape, x_test.shape)

        champion = load_champion()
        champion_params = champion.get_params()
        log.info("Champion cargado con %d parametros", len(champion_params))

        best_params = optimize_params(x_train, y_train, champion_params)
        log.info("Mejores params challenger: %s", best_params)

        model = xgb.XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=1,
        )
        model.fit(x_train, y_train)

        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated.fit(x_train, y_train)

        y_pred = calibrated.predict(x_test)
        challenger_f1 = f1_score(y_test, y_pred)
        log.info("Challenger F1 en test: %.4f", challenger_f1)

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="challenger-optuna"):
            mlflow.log_params(best_params)
            mlflow.log_metric("test_f1", challenger_f1)
            mlflow.log_param("n_trials", N_TRIALS)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("source", "retrain_challenger")

            signature = mlflow.models.infer_signature(x_train, calibrated.predict(x_train))
            mlflow.sklearn.log_model(
                calibrated,
                artifact_path="model",
                signature=signature,
                registered_model_name=REGISTERED_MODEL_NAME,
                input_example=x_train[:5],
            )

            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                REGISTERED_MODEL_NAME,
            )

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", result.version)
        log.info("Challenger version %s registrada con alias 'challenger'", result.version)

    @task.virtualenv(
        task_id="evaluate_and_promote",
        requirements=[
            "boto3~=1.34",
            "mlflow[extras]~=2.10",
            "numpy>=1.26",
            "scikit-learn>=1.4",
        ],
        system_site_packages=True,
    )
    def evaluate_and_promote():
        """
        Compara F1 del champion vs challenger y promueve al mejor.

        Carga champion y challenger desde MLflow por alias, descarga el test
        set de MinIO, calcula F1 para ambos, loguea metricas comparativas,
        y actualiza los alias en MLflow.
        """
        import io
        import logging
        import os

        import boto3
        import mlflow
        import mlflow.sklearn
        import numpy as np
        from sklearn.metrics import f1_score

        from airflow.models import Variable

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        log = logging.getLogger(__name__)

        endpoint_url = (
            os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("AWS_ENDPOINT_URL_S3")
            or "http://s3:9000"
        )
        os.environ["AWS_ENDPOINT_URL"] = endpoint_url

        DATA_BUCKET = Variable.get("DATA_BUCKET", default_var="data")
        FINAL_PREFIX = Variable.get("FINAL_PREFIX", default_var="final")
        MLFLOW_URI = Variable.get("MLFLOW_URI", default_var="http://mlflow:5000")
        EXPERIMENT_NAME = Variable.get("EXPERIMENT_NAME", default_var="movielens-rating-prediction")
        REGISTERED_MODEL_NAME = Variable.get("REGISTERED_MODEL_NAME", default_var="movielens-rating-classifier")

        s3_client = boto3.client("s3", endpoint_url=endpoint_url)

        def load_numpy_from_s3(bucket: str, key: str) -> np.ndarray:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            buffer = io.BytesIO(response["Body"].read())
            buffer.seek(0)
            return np.load(buffer, allow_pickle=False)

        def load_model_by_alias(alias: str):
            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
            return mlflow.sklearn.load_model(model_data.source)

        # Cargar test set
        x_test = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/X_test.npy")
        y_test = load_numpy_from_s3(DATA_BUCKET, f"{FINAL_PREFIX}/y_test.npy")
        log.info("Test set cargado: X_test=%s, y_test=%s", x_test.shape, y_test.shape)

        # Cargar champion y calcular F1
        champion_f1 = 0.0
        try:
            champion_model = load_model_by_alias("champion")
            champion_pred = champion_model.predict(x_test)
            champion_f1 = f1_score(y_test, champion_pred)
            log.info("Champion F1: %.4f", champion_f1)
        except Exception as e:
            log.warning("No se pudo cargar champion: %s", e)

        # Cargar challenger y calcular F1
        challenger_f1 = 0.0
        try:
            challenger_model = load_model_by_alias("challenger")
            challenger_pred = challenger_model.predict(x_test)
            challenger_f1 = f1_score(y_test, challenger_pred)
            log.info("Challenger F1: %.4f", challenger_f1)
        except Exception as e:
            log.warning("No se pudo cargar challenger: %s", e)

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        client = mlflow.MlflowClient()

        winner = "champion"
        if challenger_f1 > champion_f1:
            winner = "challenger"
            log.info("Challenger gana (%.4f > %.4f). Promoviendo...", challenger_f1, champion_f1)
            challenger_data = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "challenger")
            challenger_version = challenger_data.version
            try:
                client.delete_registered_model_alias(REGISTERED_MODEL_NAME, "champion")
            except Exception:
                pass
            try:
                client.delete_registered_model_alias(REGISTERED_MODEL_NAME, "challenger")
            except Exception:
                pass
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", challenger_version)
            log.info("Challenger version %s promovido a champion", challenger_version)
        else:
            log.info("Champion gana (%.4f >= %.4f). Descartando challenger...", champion_f1, challenger_f1)
            try:
                client.delete_registered_model_alias(REGISTERED_MODEL_NAME, "challenger")
            except Exception:
                pass

        with mlflow.start_run(run_name="champion-vs-challenger"):
            mlflow.log_metric("champion_f1", float(champion_f1))
            mlflow.log_metric("challenger_f1", float(challenger_f1))
            mlflow.log_param("winner", winner)

    train_challenger() >> evaluate_and_promote()


dag = retrain_movielens()
