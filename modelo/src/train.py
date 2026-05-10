"""
train.py — Entrenamiento del modelo XGBoost con búsqueda de hiperparámetros
           via Optuna y tracking en MLflow.

Lee los splits generados por etl.py, ejecuta la búsqueda de hiperparámetros,
entrena el modelo final, lo registra en MLflow y guarda el artefacto.

Uso:
    python train.py [--input-path ./output] [--n-trials 50]
                    [--experiment-name movielens-rating-prediction]
                    [--mlflow-uri http://localhost:5001]
"""

import argparse
import logging
import os
from pathlib import Path

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


def load_splits(input_path: Path) -> tuple:
    """Carga los splits train/test generados por etl.py."""
    X_train = np.load(input_path / "X_train.npy")
    X_test = np.load(input_path / "X_test.npy")
    y_train = np.load(input_path / "y_train.npy")
    y_test = np.load(input_path / "y_test.npy")
    log.info("Splits cargados: X_train=%s  X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def load_feature_names(input_path: Path) -> list[str]:
    names_file = input_path / "feature_names.txt"
    if names_file.exists():
        return names_file.read_text().splitlines()
    return []


def optimize_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int,
) -> dict:
    """Búsqueda de hiperparámetros con Optuna (TPE sampler)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
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
            n_jobs=N_CORES,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1", n_jobs=N_CORES)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    log.info("Iniciando búsqueda de hiperparámetros (%d trials)...", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info("Mejor F1 (CV): %.4f | Params: %s", study.best_value, study.best_params)
    return study.best_params


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
    }


def train_and_log(
    input_path: str = "./output",
    n_trials: int = 50,
    experiment_name: str = "movielens-rating-prediction",
    mlflow_uri: str = "http://localhost:5001",
) -> None:
    in_dir = Path(input_path)
    X_train, X_test, y_train, y_test = load_splits(in_dir)
    feature_names = load_feature_names(in_dir)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost-optuna"):
        # — Búsqueda de hiperparámetros —
        best_params = optimize_hyperparams(X_train, y_train, n_trials)
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("random_state", RANDOM_STATE)

        # — Entrenamiento del modelo final —
        log.info("Entrenando modelo final con mejores parámetros...")
        model = xgb.XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=N_CORES,
        )
        model.fit(X_train, y_train)

        # — Calibración con Platt scaling —
        log.info("Calibrando modelo (Platt scaling)...")
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)

        # — Evaluación —
        y_pred = calibrated.predict(X_test)
        y_proba = calibrated.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        log.info("Métricas en test: %s", metrics)

        # — Registro del modelo —
        signature = mlflow.models.infer_signature(X_train, calibrated.predict(X_train))
        mlflow.sklearn.log_model(
            calibrated,
            artifact_path="model",
            signature=signature,
            registered_model_name="movielens-rating-classifier",
            input_example=X_train[:5],
        )

        if feature_names:
            mlflow.log_text("\n".join(feature_names), "feature_names.txt")

        log.info("Run completado. F1=%.4f  AUC=%.4f", metrics["f1"], metrics["auc"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento XGBoost con MLflow")
    parser.add_argument("--input-path", default="./output")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--experiment-name", default="movielens-rating-prediction")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    args = parser.parse_args()

    train_and_log(
        input_path=args.input_path,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri,
    )
