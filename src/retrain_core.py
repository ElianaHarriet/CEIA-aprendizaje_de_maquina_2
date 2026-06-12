"""
retrain_core.py — Core logic for champion/challenger model retrain pipeline.

Functions are pure Python with no Airflow dependency, making them
testable in isolation and reusable across DAGs and scripts.

Dependencies: mlflow, boto3, numpy, scikit-learn
"""

import io
import logging
import os

import boto3
import mlflow
import mlflow.sklearn
import numpy as np

log = logging.getLogger(__name__)


def load_champion_model(model_name: str):
    """Load the current champion model from MLflow by 'champion' alias.

    Args:
        model_name: Name of the registered model in MLflow.

    Returns:
        The champion model object (scikit-learn estimator).

    Raises:
        RuntimeError: If no model with alias 'champion' exists.
    """
    client = mlflow.MlflowClient()
    try:
        model_data = client.get_model_version_by_alias(model_name, "champion")
    except Exception as e:
        raise RuntimeError(
            f"No champion model found for '{model_name}'. "
            "Run train_movielens DAG first."
        ) from e
    return mlflow.sklearn.load_model(model_data.source)


def load_splits_from_s3(data_bucket: str, final_prefix: str):
    """Load train/test .npy splits from S3/MinIO.

    Args:
        data_bucket: S3 bucket name (e.g., 'data').
        final_prefix: Prefix within bucket (e.g., 'final').

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    endpoint_url = (
        os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or "http://s3:9000"
    )
    os.environ["AWS_ENDPOINT_URL"] = endpoint_url

    s3_client = boto3.client("s3", endpoint_url=endpoint_url)

    keys = [
        f"{final_prefix}/X_train.npy",
        f"{final_prefix}/X_test.npy",
        f"{final_prefix}/y_train.npy",
        f"{final_prefix}/y_test.npy",
    ]

    result = []
    for key in keys:
        response = s3_client.get_object(Bucket=data_bucket, Key=key)
        buffer = io.BytesIO(response["Body"].read())
        buffer.seek(0)
        result.append(np.load(buffer, allow_pickle=False))

    log.info("Splits loaded from s3://%s/%s/", data_bucket, final_prefix)
    log.info("  X_train=%s  X_test=%s", result[0].shape, result[1].shape)
    return tuple(result)


def evaluate_and_promote(
    model_name: str,
    mlflow_uri: str,
    champion_f1: float,
    challenger_f1: float,
):
    """Compare champion vs challenger F1 and promote the winner via MLflow aliases.

    1. Logs both F1 scores and the winner to MLflow.
    2. If challenger wins: delete champion alias, delete challenger alias,
       set challenger version as new champion.
    3. If champion wins or tie: delete challenger alias only.

    Args:
        model_name: Registered model name in MLflow.
        mlflow_uri: MLflow tracking server URI.
        champion_f1: F1 score of the champion model on test set.
        challenger_f1: F1 score of the challenger model on test set.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("movielens-rating-prediction")

    client = mlflow.MlflowClient()

    winner = "champion"
    if challenger_f1 > champion_f1:
        winner = "challenger"
        log.info("Challenger wins (%.4f > %.4f). Promoting...", challenger_f1, champion_f1)
        challenger_data = client.get_model_version_by_alias(model_name, "challenger")
        challenger_version = challenger_data.version
        try:
            client.delete_registered_model_alias(model_name, "champion")
        except Exception:
            pass
        try:
            client.delete_registered_model_alias(model_name, "challenger")
        except Exception:
            pass
        client.set_registered_model_alias(model_name, "champion", challenger_version)
        log.info("Challenger version %s promoted to champion", challenger_version)
    else:
        log.info("Champion wins (%.4f >= %.4f). Discarding challenger...", champion_f1, challenger_f1)
        try:
            client.delete_registered_model_alias(model_name, "challenger")
        except Exception:
            pass

    with mlflow.start_run(run_name="champion-vs-challenger"):
        mlflow.log_metric("champion_f1", float(champion_f1))
        mlflow.log_metric("challenger_f1", float(challenger_f1))
        mlflow.log_param("winner", winner)
