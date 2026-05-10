"""
predict.py — Inferencia sobre el modelo registrado en MLflow.

Carga el modelo desde MLflow (por nombre y versión/stage) y realiza
predicciones sobre un array de features o un archivo .npy.

Uso:
    python predict.py --input-file ./output/X_test.npy
                      [--model-name movielens-rating-classifier]
                      [--model-stage Production]
                      [--mlflow-uri http://localhost:5001]
                      [--threshold 0.5]
"""

import argparse
import logging
from pathlib import Path

import mlflow.sklearn
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_model(
    model_name: str,
    model_stage: str,
    mlflow_uri: str,
):
    """Carga el modelo desde el Model Registry de MLflow."""
    mlflow.set_tracking_uri(mlflow_uri)
    model_uri = f"models:/{model_name}/{model_stage}"
    log.info("Cargando modelo desde %s ...", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    log.info("Modelo cargado.")
    return model


def predict(
    model,
    X: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Realiza predicciones sobre X.

    Returns:
        labels: array binario (0/1).
        probas: probabilidad de clase positiva (rating >= 4.0).
    """
    probas = model.predict_proba(X)[:, 1]
    labels = (probas >= threshold).astype(int)
    return labels, probas


def run(
    input_file: str,
    model_name: str = "movielens-rating-classifier",
    model_stage: str = "Production",
    mlflow_uri: str = "http://localhost:5001",
    threshold: float = 0.5,
) -> None:
    X = np.load(input_file)
    log.info("Input cargado: shape=%s", X.shape)

    model = load_model(model_name, model_stage, mlflow_uri)
    labels, probas = predict(model, X, threshold)

    log.info("Predicciones completadas.")
    log.info(
        "  Positivos (rating >= 4): %d / %d (%.1f%%)",
        labels.sum(),
        len(labels),
        labels.mean() * 100,
    )

    # Guardar resultados junto al input
    out_dir = Path(input_file).parent
    np.save(out_dir / "predictions_labels.npy", labels)
    np.save(out_dir / "predictions_probas.npy", probas)
    log.info("Resultados guardados en %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia modelo MovieLens")
    parser.add_argument("--input-file", required=True, help="Ruta al archivo .npy con features")
    parser.add_argument("--model-name", default="movielens-rating-classifier")
    parser.add_argument("--model-stage", default="Production")
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run(
        input_file=args.input_file,
        model_name=args.model_name,
        model_stage=args.model_stage,
        mlflow_uri=args.mlflow_uri,
        threshold=args.threshold,
    )
