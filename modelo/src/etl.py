"""
etl.py — Descarga y preprocesamiento del dataset MovieLens 25M.

Descarga el dataset si no existe, construye la matriz de features usando
MovieLensDataset y guarda los splits train/test listos para el entrenamiento.

Uso:
    python etl.py [--data-path ./data/ml-25m] [--output-path ./output]
                  [--n-users 20000] [--n-ratings 1000000]
"""

import argparse
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from movielens_data import MovieLensDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
REQUIRED_FILES = [
    "ratings.csv",
    "movies.csv",
    "genome-scores.csv",
    "genome-tags.csv",
    "links.csv",
    "tags.csv",
]


def download_dataset(data_path: Path) -> None:
    """Descarga y extrae el dataset MovieLens 25M si no existe."""
    if data_path.exists() and all((data_path / f).exists() for f in REQUIRED_FILES):
        log.info("Dataset ya presente en %s", data_path)
        return

    log.info("Descargando MovieLens 25M desde %s ...", DATA_URL)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = data_path.parent / "ml-25m.zip"

    urllib.request.urlretrieve(DATA_URL, zip_path)
    log.info("Extrayendo zip...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path.parent)

    # grouplens extrae en ml-25m/ — movemos los archivos a data_path
    extracted = data_path.parent / "ml-25m"
    if extracted.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        for f in extracted.iterdir():
            shutil.move(str(f), str(data_path / f.name))
        extracted.rmdir()

    zip_path.unlink()
    log.info("Dataset disponible en %s", data_path)


def build_features(
    data_path: Path,
    n_users: int,
    n_ratings: int,
) -> MovieLensDataset:
    """Construye el dataset con features usando MovieLensDataset."""
    log.info("Construyendo features (n_users=%d, n_ratings=%d)...", n_users, n_ratings)
    dataset = MovieLensDataset(
        data_path=str(data_path),
        n_users=n_users,
        n_ratings=n_ratings,
    )
    log.info(
        "Features listas: X_train=%s  X_test=%s",
        dataset.data_tuple[0].shape,
        dataset.data_tuple[1].shape,
    )
    return dataset


def save_splits(dataset: MovieLensDataset, output_path: Path) -> None:
    """Guarda los splits train/test en disco como archivos .npy."""
    output_path.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = dataset.data_tuple

    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "y_test.npy", y_test)

    feature_names = dataset.feature_names
    with open(output_path / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    log.info("Splits guardados en %s", output_path)


def run(
    data_path: str = "./data",
    output_path: str = "./output",
    n_users: int = 20_000,
    n_ratings: int = 1_000_000,
) -> None:
    data_dir = Path(data_path)
    out_dir = Path(output_path)

    download_dataset(data_dir)
    dataset = build_features(data_dir, n_users, n_ratings)
    save_splits(dataset, out_dir)
    log.info("ETL completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL MovieLens 25M")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--output-path", default="./output")
    parser.add_argument("--n-users", type=int, default=20_000)
    parser.add_argument("--n-ratings", type=int, default=1_000_000)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        output_path=args.output_path,
        n_users=args.n_users,
        n_ratings=args.n_ratings,
    )
