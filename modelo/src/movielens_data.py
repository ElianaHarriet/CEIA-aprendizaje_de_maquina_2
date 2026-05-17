"""
movielens_data.py - Feature engineering y utils para el dataset MovieLens 25M.

Reconstruido a partir del TP final de Aprendizaje de Máquina I (CEIA - FIUBA).
Autores originales: Jose Miguel Silva Pavón, Pablo Santiago Rodríguez Castro,
                    Damian Nicolas Smilovich.

Features construidas (50 en total):
    0-19  : géneros binarios (GENRES)
    20    : movie_avg_rating
    21    : movie_rating_count_log
    22    : movie_rating_std
    23    : year
    24-43 : genome_pca_0 … genome_pca_19
    44    : user_avg_rating
    45    : user_avg_rating_centered
    46    : user_rating_count_log
    47    : user_rating_std
    48    : genre_cosine_similarity
    49    : user_deviation_from_movie_avg
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_GENOME_COMPONENTS = 20

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)",
]


class MovieLensDataset:
    """Carga, procesa y expone los splits train/test del dataset MovieLens 25M."""

    GENRES = GENRES

    def __init__(
        self,
        data_path: str = "./data",
        n_users: int = 20_000,
        n_ratings: int = 1_000_000,
        test_size: float = 0.2,
        n_genome_components: int = N_GENOME_COMPONENTS,
        random_state: int = RANDOM_STATE,
    ):
        self.data_path = Path(data_path)
        self.n_users = n_users
        self.n_ratings = n_ratings
        self.test_size = test_size
        self.n_genome_components = n_genome_components
        self.random_state = random_state

        self._build()

    # ------------------------------------------------------------------
    # Construcción interna
    # ------------------------------------------------------------------

    def _build(self) -> None:
        ratings, movies, genome_scores = self._load_csvs()
        ratings = self._sample(ratings)
        movie_feats = self._movie_features(ratings, movies)
        genome_pca_df = self._genome_pca(genome_scores)
        user_feats = self._user_features(ratings)
        df = self._merge(ratings, movie_feats, genome_pca_df, user_feats)
        df = self._interaction_features(df)
        self._finalize(df)

    def _load_csvs(self):
        ratings = pd.read_csv(self.data_path / "ratings.csv")
        movies = pd.read_csv(self.data_path / "movies.csv")
        genome_scores = pd.read_csv(self.data_path / "genome-scores.csv")
        return ratings, movies, genome_scores

    def _sample(self, ratings: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        all_users = ratings["userId"].unique()
        sampled = rng.choice(all_users, size=min(self.n_users, len(all_users)), replace=False)
        ratings = ratings[ratings["userId"].isin(sampled)]
        if len(ratings) > self.n_ratings:
            ratings = ratings.sample(n=self.n_ratings, random_state=self.random_state)
        return ratings.reset_index(drop=True)

    def _movie_features(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(
                movie_avg_rating="mean",
                movie_rating_count="count",
                movie_rating_std="std",
            )
            .fillna(0)
        )
        stats["movie_rating_count_log"] = np.log1p(stats["movie_rating_count"])

        movies = movies.copy()
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
        median_year = movies["year"].median()
        movies["year"] = movies["year"].fillna(median_year)

        for genre in GENRES:
            movies[genre] = movies["genres"].str.contains(genre, regex=False).astype(int)

        movie_feats = stats[
            ["movie_avg_rating", "movie_rating_count_log", "movie_rating_std"]
        ].join(movies.set_index("movieId")[["year"] + GENRES], how="left")
        return movie_feats

    def _genome_pca(self, genome_scores: pd.DataFrame) -> pd.DataFrame:
        pivot = genome_scores.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)
        pca = PCA(n_components=self.n_genome_components, random_state=self.random_state)
        components = pca.fit_transform(pivot.values)
        self.genome_pca = pca
        self.genome_explained_variance = pca.explained_variance_ratio_

        cols = [f"genome_pca_{i}" for i in range(self.n_genome_components)]
        return pd.DataFrame(components, index=pivot.index, columns=cols)

    def _user_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        global_mean = ratings["rating"].mean()
        stats = (
            ratings.groupby("userId")["rating"]
            .agg(
                user_avg_rating="mean",
                user_rating_count="count",
                user_rating_std="std",
            )
            .fillna(0)
        )
        stats["user_rating_count_log"] = np.log1p(stats["user_rating_count"])
        stats["user_avg_rating_centered"] = stats["user_avg_rating"] - global_mean
        return stats[
            [
                "user_avg_rating",
                "user_avg_rating_centered",
                "user_rating_count_log",
                "user_rating_std",
            ]
        ]

    def _merge(
        self,
        ratings: pd.DataFrame,
        movie_feats: pd.DataFrame,
        genome_pca_df: pd.DataFrame,
        user_feats: pd.DataFrame,
    ) -> pd.DataFrame:
        df = ratings[["userId", "movieId", "rating"]].copy()
        df = df.merge(movie_feats, on="movieId", how="left")
        df = df.merge(genome_pca_df, on="movieId", how="left")
        df = df.merge(user_feats, on="userId", how="left")
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["user_deviation_from_movie_avg"] = df["user_avg_rating"] - df["movie_avg_rating"]

        # Cosine similarity entre vector de géneros de la película y
        # preferencia de género promedio del usuario
        user_genre_pref = df.groupby("userId")[GENRES].mean()
        pref_cols = {g: f"user_pref_{g}" for g in GENRES}
        df = df.merge(user_genre_pref.rename(columns=pref_cols), on="userId", how="left")

        movie_vecs = df[GENRES].values.astype(float)
        user_vecs = df[list(pref_cols.values())].values.astype(float)
        dot = (movie_vecs * user_vecs).sum(axis=1)
        norm_m = np.linalg.norm(movie_vecs, axis=1)
        norm_u = np.linalg.norm(user_vecs, axis=1)
        denom = norm_m * norm_u
        df["genre_cosine_similarity"] = np.where(denom > 0, dot / denom, 0.0)

        return df

    def _finalize(self, df: pd.DataFrame) -> None:
        genome_cols = [f"genome_pca_{i}" for i in range(self.n_genome_components)]
        # Orden de features que reproduce los índices del notebook original:
        # 0-19: GENRES | 20: movie_avg_rating | 21-23: count_log, std, year
        # 24-43: genome_pca | 44: user_avg_rating | 45-47: centered, count_log, std
        # 48: genre_cosine_similarity | 49: user_deviation_from_movie_avg
        feature_cols = (
            GENRES
            + ["movie_avg_rating", "movie_rating_count_log", "movie_rating_std", "year"]
            + genome_cols
            + [
                "user_avg_rating",
                "user_avg_rating_centered",
                "user_rating_count_log",
                "user_rating_std",
            ]
            + ["genre_cosine_similarity", "user_deviation_from_movie_avg"]
        )

        df["target"] = (df["rating"] >= 4.0).astype(int)
        df = df.dropna(subset=feature_cols + ["target"])

        self.df_features = df[feature_cols + ["rating", "target"]].copy()
        self.feature_names = feature_cols

        X = df[feature_cols].values.astype(float)
        y = df["target"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.scaler = scaler

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        self._data_tuple = (X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------
    # Propiedades públicas
    # ------------------------------------------------------------------

    @property
    def data_tuple(self) -> tuple:
        """(X_train, X_test, y_train, y_test) como arrays numpy."""
        return self._data_tuple

    def get_subset(self, n: int = 30_000) -> tuple:
        """Devuelve un subconjunto aleatorio de train/test (para modelos costosos)."""
        X_train, X_test, y_train, y_test = self._data_tuple
        rng = np.random.default_rng(self.random_state)
        n_train = min(n, len(X_train))
        n_test = min(n // 4, len(X_test))
        idx_tr = rng.choice(len(X_train), size=n_train, replace=False)
        idx_te = rng.choice(len(X_test), size=n_test, replace=False)
        return (X_train[idx_tr], X_test[idx_te], y_train[idx_tr], y_test[idx_te])


# ------------------------------------------------------------------
# Funciones de utilidad
# ------------------------------------------------------------------


def evaluate_classifier(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Calcula métricas estándar de clasificación binaria."""
    return {
        "modelo": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc_score": roc_auc_score(y_true, y_proba),
    }


def train_test_generic(name: str, model, data_tuple: tuple) -> tuple:
    """Entrena un modelo y devuelve (modelo_entrenado, métricas)."""
    X_train, X_test, y_train, y_test = data_tuple
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )
    metrics = evaluate_classifier(name, y_test, y_pred, y_proba)
    return model, metrics


def obtain_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Umbral óptimo por Youden's J statistic (maximiza TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    best_idx = np.argmax(tpr - fpr)
    return float(thresholds[best_idx])
