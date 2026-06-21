"""
FastAPI model serving for MovieLens rating prediction.

Loads calibrated XGBoost model from MLflow and serves predictions via POST /predict.
"""

import os
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist

EXPECTED_FEATURE_COUNT = 50

FEATURE_NAMES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
    "Western", "(no genres listed)",
    "movie_avg_rating", "movie_rating_count_log", "movie_rating_std", "year",
    "genome_pca_0", "genome_pca_1", "genome_pca_2", "genome_pca_3",
    "genome_pca_4", "genome_pca_5", "genome_pca_6", "genome_pca_7",
    "genome_pca_8", "genome_pca_9", "genome_pca_10", "genome_pca_11",
    "genome_pca_12", "genome_pca_13", "genome_pca_14", "genome_pca_15",
    "genome_pca_16", "genome_pca_17", "genome_pca_18", "genome_pca_19",
    "user_avg_rating", "user_avg_rating_centered", "user_rating_count_log", "user_rating_std",
    "genre_cosine_similarity", "user_deviation_from_movie_avg",
]

MODEL = None


def load_model():
    """Load the calibrated model from MLflow model registry."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    model_uri = "models:/movielens-rating-classifier@champion"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_uri}: {e}")


def initialize_model():
    """Initialize the global model at startup."""
    global MODEL
    MODEL = load_model()


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    features: conlist(float, min_length=EXPECTED_FEATURE_COUNT, max_length=EXPECTED_FEATURE_COUNT) = Field(
        ..., description=f"Exactly {EXPECTED_FEATURE_COUNT} numerical features in training order"
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of rating >= 4.0")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (1 = rating >= 4.0)")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool
    model_uri: str = "models:/movielens-rating-classifier@champion"


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load model on startup and clean up on shutdown."""
    initialize_model()
    yield


app = FastAPI(
    title="MovieLens Rating Prediction API",
    description="Predicts whether a user will rate a movie >= 4 stars",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint verifying model is loaded."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict probability of user rating a movie >= 4 stars.
    
    Expects exactly 50 features in the order defined by FEATURE_NAMES.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_array = np.array(request.features, dtype=np.float32).reshape(1, -1)
        proba = MODEL.predict_proba(features_array)[0]
        pred = MODEL.predict(features_array)[0]
        
        return PredictResponse(
            probability=float(proba[1]),
            prediction=int(pred)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/")
async def root():
    return {"message": "MovieLens Rating Prediction API", "docs": "/docs", "health": "/health"}