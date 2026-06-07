"""
TDD Tests for FastAPI /predict endpoint.

Tests the model serving endpoint that loads from MLflow and returns predictions.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked MLflow model loading."""
        from dockerfiles.fastapi.app import app
        return TestClient(app)

    @pytest.fixture
    def mock_model(self):
        """Create a mock calibrated classifier that returns predictable outputs."""
        mock = MagicMock()
        # Return probability ~0.7 for class 1, ~0.3 for class 0
        mock.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock.predict.return_value = np.array([1])
        return mock

    @pytest.fixture(autouse=True)
    def setup_model(self, mock_model):
        """Patch MLflow model loading at module level."""
        with patch('dockerfiles.fastapi.app.load_model', return_value=mock_model):
            with patch('dockerfiles.fastapi.app.MODEL', mock_model):
                yield

    def test_predict_returns_probability_and_prediction(self, client):
        """POST /predict returns probability and binary prediction."""
        # 50 features as expected by the model
        features = [0.1] * 50
        response = client.post("/predict", json={"features": features})
        
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "prediction" in data
        assert isinstance(data["probability"], float)
        assert isinstance(data["prediction"], int)
        assert 0 <= data["probability"] <= 1
        assert data["prediction"] in (0, 1)

    def test_predict_with_valid_input_structure(self, client):
        """POST /predict accepts exactly 50 features."""
        features = [i * 0.01 for i in range(50)]
        response = client.post("/predict", json={"features": features})
        
        assert response.status_code == 200

    def test_predict_rejects_wrong_feature_count(self, client):
        """POST /predict rejects input with != 50 features."""
        response = client.post("/predict", json={"features": [0.1] * 49})
        assert response.status_code == 422
        
        response = client.post("/predict", json={"features": [0.1] * 51})
        assert response.status_code == 422

    def test_predict_rejects_non_numeric_features(self, client):
        """POST /predict rejects non-numeric feature values."""
        response = client.post("/predict", json={"features": ["a"] * 50})
        assert response.status_code == 422

    def test_predict_rejects_missing_features_key(self, client):
        """POST /predict rejects request without 'features' key."""
        response = client.post("/predict", json={"data": [0.1] * 50})
        assert response.status_code == 422

    def test_health_endpoint_exists(self, client):
        """GET /health returns 200 and model status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestModelLoading:
    """Tests for MLflow model loading at startup."""

    def test_load_model_from_mlflow(self):
        """load_model() loads from models:/movielens-rating-classifier/latest."""
        with patch('mlflow.sklearn.load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            from dockerfiles.fastapi.app import load_model
            result = load_model()
            
            mock_load.assert_called_once_with("models:/movielens-rating-classifier/latest")
            assert result is mock_model

    def test_load_model_raises_on_failure(self):
        """load_model() raises RuntimeError if MLflow load fails."""
        with patch('mlflow.sklearn.load_model', side_effect=Exception("MLflow down")):
            from dockerfiles.fastapi.app import load_model
            with pytest.raises(RuntimeError, match="Failed to load model"):
                load_model()


class TestFeatureNames:
    """Tests for feature name validation."""

    def test_expected_feature_count_is_50(self):
        """Model expects exactly 50 features."""
        from dockerfiles.fastapi.app import EXPECTED_FEATURE_COUNT
        assert EXPECTED_FEATURE_COUNT == 50

    def test_feature_names_match_training_order(self):
        """Feature names match the training pipeline order."""
        from dockerfiles.fastapi.app import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 50
        # First 20 are genres
        assert FEATURE_NAMES[0] == "Action"
        assert FEATURE_NAMES[19] == "(no genres listed)"
        # Movie features at indices 20-23
        assert FEATURE_NAMES[20] == "movie_avg_rating"
        assert FEATURE_NAMES[21] == "movie_rating_count_log"
        assert FEATURE_NAMES[22] == "movie_rating_std"
        assert FEATURE_NAMES[23] == "year"
        # Genome PCA at 24-43
        assert FEATURE_NAMES[24] == "genome_pca_0"
        assert FEATURE_NAMES[43] == "genome_pca_19"
        # User features at 44-47
        assert FEATURE_NAMES[44] == "user_avg_rating"
        assert FEATURE_NAMES[45] == "user_avg_rating_centered"
        assert FEATURE_NAMES[46] == "user_rating_count_log"
        assert FEATURE_NAMES[47] == "user_rating_std"
        # Interaction features at 48-49
        assert FEATURE_NAMES[48] == "genre_cosine_similarity"
        assert FEATURE_NAMES[49] == "user_deviation_from_movie_avg"