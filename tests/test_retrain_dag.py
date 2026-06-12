"""
Tests for retrain_movielens DAG — champion/challenger pipeline.

Tests the core logic functions that handle model loading, training,
F1 comparison, and MLflow alias promotion/demotion.
These are unit tests: all MLflow, boto3, sklearn calls are mocked.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrain_core import (
    evaluate_and_promote,
    load_champion_model,
    load_splits_from_s3,
)


class TestChampionLoading:
    """Tests for load_champion_model() — loading champion by alias from MLflow."""

    def test_load_champion_by_alias(self):
        """load_champion_model() loads model from MLflow using 'champion' alias."""
        with patch("mlflow.MlflowClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client

            mock_model_version = MagicMock()
            mock_model_version.source = "models:/movielens-rating-classifier/1"
            mock_client.get_model_version_by_alias.return_value = mock_model_version

            with patch("mlflow.sklearn.load_model") as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model

                result = load_champion_model("movielens-rating-classifier")

                mock_client.get_model_version_by_alias.assert_called_once_with(
                    "movielens-rating-classifier", "champion"
                )
                mock_load.assert_called_once_with(mock_model_version.source)
                assert result is mock_model

    def test_load_champion_raises_when_no_champion(self):
        """load_champion_model() raises RuntimeError when no champion alias exists."""
        with patch("mlflow.MlflowClient") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_client.get_model_version_by_alias.side_effect = Exception("No alias found")

            with pytest.raises(RuntimeError, match="No champion model found"):
                load_champion_model("movielens-rating-classifier")


class TestS3Loading:
    """Tests for load_splits_from_s3() — loading .npy files from MinIO."""

    def test_load_splits_returns_four_arrays(self):
        """load_splits_from_s3() returns 4 numpy arrays."""
        mock_s3 = MagicMock()

        def mock_get_object(Bucket, Key):
            import io
            buffer = io.BytesIO()
            arr = np.array([1.0, 2.0, 3.0])
            np.save(buffer, arr)
            buffer.seek(0)
            mock_response = MagicMock()
            mock_response["Body"].read.return_value = buffer.getvalue()
            return mock_response

        mock_s3.get_object.side_effect = mock_get_object

        with patch("boto3.client", return_value=mock_s3):
            result = load_splits_from_s3("data", "final")

            assert len(result) == 4
            for arr in result:
                assert isinstance(arr, np.ndarray)

    def test_load_splits_defaults_endpoint(self):
        """load_splits_from_s3() uses default endpoint when no env var is set."""
        import os
        saved = os.environ.pop("AWS_ENDPOINT_URL", None)
        saved_s3 = os.environ.pop("AWS_ENDPOINT_URL_S3", None)
        try:
            mock_s3 = MagicMock()
            mock_s3.get_object.side_effect = Exception("connection")
            with patch("boto3.client") as mock_boto:
                mock_boto.return_value = mock_s3
                with pytest.raises(Exception):
                    load_splits_from_s3("data", "final")
                mock_boto.assert_called_once_with("s3", endpoint_url="http://s3:9000")
        finally:
            if saved:
                os.environ["AWS_ENDPOINT_URL"] = saved
            if saved_s3:
                os.environ["AWS_ENDPOINT_URL_S3"] = saved_s3


class TestComparisonLogic:
    """Tests for evaluate_and_promote() — F1 comparison and alias management."""

    @pytest.fixture
    def mlflow_patches(self):
        with patch("mlflow.MlflowClient") as MockClient, \
             patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run") as mock_start_run, \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_param"):
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            mock_run = MagicMock()
            mock_run.info.run_id = "test-run-id"
            mock_start_run.return_value.__enter__.return_value = mock_run
            yield mock_client

    def test_promote_challenger_when_better(self, mlflow_patches):
        """evaluate_and_promote() promotes challenger when its F1 is higher."""
        mock_client = mlflow_patches
        mock_challenger_data = MagicMock()
        mock_challenger_data.version = "2"
        mock_client.get_model_version_by_alias.return_value = mock_challenger_data

        evaluate_and_promote(
            model_name="movielens-rating-classifier",
            mlflow_uri="http://mlflow:5000",
            champion_f1=0.70,
            challenger_f1=0.75,
        )

        mock_client.delete_registered_model_alias.assert_any_call(
            "movielens-rating-classifier", "champion"
        )
        mock_client.delete_registered_model_alias.assert_any_call(
            "movielens-rating-classifier", "challenger"
        )
        mock_client.set_registered_model_alias.assert_called_once_with(
            "movielens-rating-classifier", "champion", "2"
        )

    def test_demote_challenger_when_worse(self, mlflow_patches):
        """evaluate_and_promote() demotes challenger when champion's F1 is higher or equal."""
        mock_client = mlflow_patches

        evaluate_and_promote(
            model_name="movielens-rating-classifier",
            mlflow_uri="http://mlflow:5000",
            champion_f1=0.75,
            challenger_f1=0.70,
        )

        mock_client.delete_registered_model_alias.assert_called_once_with(
            "movielens-rating-classifier", "challenger"
        )
        mock_client.set_registered_model_alias.assert_not_called()
