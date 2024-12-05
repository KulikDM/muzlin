import importlib
import logging
import os
import importlib.util
from unittest.mock import patch

import mlflow
import numpy as np
import pytest
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pythresh.thresholds.iqr import IQR
from pythresh.thresholds.clf import CLF
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import muzlin.anomaly.detector as det
from muzlin.anomaly import OutlierDetector


logging.basicConfig(level=logging.ERROR)

mlflow_logger = logging.getLogger('mlflow')
mlflow_logger.setLevel(logging.ERROR)


@pytest.fixture
def sample_data():
    """Generates a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.normal(loc=0, scale=1, size=(1000, 384))
    y = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    return X, y

@pytest.fixture
def outlier_detector():
    """Fixture to create OutlierDetector instances with custom parameters."""
    def _create_outlier_detector(**kwargs):
        return OutlierDetector(**kwargs)
    return _create_outlier_detector

def pipeline_checks(detector, X, y):
    """Standard checks for a fitted OutlierDetector."""
    assert hasattr(detector, 'threshold_')
    assert hasattr(detector, 'labels_')
    assert hasattr(detector.pipeline.named_steps['detector'], 'decision_scores_')
    assert hasattr(detector.pipeline.named_steps['detector'], 'Xstd_')

    assert len(detector.pipeline.named_steps['detector'].decision_scores_) == len(y)
    assert len(detector.labels_) == len(y)
    assert np.all(np.isin(detector.labels_, [0, 1]))
    assert detector.pipeline.named_steps['detector'].Xstd_ == pytest.approx(np.std(X))


class TestOutlierDetector:
    def test_initialization_with_default_detector(self, outlier_detector):
        """Test initialization without a custom detector."""
        detector = outlier_detector()
        assert isinstance(detector, BaseEstimator)
        assert detector.pipeline.named_steps['detector'] is not None
        assert isinstance(detector.pipeline, Pipeline)

    def test_fit_with_default_detector(self, outlier_detector, sample_data):
        """Test the `fit` method using default detector."""
        X, y = sample_data
        detector = outlier_detector()
        detector.fit(X)

        pipeline_checks(detector, X, y)
        assert detector.threshold_ is None

    @pytest.mark.parametrize("method", [KNN(), IForest(), LogisticRegression()])
    def test_fit_with_detector(self, outlier_detector, sample_data, method):
        """Test fitting with different outlier detector methods"""
        X, y = sample_data
        detector = outlier_detector(detector=method)
        detector.fit(X, y)

        pipeline_checks(detector, X, y)
        assert detector.detector == method

    @pytest.mark.parametrize("contamination", [0.02, 0.15, 85, 110, IQR(), CLF()])
    def test_fit_with_contamination(self, outlier_detector, sample_data, contamination):
        """Test fitting with different contamination settings"""
        X, y = sample_data
        detector = outlier_detector(contamination=contamination)
        detector.fit(X)

        pipeline_checks(detector, X, y)
        assert detector.contamination == contamination
        assert detector.threshold_ is not None
        assert detector.threshold_ > 0

    def test_fit_with_supervised_labels(self, outlier_detector, sample_data):
        """Test fitting with supervised binary labels."""
        X, y = sample_data
        method = LogisticRegression()
        detector = outlier_detector(detector=method)
        detector.fit(X, y=y)

        pipeline_checks(detector, X, y)
        assert detector.detector == method
        assert detector.pipeline.threshold_ is None

    @pytest.mark.parametrize("contamination", [None, 0.1])
    def test_predict_function(self, outlier_detector, sample_data, contamination):
        """Test the `predict` function."""
        X, _ = sample_data
        detector = outlier_detector(contamination=contamination)
        detector.fit(X)
        labels = detector.predict(X)
        
        assert detector.contamination == contamination
        assert labels.shape[0] == X.shape[0]
        assert set(labels).issubset({0, 1})

    def test_decision_function(self, outlier_detector, sample_data):
        """Test the `decision_function` method."""
        X, _ = sample_data
        detector = outlier_detector()
        detector.fit(X)
        decision_scores = detector.decision_function(X)
        assert decision_scores.shape[0] == X.shape[0]
        assert isinstance(decision_scores, np.ndarray)

    def test_invalid_labels_error(self, outlier_detector, sample_data):
        """Test error handling for invalid labels."""
        X, y = sample_data
        y_invalid = np.random.randint(2, 5, size=y.shape[0])
        with pytest.raises(ValueError, match="y should only contain binary values 0 or 1."):
            detector = outlier_detector()
            detector.fit(X, y=y_invalid)

    def test_model_loading_and_reinitialization(self, tmp_path, outlier_detector, sample_data):
        """Test saving, loading, and reinitializing the model."""
        
        model_path = tmp_path / "outlier_detector.pkl"
        X, y = sample_data

        detector = outlier_detector(mlflow=False, model=str(model_path))
        detector.fit(X)
        labels = detector.predict(X)
        
        # Check the saved file exists
        assert model_path.exists()

        # Reinitialize and load the model
        new_detector = OutlierDetector(model=str(model_path))
        pipeline_checks(detector, X, y)

        new_labels = new_detector.predict(X)

        decision_scores = detector.pipeline.named_steps['detector'].decision_scores_
        assert np.allclose(decision_scores, new_detector.decision_scores_)
        assert np.allclose(labels, new_labels)

    def test_mlflow_not_installed(self, sample_data):
        """Test behavior when MLflow is not installed."""

        X, _ = sample_data

        original_find_spec = importlib.util.find_spec
        
        def custom_find_spec(name):
            if name == 'mlflow':
                return None
            return original_find_spec(name)

        with patch("importlib.util.find_spec", side_effect=custom_find_spec):
            importlib.reload(det)
            detector = OutlierDetector(mlflow=True)
            detector.fit(X)
            assert not detector.mlflow

        # Reload the module to restore original state
        importlib.reload(det)
