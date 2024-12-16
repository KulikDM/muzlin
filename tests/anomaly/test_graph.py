import importlib
import importlib.util
import logging
from unittest.mock import patch

import mlflow
import networkx as nx
import numpy as np
import pytest
from pygod.detector import ANOMALOUS, GUIDE, AnomalyDAE
from pythresh.thresholds.clf import CLF
from pythresh.thresholds.iqr import IQR
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import muzlin.anomaly.graph as det
from muzlin.anomaly import GraphOutlierDetector

_ = mlflow.config.get_registry_uri()
logging.basicConfig(level=logging.ERROR)

mlflow_logger = logging.getLogger('mlflow')
mlflow_logger.setLevel(logging.ERROR)


@pytest.fixture
def random_graph():
    """Generates a random graph with random attributes for testing."""
    np.random.seed(42)
    X = np.random.normal(loc=0, scale=1, size=(100, 384))
    y = np.random.choice([0, 1], size=100, p=[0.95, 0.05])

    G = nx.gnm_random_graph(100, 500, seed=42)
    for i, node in enumerate(G.nodes):
        G.nodes[node]['x'] = X[i].squeeze().tolist()
    return G, X, y


@pytest.fixture
def outlier_detector():
    """Fixture to create OutlierDetector instances with custom parameters."""
    def _create_outlier_detector(**kwargs):
        return GraphOutlierDetector(**kwargs)
    return _create_outlier_detector


def pipeline_checks(detector, X, y):
    """Standard checks for a fitted GraphOutlierDetector."""
    assert hasattr(detector, 'threshold_')
    assert hasattr(detector, 'labels_')
    assert hasattr(
        detector.pipeline.named_steps['detector'], 'decision_score_')
    assert hasattr(detector.regressor, 'reg_R2_')

    assert len(
        detector.pipeline.named_steps['detector'].decision_score_.numpy()) == len(y)
    assert len(detector.labels_) == len(y)
    assert np.all(np.isin(detector.labels_, [0, 1]))
    assert detector.regressor.reg_R2_ is not None


class TestGraphOutlierDetector:
    def test_initialization_with_default_detector(self, outlier_detector):
        """Test initialization without a custom detector."""
        detector = outlier_detector()
        assert isinstance(detector, BaseEstimator)
        assert detector.pipeline.named_steps['detector'] is not None
        assert detector.regressor is not None
        assert isinstance(detector.pipeline, Pipeline)

    def test_fit_with_default_detector(self, outlier_detector, random_graph):
        """Test the `fit` method using default detector."""
        G, X, y = random_graph
        detector = outlier_detector()
        detector.fit(G)

        pipeline_checks(detector, X, y)
        assert detector.threshold_ is not None

    @pytest.mark.parametrize('method', [AnomalyDAE(), ANOMALOUS(), GUIDE()])
    def test_fit_with_detector(self, outlier_detector, random_graph, method):
        """Test fitting with different graph outlier detection methods."""
        G, X, y = random_graph
        detector = outlier_detector(detector=method)
        detector.fit(G, y)

        pipeline_checks(detector, X, y)
        assert detector.detector == method

    @pytest.mark.parametrize('contamination', [0.02, 0.15, 85, 110, IQR(), CLF()])
    def test_fit_with_contamination(self, outlier_detector, random_graph, contamination):
        """Test fitting with different contamination setting."""
        G, X, y = random_graph
        detector = outlier_detector(contamination=contamination)
        detector.fit(G)

        pipeline_checks(detector, X, y)
        assert detector.contamination == contamination
        assert detector.threshold_ is not None
        assert detector.threshold_ > 0

    @pytest.mark.parametrize('regressor', [LinearRegression(), RidgeCV(), DecisionTreeRegressor()])
    def test_fit_with_regressor(self, outlier_detector, random_graph, regressor):
        """Test fitting with different mapping regression methods."""
        G, X, y = random_graph
        detector = outlier_detector(regressor=regressor)
        detector.fit(G, y=y)

        pipeline_checks(detector, X, y)
        assert detector.regressor == regressor

    def test_predict_function(self, outlier_detector, random_graph):
        """Test the `predict` function."""
        G, X, y = random_graph
        detector = outlier_detector()
        detector.fit(G)
        labels = detector.predict(X)

        assert labels.shape[0] == X.shape[0]
        assert set(labels).issubset({0, 1})

    def test_decision_function(self, outlier_detector, random_graph):
        """Test the `decision_function` method."""
        G, X, y = random_graph
        detector = outlier_detector()
        detector.fit(G)
        decision_scores = detector.decision_function(X)
        assert decision_scores.shape[0] == X.shape[0]
        assert isinstance(decision_scores, np.ndarray)

    def test_low_degree_extra_attr(self, outlier_detector):
        """Test adding orphaned nodes and extra node attributes."""

        np.random.seed(42)
        X = np.random.normal(loc=0, scale=1, size=(100, 384))
        y = np.random.choice([0, 1], size=100, p=[0.95, 0.05])

        G = nx.gnm_random_graph(100, 50, seed=42)
        for i, node in enumerate(G.nodes):
            G.nodes[node]['x'] = X[i].squeeze().tolist()
            G.nodes[node]['rnd_attr'] = 'test'

        detector = outlier_detector()
        detector.fit(G)

        decision_scores = detector.pipeline.named_steps['detector'].decision_score_.numpy(
        )
        assert decision_scores.shape[0] < X.shape[0]

    def test_missing_x_attr(self, outlier_detector):
        """Test fit failure due to missing x node attribute."""

        np.random.seed(42)
        G = nx.gnm_random_graph(100, 200)
        for i, node in enumerate(G.nodes):
            G.nodes[node]['rnd_attr'] = 'test'

        detector = outlier_detector()
        with pytest.raises(ValueError) as error:
            detector.fit(G)

    def test_model_loading_and_reinitialization(self, tmp_path, outlier_detector, random_graph):
        """Test saving, loading, and reinitializing the model."""

        model_path = tmp_path / 'outlier_detector.pkl'
        regressor_path = tmp_path / 'regressor.pkl'
        G, X, y = random_graph

        detector = outlier_detector(mlflow=False, model=str(model_path),
                                    regressor_model=str(regressor_path))
        detector.fit(G)
        labels = detector.predict(X)

        # Check the saved file exists
        assert model_path.exists()
        assert regressor_path.exists()

        # Reinitialize and load the model
        new_detector = GraphOutlierDetector(
            model=str(model_path), regressor_model=str(regressor_path))
        pipeline_checks(detector, X, y)

        new_labels = new_detector.predict(X)

        decision_scores = detector.pipeline.named_steps['detector'].decision_score_.numpy(
        )
        new_decision_scores = new_detector.pipeline.named_steps['detector'].decision_score_.numpy(
        )
        assert np.allclose(decision_scores, new_decision_scores)
        assert np.allclose(labels, new_labels)

    def test_mlflow_not_installed(self, random_graph):
        """Test behavior when MLflow is not installed."""

        G, _, _ = random_graph

        original_find_spec = importlib.util.find_spec

        def custom_find_spec(name):
            if name == 'mlflow':
                return None
            return original_find_spec(name)

        with patch('importlib.util.find_spec', side_effect=custom_find_spec):
            importlib.reload(det)
            detector = GraphOutlierDetector(mlflow=True)
            detector.fit(G)
            assert not detector.mlflow

        # Reload the module to restore original state
        importlib.reload(det)
