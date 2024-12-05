import importlib
import logging
import os
import importlib.util
from unittest.mock import patch

import mlflow
import numpy as np
import pytest
import umap
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, Birch, HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline

import muzlin.anomaly.cluster as det
from muzlin.anomaly import OutlierCluster


logging.basicConfig(level=logging.ERROR)

mlflow_logger = logging.getLogger('mlflow')
mlflow_logger.setLevel(logging.ERROR)


@pytest.fixture
def sample_data():
    """Generates a synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.normal(loc=0, scale=1, size=(1000, 384))
    return X

@pytest.fixture
def outlier_cluster():
    """Fixture to create OutlierDetector instances with custom parameters."""
    def _create_outlier_cluster(**kwargs):
        return OutlierCluster(**kwargs)
    return _create_outlier_cluster

def pipeline_checks(cluster, X):
    """Fixture to create OutlierDetector instances with custom parameters."""
    assert hasattr(cluster, 'avg_std_')
    assert hasattr(cluster.pipeline.named_steps['cluster'], 'labels_')
    assert hasattr(cluster.pipeline.named_steps['cluster'], 'avg_std_')

    assert len(cluster.pipeline.named_steps['cluster'].labels_) == len(X)

class TestOutlierCluster:
    def test_initialization_with_default_cluster(self, outlier_cluster):
        """Test initialization without a custom detector."""
        cluster = outlier_cluster()
        assert isinstance(cluster, BaseEstimator)
        assert cluster.pipeline.named_steps['cluster'] is not None
        assert cluster.pipeline.named_steps['decompose'] is not None
        assert isinstance(cluster.pipeline, Pipeline)

    def test_fit_with_default_cluster(self, outlier_cluster, sample_data):
        """Test the `fit` method using default detector."""
        X = sample_data
        cluster = outlier_cluster()
        cluster.fit(X)

        pipeline_checks(cluster, X)

    @pytest.mark.parametrize("method", [AgglomerativeClustering(), Birch(), HDBSCAN()])
    def test_fit_with_method(self, outlier_cluster, sample_data, method):
        """Test fitting with different clustering methods"""
        X = sample_data
        cluster = outlier_cluster(method=method)
        cluster.fit(X)

        pipeline_checks(cluster, X)
        assert cluster.method == method

    @pytest.mark.parametrize("decomposer", [None, PCA(n_components=10), TruncatedSVD(n_components=10)])
    def test_fit_with_decomposer(self, outlier_cluster, sample_data, decomposer):
        """Test fitting with different dimensional decomposition methods"""
        X = sample_data
        cluster = outlier_cluster(decomposer=decomposer)
        cluster.fit(X)

        pipeline_checks(cluster, X)

    @pytest.mark.parametrize("n_retrieve", [3, 5, 10, 20, 50])
    def test_fit_with_n_retrieve(self, outlier_cluster, sample_data, n_retrieve):
        """Test fitting with different retrieve sizes"""
        X = sample_data
        cluster = outlier_cluster(n_retrieve=n_retrieve)
        cluster.fit(X)

        pipeline_checks(cluster, X)
        assert cluster.n_retrieve == n_retrieve

    def test_predict_function(self, outlier_cluster, sample_data):
        """Test the `predict` function."""
        X  = sample_data
        np.random.seed(42)
        docs = np.random.normal(loc=0, scale=1, size=(10, 384))
        query = np.random.normal(loc=0, scale=1, size=(1, 384))
        
        cluster = outlier_cluster()
        cluster.fit(X)
        result = cluster.predict(query, docs)
        
        assert hasattr(result, 'nclust_cls')
        assert result.nclust_cls is not None

        assert hasattr(result, 'topk_cls')
        assert result.topk_cls is not None

        assert hasattr(result, 'density_cls')
        assert result.density_cls is not None

    def test_decision_function(self, outlier_cluster, sample_data):
        """Test the `decision_function` method."""
        X  = sample_data
        np.random.seed(42)
        docs = np.random.normal(loc=0, scale=1, size=(10, 384))
        query = np.random.normal(loc=0, scale=1, size=(1, 384))
        
        cluster = outlier_cluster()
        cluster.fit(X)
        result = cluster.decision_function(query, docs)

        assert hasattr(result, 'nclust_dev')
        assert result.nclust_dev is not None

        assert hasattr(result, 'topk_dev')
        assert result.topk_dev is not None

        assert hasattr(result, 'density_dev')
        assert result.density_dev is not None

    def test_model_loading_and_reinitialization(self, tmp_path, outlier_cluster, sample_data):
        """Test saving, loading, and reinitializing the model."""
        
        model_path = tmp_path / "outlier_cluster.pkl"
        X = sample_data
        np.random.seed(42)
        docs = np.random.normal(loc=0, scale=1, size=(10, 384))
        query = np.random.normal(loc=0, scale=1, size=(1, 384))

        cluster = outlier_cluster(mlflow=False, model=str(model_path))
        cluster.fit(X)
        result = cluster.predict(query, docs)
        
        # Check the saved file exists
        assert model_path.exists()

        # Reinitialize and load the model
        new_cluster = OutlierCluster(model=str(model_path))
        pipeline_checks(cluster, X)

        new_result = new_cluster.predict(query, docs)

        labels = cluster.pipeline.named_steps['cluster'].labels_
        new_labels = new_cluster.labels_

        avg_std = cluster.avg_std_
        new_avg_std = new_cluster.avg_std_

        assert np.allclose(labels, new_labels)
        assert avg_std == new_avg_std

        assert result.nclust_cls == new_result.nclust_cls
        assert result.topk_cls == new_result.topk_cls
        assert result.density_cls == new_result.density_cls

    def test_mlflow_not_installed(self, sample_data):
        """Test behavior when MLflow is not installed."""

        X = sample_data

        original_find_spec = importlib.util.find_spec
        
        def custom_find_spec(name):
            if name == 'mlflow':
                return None
            return original_find_spec(name)

        with patch("importlib.util.find_spec", side_effect=custom_find_spec):
            importlib.reload(det)
            cluster = OutlierCluster(mlflow=True)
            cluster.fit(X)
            assert not cluster.mlflow

        # Reload the module to restore original state
        importlib.reload(det)
