import logging

import mlflow
import numpy as np
import pytest

from muzlin.anomaly import OutlierDetector, optimize_threshold

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

class TestUtils:

    @pytest.mark.parametrize("policy", ['hard', 'soft', 'balanced'])
    def test_optimize_with_policy(self, outlier_detector, sample_data, policy):
        """Test comtaination optimization using different policies"""
        X, y = sample_data
        detector = outlier_detector()
        detector.fit(X)

        np.random.seed(42)
        ref_vectors = np.random.normal(loc=2, scale=1, size=(4, 384))

        fitted_scores = detector.decision_scores_
        pred_scores = detector.decision_function(ref_vectors)

        real_labels = [1, 0, 1, 0,]

        thresh_score, thresh_perc = optimize_threshold(fitted_scores, pred_scores, real_labels, policy=policy)

        assert thresh_score is not None
        assert 0 <= thresh_perc <= 300

