import sys
from importlib import reload
from unittest.mock import patch

import muzlin.anomaly as det


class TestAnomalyInit:
    def test_lazy_loading(self):

        with patch('typing.TYPE_CHECKING', False), \
                patch('apipkg.initpkg') as mock_initpkg:

            reload(det)

            # Check if apipkg.initpkg was called with the correct arguments
            mock_initpkg.assert_called_once_with('muzlin.anomaly', {
                'OutlierCluster': 'muzlin.anomaly.cluster:OutlierCluster',
                'OutlierDetector': 'muzlin.anomaly.detector:OutlierDetector',
                'GraphOutlierDetector': 'muzlin.anomaly.graph:GraphOutlierDetector',
                'optimize_threshold': 'muzlin.anomaly.utils:optimize_threshold',
            })

    def test_type_checking_imports(self):

        with patch('typing.TYPE_CHECKING', True), \
                patch('muzlin.anomaly.cluster.OutlierCluster', create=True), \
                patch('muzlin.anomaly.detector.OutlierDetector', create=True), \
                patch('muzlin.anomaly.graph.GraphOutlierDetector', create=True), \
                patch('muzlin.anomaly.utils.optimize_threshold', create=True):

            reload(det)

            # Ensure lazy loading is skipped
            assert 'muzlin.anomaly.cluster' in sys.modules
            assert 'muzlin.anomaly.detector' in sys.modules
            assert 'muzlin.anomaly.graph' in sys.modules
            assert 'muzlin.anomaly.utils' in sys.modules
