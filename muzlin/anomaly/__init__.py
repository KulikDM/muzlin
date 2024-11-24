import apipkg
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    # Lazy load the package using apipkg
    apipkg.initpkg(__name__, {
        'OutlierCluster': 'muzlin.anomaly.cluster:OutlierCluster',
        'OutlierDetector': 'muzlin.anomaly.detector:OutlierDetector',
        'GraphOutlierDetector': 'muzlin.anomaly.graph:GraphOutlierDetector',
        'optimize_threshold': 'muzlin.anomaly.utils:optimize_threshold',
    })

else:
    # Direct imports for type checking and static analysis
    from muzlin.anomaly.cluster import OutlierCluster
    from muzlin.anomaly.detector import OutlierDetector
    from muzlin.anomaly.graph import GraphOutlierDetector
    from muzlin.anomaly.utils import optimize_threshold