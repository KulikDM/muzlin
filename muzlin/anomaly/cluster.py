import importlib.util
import os
from collections import namedtuple
from typing import List, Optional, Type, Union

import joblib
import numpy as np
import umap
from pydantic.v1 import BaseModel, Field
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin, OutlierMixin, TransformerMixin
from sklearn.mixture._base import BaseMixture
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from muzlin.utils.logger import logger

XType = Union[np.ndarray, List[List[Union[float, int]]]]
is_mlflow = package_spec = importlib.util.find_spec('mlflow')

ClusterPredict = namedtuple(
    'ClusterPredict', ('nclust_cls', 'topk_cls', 'density_cls'))
ClusterDecision = namedtuple(
    'ClusterDecision', ('nclust_dev', 'topk_dev', 'density_dev'))

dim_reducer = umap.UMAP(n_components=10,
                        metric='euclidean',
                        set_op_mix_ratio=0.3,
                        min_dist=0.98,
                        local_connectivity=5,
                        random_state=1234)


# https://github.com/parthsarthi03/raptor/blob/master/raptor/cluster_utils.py

class OutlierCluster(BaseEstimator, OutlierMixin, BaseModel):
    r"""OutlierCluster class for vector based anomaly cluster detection.

    Given a set of embedded vectors the OutlierCluster class can be
    used to fit an anomaly cluster detection model and predict on new incoming data
    if the data belongs to a subcluster e.g. matched context from a RAG or not.

    Args:
        method (object, optional): The anomaly cluster detection model. Can either be a sklearn cluster or mixture model type. Defaults to sklearn KMeans.
        decomposer (object, None,  optional): The preprocessing decomposition model to reduce the dimensionality of the data to speed up clustering. Can be set to None to not be applied. Defaults to UMAP.
        n_retrieve (int, optional): Number of context new data must be compared to e.g. 10 retrieved docs per query. Defaults to 10.
        mlflow (bool, optional): To use mlflow experiment tracking during model fitting. Setting False will fit a pickle file of the fitted model in the local folder. Defaults to True.
        model (str, optional): Name of the model to load/save. Defaults to 'outlier_cluster.pkl'.
        random_state: (int, optional): The random seed for model fitting. Defaults to 1234.


    Attributes:
        pipeline (object): The sklearn pipeline of the fitted model.
        labels_ (array-like): The array of the fitted labels for clusters of the training data.
        avg_std_ (float): The average cluster centriod deviation of the training data.

    """

    method: Optional[Union[ClusterMixin, BaseMixture, None]] = None
    decomposer: Optional[Union[TransformerMixin, None]] = dim_reducer
    n_retrieve: Optional[int] = 10
    mlflow: Optional[bool] = True
    model: Optional[str] = 'outlier_cluster.pkl'
    random_state: Optional[int] = 1234

    pipeline: Pipeline = Field(default=None, exclude=True)
    avg_std_: float = Field(default=None, exclude=True)
    labels_: Type[np.ndarray] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):

        super().__init__(**data)

        self._check_is_initalized()
        if self.pipeline is not None:
            return

        # Apply dummy function
        if self.decomposer is None:
            from sklearn.preprocessing import FunctionTransformer
            self.decomposer = FunctionTransformer(lambda x: x)

        if hasattr(self.decomposer, 'random_state'):
            self.decomposer.random_state = self.random_state

        if self.method is None:
            logger.info(
                'No clustering method was provided defaulting to KMeans.'
            )
            from sklearn.cluster import KMeans
            self.method = KMeans(n_clusters=2, random_state=self.random_state)

        self.pipeline = Pipeline([
            ('decompose', self.decomposer),
            ('cluster', self.method)
        ])

    def fit(self, X: XType, y=None):
        r"""Fit function of the OutlierCluster class.

        Args:
            X (array-like): The vectors of the training dataset.
            y (array-like, or None, optional): Not required.

        """

        if self.mlflow:

            if is_mlflow:
                global ml
                import mlflow as ml
                ml.autolog()
            else:
                logger.info('MLFlow not installed, defaulting to joblib')
                self.mlflow = False

        X = check_array(X, ensure_2d=True)

        # Set dynamic n_neighbors (applies only to umap)
        if hasattr(self.pipeline.named_steps['decompose'], 'n_neighbors'):
            n_neighbors = int(X.shape[1]/15)
            self.pipeline.named_steps['decompose'].n_neighbors = n_neighbors

        # Optimal clusters should be ~2*topk retrieved context
        optimal_n_clusters = int(X.shape[0] / (2 * self.n_retrieve))

        if hasattr(self.pipeline.named_steps['cluster'], 'n_components'):
            self.pipeline.named_steps['cluster'].n_components = optimal_n_clusters
        elif hasattr(self.pipeline.named_steps['cluster'], 'n_clusters'):
            self.pipeline.named_steps['cluster'].n_clusters = optimal_n_clusters
        else:
            pass

        labels = self.pipeline.fit_predict(X, y)

        # Get the avg median centroid deviation
        unq_labels = np.unique(labels)
        stds = [np.median(np.abs(np.mean(X[label], axis=0) - X[label]))
                for label in unq_labels]
        avg_std = np.mean(stds)

        setattr(self.pipeline, 'avg_std_', avg_std)
        setattr(self.pipeline.named_steps['cluster'], 'avg_std_', avg_std)

        # Relog model to save attr
        if self.mlflow:
            run_id = self._fetch_mlflow_run_id()
            with ml.start_run(run_id=run_id) as _:
                ml.sklearn.log_model(self.pipeline, 'model')

        if not self.mlflow:
            joblib.dump(self.pipeline, self.model)

        self._check_is_initalized()

        return self

    def predict(self, query: XType, docs: XType) -> namedtuple:
        r"""Predict function of the OutlierCluster class.

        Args:
            query (array-like): The vectors to predict their labels.
            docs (array-like): The vectors of the reference for which to compare to e.g. context

        Returns:
            labels (tuple): NamedTuple of predicted class for the three tests:
                - clust_cls: Binary label for belonging to an optimal number of clusters (e.g. not to dense or sparse in detail).
                - topk_cls: Binary label for being a realistic cluster with respect to the entire fitted data.
                - density_cls: Binary label for being from context centriod with respect to compactness.

        """

        check_is_fitted(self.pipeline)
        query = check_array(query, ensure_2d=True)
        docs = check_array(docs, ensure_2d=True)

        # Get scores and binary threshold
        scores = self.decision_function(query, docs)

        nclust_dev = scores.nclust_dev
        topk_dev = scores.topk_dev
        density_dev = scores.density_dev

        nclust_class = 0 if nclust_dev <= 0.5 else 1
        topk_class = 0 if topk_dev <= 1 else 1
        density_class = 0 if density_dev <= 1 else 1

        return ClusterPredict(nclust_cls=nclust_class, topk_cls=topk_class, density_cls=density_class)

    def decision_function(self, query: XType, docs: XType) -> namedtuple:
        r"""Decision function of the OutlierCluster class.

        Args:
            query (array-like): The vectors to predict their decision scores.
            docs (array-like): The vectors of the reference for which to compare to e.g. context

        Returns:
            scores (tuple): NamedTuple of predicted class for the three tests:
                - nclust_dev: Deviation from belonging to an optimal number of clusters (e.g. not to dense or sparse in detail).
                - topk_dev: Deviation from being a realistic cluster with respect to the entire fitted data.
                - density_dev: Deviation from context centriod with respect to compactness.

        """

        check_is_fitted(self.pipeline)
        query = check_array(query, ensure_2d=True)
        docs = check_array(docs, ensure_2d=True)

        top_k = len(docs)

        # Find the deviation from best variety response ~ 1/2 the number of requested context
        # e.g too few clusters signify dense data that may be too limited/focused
        # e.g too many clusters signify sparse data that may be too broad/general
        labels = self.get_cluster(docs)
        matched_nclust = len(np.unique(labels))
        base_k = top_k/2

        nclust_dev = abs(base_k - matched_nclust)/base_k

        # Get the mean std of all the fitted clusters
        # Assume the question is the centriod of the cluster and compare
        # the mean deviation from the psuedo-centriod to the mean std of the fitted clusters
        index_dev = self.pipeline.avg_std_
        centroid_dev = np.mean(np.abs(docs - query.squeeze()))
        topk_dev = centroid_dev/index_dev

        # Get the cosine distance from the retrived context centriod and the query
        # Compare this distance in both compactness of the cluster and zscore
        centroid = docs.mean(axis=0)

        cent_dist = np.abs(cdist([centroid], docs, metric='cosine').flatten())
        cent_q_diff = np.abs(
            cdist([centroid], [query.ravel()], metric='cosine').flatten())

        dist_range = np.max(cent_dist) - np.min(cent_dist)
        density_dev = cent_q_diff[0]/(2 * np.std(cent_dist)) * \
            (1 - np.std(cent_dist)/dist_range)

        return ClusterDecision(nclust_dev=nclust_dev, topk_dev=topk_dev, density_dev=density_dev)

    def get_cluster(self, X: XType) -> np.ndarray:
        r"""Decision function of the OutlierDetector class.

        Args:
            X (array-like): The vectors to predict their cluster label.

        Returns:
            label (array-like): The predicted cluster label for the vectors.

        """
        check_is_fitted(self.pipeline)
        X = check_array(X, ensure_2d=True)
        return self.pipeline.predict(X)

    def _fetch_mlflow_run_id(self):
        run = ml.last_active_run()
        run_id = run.info.run_id
        return run_id

    def _check_is_initalized(self):

        if (os.path.exists(self.model)) & (self.pipeline is None):
            self.pipeline = joblib.load(self.model)
        elif self.pipeline is None:
            return

        check_is_fitted(self.pipeline)

        self.avg_std_ = self.pipeline.avg_std_
        self.labels_ = self.pipeline.named_steps['cluster'].labels_
