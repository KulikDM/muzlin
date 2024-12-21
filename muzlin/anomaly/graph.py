import importlib.util
import os
from typing import List, Optional, Tuple, Type, Union

import joblib
import networkx as nx
import numpy as np
import torch
import torch_geometric
from pydantic import BaseModel, Field
from pygod.detector.base import Detector
from pythresh.thresholds.base import BaseThresholder
from sklearn.base import BaseEstimator, OutlierMixin, RegressorMixin
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch_geometric.utils import from_networkx

from muzlin.utils.logger import logger

Gtype = nx.Graph
Ttype = torch_geometric.data.Data
XType = Union[np.ndarray, List[List[Union[float, int]]]]
is_mlflow = importlib.util.find_spec('mlflow')


class GraphOutlierDetector(BaseEstimator, OutlierMixin, BaseModel):
    r"""GraphOutlierDetector class for graph based anomaly detection.

    Given a networkx graph the GraphOutlierDetector class can be
    used to fit an anomaly detection model and predict of new incoming data.

    Args:
        detector (object, optional): The PyGOD graph anomaly detection model. Defaults to PyGOD AnomalyDAE detector.
        regressor (object, optional): The regression for linking the encoded vectors with the graph anonamly likelihood scores. Defaults to RidgeCV.
        contamination (object, float, or int, optional): The level of contamation that is present in the data or unseen data. Used for setting the inlier/outlier threshold.
            - Type object: A PyThresh dynamic thresholding method can be used e.g. inter-quartile range.
            - Type float: The fraction of the fitted data that is outliers.
            - Type int: The percentile to set the threshold level e.g. 78 -> 78%, max is 100%.
            Defaults to None.
        mlflow (bool, optional): To use mlflow experiment tracking during model fitting. Setting False will fit a pickle file of the fitted model in the local folder. Defaults to True.
        model (str, optional): Name of the graph outlier detctor model to load/save. Defaults to 'outlier_detector.pkl'.
        regression_model (str, optional): Name of the vector-graph regression model to load/save. Defaults to 'regressor.pkl'.
        random_state: (int, optional): The random seed for model fitting. Defaults to 1234.

    Attributes:
        pipeline (object): The sklearn pipeline of the fitted model.
        decision_scores_ (array-like): The array of the fitted decision scores for the training data.
        threshold_ (float): The percentile used to threshold inliers from outliers in the model.
        labels_ (array-like): The array of the fitted binary labels for the training data.
        reg_R2_ (float): The R2 score of the fitted regression model on the training data.

    """

    detector: Optional[Detector] = None
    regressor: Optional[RegressorMixin] = None
    contamination: Optional[Union[BaseThresholder, float, int]] = 0.1
    mlflow: Optional[bool] = True
    model: Optional[str] = 'outlier_detector.pkl'
    regressor_model: Optional[str] = 'regressor.pkl'
    random_state: Optional[int] = 1234

    pipeline: Pipeline = Field(default=None, exclude=True)
    decision_scores_: Type[np.ndarray] = Field(default=None, exclude=True)
    threshold_: float = Field(default=None, exclude=True)
    labels_: Type[np.ndarray] = Field(default=None, exclude=True)
    reg_R2_: float = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):

        super().__init__(**data)

        torch.manual_seed(self.random_state)

        self._check_is_initalized()
        if self.pipeline is not None:
            return

        if self.detector is None:

            logger.info(
                'No graph outlier detector was provided defaulting to PyGOD AnomalyDAE detector.'
            )
            global AnomalyDAE
            from pygod.detector import AnomalyDAE
            self.detector = AnomalyDAE()

        if self.regressor is None:

            logger.info(
                'No regression model was provided defaulting to sklearn RidgeCV.'
            )
            global RidgeCV
            from sklearn.linear_model import RidgeCV
            self.regressor = RidgeCV()

        self.pipeline = Pipeline([
            ('detector', self.detector)
        ])

    def fit(self, graph: Gtype, y=None):
        r"""Fit function of the OutlierDetector class.

        Args:
            graph (object): The nx.graph of the training dataset.
            y (array-like, or None, optional): Not required.
        """

        # Prepare graph and vector data before fitting
        graph_torch, vectors = self._preprocess_graph(graph)

        if self.mlflow:

            if is_mlflow:
                global ml
                import mlflow as ml
                ml.autolog()
            else:
                logger.info('MLFlow not installed, defaulting to joblib')
                self.mlflow = False

        self.pipeline.fit(graph_torch)

        scores = self.pipeline.named_steps['detector'].decision_score_.numpy()
        contam = self.contamination

        # Fit mapping function
        if self.mlflow:
            ml.autolog(disable=True)
        self.regressor.fit(vectors, scores)
        pred = self.regressor.predict(vectors)
        reg_R2_ = r2_score(scores, pred)

        # Cater for different types of thresholding options
        if isinstance(self.contamination, BaseThresholder):
            lbls = self.contamination.eval(scores)
            contam = len(lbls[lbls == 0])/len(lbls)
        elif self.contamination > 1.0:
            contam /= 100
        else:
            contam = 1 - contam

        self.threshold_ = (np.percentile(scores, contam*100) if
                           contam <= 1.0 else contam * np.max(scores))

        labels = (scores > self.threshold_).astype('int').ravel()

        setattr(self.pipeline, 'threshold_', self.threshold_)
        setattr(self.pipeline, 'labels_', labels)
        setattr(self.regressor, 'reg_R2_', reg_R2_)

        # Relog model to save attr
        if self.mlflow:
            run_id = self._fetch_mlflow_run_id()
            with ml.start_run(run_id=run_id) as _:
                ml.sklearn.log_model(self.pipeline, 'model')
                ml.sklearn.log_model(self.regressor, 'regression_model')
                ml.log_param('regressor', self.regressor)
                ml.log_metric('regressor_r2_score', reg_R2_)

        if not self.mlflow:
            joblib.dump(self.pipeline, self.model)
            joblib.dump(self.regressor, self.regressor_model)

        self._check_is_initalized()

        return self

    def predict(self, X: XType) -> np.ndarray:
        r"""Predict function of the OutlierDetector class.

        Args:
            X (array-like): The vectors to predict their labels.

        Returns:
            labels (array-like): The predicted binary labels.

        """
        check_is_fitted(self.regressor)
        X = check_array(X, ensure_2d=True)

        scores = self.decision_function(X)
        labels = (scores > self.threshold_).astype('int').ravel()
        return labels

    def decision_function(self, X: XType) -> np.ndarray:
        r"""Decision function of the OutlierDetector class.

        Args:
            X (array-like): The vectors to predict their decision scores.

        Returns:
            decision scores (array-like): The predicted decision scores.

        """
        check_is_fitted(self.regressor)
        X = check_array(X, ensure_2d=True)
        score = self.regressor.predict(X)

        return score

    def _preprocess_graph(self, graph: Gtype) -> Tuple[Ttype, XType]:

        # Check no orphaned nodes
        nodes_and_indices = [(index, node) for index, (node, degree) in
                             enumerate(dict(graph.degree()).items()) if degree == 0]

        indices, nodes_to_remove = zip(
            *nodes_and_indices) if nodes_and_indices else ([], [])

        graph.remove_nodes_from(nodes_to_remove)

        # Clean node attributes: retain only 'x' (first pass)
        vectors = []
        for node, attr in graph.nodes(data=True):
            if 'x' in attr:
                x = graph.nodes[node]['x']
                graph.nodes[node].clear()
                graph.nodes[node]['x'] = x
                vectors.append(x)
            else:
                raise ValueError(
                    f"Node {node} does not have the 'x' attribute.")

        graph_torch = from_networkx(graph)

        # Clean node attributes: retain only 'x' (second pass)
        attrs_to_keep = ['x', 'edge_index']
        for attr in list(graph_torch.keys()):
            if attr not in attrs_to_keep:
                delattr(graph_torch, attr)

        return graph_torch, np.array(vectors)

    def _fetch_mlflow_run_id(self):
        run = ml.last_active_run()
        run_id = run.info.run_id
        return run_id

    def _check_is_initalized(self):

        if ((os.path.exists(self.model)) & (self.pipeline is None) &
           (os.path.exists(self.regressor_model)) & (self.regressor is None)):
            self.pipeline = joblib.load(self.model)
            self.regressor = joblib.load(self.regressor_model)
        elif self.pipeline is None:
            return

        check_is_fitted(self.pipeline)
        check_is_fitted(self.regressor)

        self.threshold_ = self.pipeline.threshold_
        self.labels_ = self.pipeline.labels_
        self.decision_scores_ = self.pipeline.named_steps['detector'].decision_score_.numpy(
        )
        self.reg_R2_ = self.regressor.reg_R2_
