import importlib.util
import os
from typing import List, Optional, Type, Union

import joblib
import numpy as np
from pydantic.v1 import BaseModel, Field
from pyod.models.base import BaseDetector
from pythresh.thresholds.base import BaseThresholder
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from muzlin.utils.logger import logger

XType = Union[np.ndarray, List[List[Union[float, int]]]]
is_mlflow = importlib.util.find_spec('mlflow')


class OutlierDetector(BaseEstimator, OutlierMixin, BaseModel):
    r"""OutlierDetector class for vector based anomaly detection.

    Given a set of embedded vectors the OutlierDetector class can be
    used to fit an anomaly detection model and predict of new incoming data.

    Args:
        detector (object, optional): The anomaly detection model. Can either be a pyod or sklearn classification model. Defaults to PyOD PCA.
        contamination (object, float, or int, optional): The level of contamation that is present in the data or unseen data. Used for setting the inlier/outlier threshold.
            - Type object: A PyThresh dynamic thresholding method can be used e.g. inter-quartile range.
            - Type float: The fraction of the fitted data that is outliers.
            - Type int: The percentile to set the threshold level e.g. 78 -> 78%, 120 -> 120%.
            Defaults to None.
        mlflow (bool, optional): To use mlflow experiment tracking during model fitting. Setting False will fit a pickle file of the fitted model in the local folder. Defaults to True.
        model (str, optional): Name of the model to load/save. Defaults to 'outlier_detector.pkl'.
        random_state: (int, optional): The random seed for model fitting. Defaults to 1234.

    Attributes:
        pipeline (object): The sklearn pipeline of the fitted model.
        decision_scores_ (array-like): The array of the fitted decision scores for the training data.
        threshold_ (float): The percentile used to threshold inliers from outliers in the model.
        labels_ (array-like): The array of the fitted binary labels for the training data.

    """

    detector: Optional[Union[BaseDetector, ClassifierMixin]] = None
    contamination: Optional[Union[BaseThresholder, float, int, None]] = None
    mlflow: Optional[bool] = True
    model: Optional[str] = 'outlier_detector.pkl'
    random_state: Optional[int] = 1234

    pipeline: Pipeline = Field(default=None, exclude=True)
    decision_scores_: Type[np.ndarray] = Field(default=None, exclude=True)
    threshold_: float = Field(default=None, exclude=True)
    Xstd_: float = Field(default=None, exclude=True)
    labels_: Type[np.ndarray] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):

        super().__init__(**data)

        self._check_is_initalized()
        if self.pipeline is not None:
            return

        if self.detector is None:

            logger.info(
                'No outlier detector was provided defaulting to PyOD PCA detector.'
            )
            global PCA
            from pyod.models.pca import PCA
            self.detector = PCA(contamination=0.1,
                                random_state=self.random_state)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('detector', self.detector)
        ])

    def fit(self, X: XType, y: Optional[Union[np.ndarray, list, None]] = None):
        r"""Fit function of the OutlierDetector class.

        Args:
            X (array-like): The vectors of the training dataset.
            y (array-like, or None, optional): Not required if using a PyOD model. A binary set of labels are needed if using a Sklearn classification model.

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
        y = check_array(y, ensure_2d=False) if y is not None else y

        if (not np.all(np.isin(y, [0, 1]))) & (y is not None):
            raise ValueError('y should only contain binary values 0 or 1.')

        # Log training data deviation for future use
        self.pipeline.named_steps['detector'].Xstd_ = np.std(X)

        self.pipeline.fit(X, y)

        # Cater for supervised method
        if isinstance(self.pipeline.named_steps['detector'], ClassifierMixin):

            scores = self.pipeline.predict_proba(X)
            setattr(
                self.pipeline.named_steps['detector'], 'decision_scores_', scores)

        labels = self.pipeline.predict(
            X) if y is not None else self.pipeline.named_steps['detector'].labels_

        # Cater for different types of thresholding options
        if (self.contamination is not None) & (y is None):

            scores = self.pipeline.named_steps['detector'].decision_scores_
            contam = self.contamination

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

        # Relog model to save attr
        if self.mlflow:
            run_id = self._fetch_mlflow_run_id()
            with ml.start_run(run_id=run_id) as _:
                ml.sklearn.log_model(self.pipeline, 'model')

        if not self.mlflow:
            joblib.dump(self.pipeline, self.model)

        self._check_is_initalized()

        return self

    def predict(self, X: XType) -> np.ndarray:
        r"""Predict function of the OutlierDetector class.

        Args:
            X (array-like): The vectors to predict their labels.

        Returns:
            labels (array-like): The predicted binary labels.

        """
        check_is_fitted(self.pipeline)
        X = check_array(X, ensure_2d=True)

        if self.threshold_ is not None:
            scores = self.decision_function(X)
            labels = (scores > self.threshold_).astype('int').ravel()
        else:
            labels = self.pipeline.predict(X)
        return labels

    def decision_function(self, X: XType) -> np.ndarray:
        r"""Decision function of the OutlierDetector class.

        Args:
            X (array-like): The vectors to predict their decision scores.

        Returns:
            decision scores (array-like): The predicted decision scores.

        """
        check_is_fitted(self.pipeline)
        X = check_array(X, ensure_2d=True)
        return self.pipeline.decision_function(X)

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

        self.threshold_ = self.pipeline.threshold_
        self.labels_ = self.pipeline.labels_
        self.decision_scores_ = self.pipeline.named_steps['detector'].decision_scores_
        self.Xstd_ = self.pipeline.named_steps['detector'].Xstd_
