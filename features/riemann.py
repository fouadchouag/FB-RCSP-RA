import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM


class RiemannMDMClassifier:
    """
    Riemannian Minimum Distance to Mean (MDM) Classifier.

    Treats EEG covariance matrices as points on the Symmetric Positive
    Definite (SPD) manifold. Classification is performed by computing
    Riemannian distances between a trial's covariance matrix and the
    Riemannian mean of each class.

    This is a standalone classifier (not a feature extractor):
    fit() and predict() operate directly on covariance matrices.

    Reference:
        Barachant et al. (2012), "Multiclass Brain-Computer Interface
        Classification by Riemannian Geometry", IEEE TBME.
    """

    def __init__(self):
        self.cov_estimator = Covariances(estimator='oas')
        self.clf = MDM(metric='riemann')

    def fit(self, X, y):
        """
        Estimate covariance matrices and fit MDM classifier.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,)
        """
        covs = self.cov_estimator.fit_transform(X)
        self.clf.fit(covs, y)
        return self

    def predict(self, X):
        """
        Predict class labels for new trials.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        y_pred : np.ndarray, shape (n_trials,)
        """
        covs = self.cov_estimator.transform(X)
        return self.clf.predict(covs)

    def fit_predict(self, X_train, y_train, X_test):
        """Convenience method: fit on train, predict on test."""
        self.fit(X_train, y_train)
        return self.predict(X_test)