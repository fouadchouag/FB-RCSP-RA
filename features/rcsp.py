import numpy as np
from sklearn.preprocessing import LabelBinarizer
from mne.decoding import CSP


class RCSPFeatureExtractor:
    """
    Multi-class Regularized CSP (RCSP) Feature Extractor.
    Uses Ledoit-Wolf shrinkage regularization on covariance matrices.
    Strategy: One-vs-Rest (OVR) for multi-class extension.

    Reference:
        Lotte & Guan (2011), "Regularizing Common Spatial Patterns to
        Improve BCI Designs", IEEE TNSRE.
    """

    def __init__(self, n_components=6, reg='ledoit_wolf'):
        """
        Parameters
        ----------
        n_components : int
            Number of CSP spatial filters per class (OVR).
        reg : str or float
            Regularization parameter passed to MNE CSP.
            'ledoit_wolf' applies automatic shrinkage estimation.
            A float in [0,1] applies Tikhonov regularization.
        """
        self.n_components = n_components
        self.reg = reg
        self.csp_dict = {}
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit one regularized CSP per class (OVR).

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,)
        """
        self.classes_ = np.unique(y)
        self.csp_dict = {}

        lb = LabelBinarizer()
        Y_bin = lb.fit_transform(y)  # (n_trials, n_classes)

        for i, cls in enumerate(self.classes_):
            y_binary = Y_bin[:, i]  # 1 = this class, 0 = rest

            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,          # ← REAL regularization
                log=True,
                norm_trace=False
            )
            csp.fit(X, y_binary)
            self.csp_dict[cls] = csp

        return self

    def transform(self, X):
        """
        Extract regularized CSP features for all classes (OVR concatenation).

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_components * n_classes)
        """
        feat_list = []
        for cls in self.classes_:
            csp = self.csp_dict[cls]
            feat = csp.transform(X)
            feat_list.append(feat)
        return np.concatenate(feat_list, axis=1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)