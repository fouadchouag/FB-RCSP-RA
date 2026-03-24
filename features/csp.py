import numpy as np
from mne.decoding import CSP


class CSPFeatureExtractor:
    """
    Standard Common Spatial Pattern (CSP) Feature Extractor.

    Wraps MNE's CSP implementation with log-variance features.
    Used as the baseline spatial filtering method.

    Reference:
        Ramoser et al. (2000), "Optimal spatial filtering of single trial
        EEG during imagined hand movement", IEEE TRE.
    """

    def __init__(self, n_components=6):
        """
        Parameters
        ----------
        n_components : int
            Number of CSP spatial filters (default: 6).
        """
        self.n_components = n_components
        self.csp = CSP(
            n_components=n_components,
            reg=None,
            log=True,
            norm_trace=False
        )

    def fit(self, X, y):
        """
        Fit CSP spatial filters.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,)
        """
        self.csp.fit(X, y)
        return self

    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.csp.fit_transform(X, y)

    def transform(self, X):
        """
        Apply fitted CSP filters to extract features.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_components)
        """
        return self.csp.transform(X)