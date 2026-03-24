import numpy as np
from mne.decoding import CSP


class ACMCSPFeatureExtractor:
    """
    Adaptive Multi-Class Common Spatial Pattern (ACMCSP).

    Applies a subject-to-population adaptive whitening transform before
    standard multi-class CSP, reducing inter-subject covariance mismatch.

    The population covariance W_population is computed as the average
    normalized covariance matrix across all training subjects (excluding
    the test subject in LOSO). This must be provided via fit_population()
    before calling fit().

    The adaptive projection is:
        C_global = alpha * C_subject + (1 - alpha) * C_population

    where alpha in [0,1] controls the trade-off between subject-specific
    and population-level spatial statistics.

    Reference:
        Inspired by Samek et al. (2013), "Transfer learning for BCIs",
        IEEE Signal Processing Magazine.
    """

    def __init__(self, n_components=6, alpha=0.5):
        """
        Parameters
        ----------
        n_components : int
            Number of CSP components extracted.
        alpha : float in [0, 1]
            Adaptation strength. alpha=1.0 → subject-specific only.
            alpha=0.0 → population-level only. alpha=0.5 → balanced fusion.
        """
        self.alpha = alpha
        self.n_components = n_components

        self.csp = CSP(
            n_components=n_components,
            log=True
        )

        self.population_cov = None
        self.whitening = None

    def _compute_covariance(self, X):
        """
        Compute the mean normalized covariance matrix over all trials.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        mean_cov : np.ndarray, shape (n_channels, n_channels)
        """
        covs = []
        for trial in X:
            # Zero-mean each channel
            trial = trial - np.mean(trial, axis=1, keepdims=True)
            C = trial @ trial.T
            C = C / np.trace(C)   # normalize by trace for amplitude invariance
            covs.append(C)
        return np.mean(covs, axis=0)

    def fit_population(self, X_population):
        """
        Compute population-level covariance from all training subjects.

        This method must be called BEFORE fit() in a LOSO setup.
        X_population should contain all training subjects' trials
        concatenated (excluding the test subject).

        Parameters
        ----------
        X_population : np.ndarray, shape (n_trials_total, n_channels, n_times)
            Concatenated EEG trials from all training subjects.
        """
        self.population_cov = self._compute_covariance(X_population)

    def fit(self, X, y):
        """
        Fit the adaptive whitening transform and multi-class CSP.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)
        y : np.ndarray, shape (n_trials,)
        """
        # Compute subject-specific covariance
        subject_cov = self._compute_covariance(X)

        # Fallback: if no population provided, use subject covariance only
        if self.population_cov is None:
            self.population_cov = subject_cov.copy()

        # Adaptive fusion: blend subject and population covariances
        # C_global = alpha * C_subject + (1 - alpha) * C_population
        C_global = (
            self.alpha * subject_cov +
            (1.0 - self.alpha) * self.population_cov
        )

        # Compute whitening matrix from fused covariance
        eigvals, eigvecs = np.linalg.eigh(
            C_global + 1e-6 * np.eye(C_global.shape[0])
        )
        eigvals = np.maximum(eigvals, 1e-10)

        # W_adapted = eigvecs * diag(1/sqrt(eigvals)) * eigvecs^T
        self.whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Apply whitening to training data
        X_adapted = np.array([
            self.whitening @ trial for trial in X
        ])

        # Fit standard CSP on whitened data
        self.csp.fit(X_adapted, y)

        return self

    def transform(self, X):
        """
        Apply the fitted adaptive whitening and CSP transform.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_components)
        """
        if self.whitening is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_adapted = np.array([
            self.whitening @ trial for trial in X
        ])
        return self.csp.transform(X_adapted)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)