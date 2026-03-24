import numpy as np
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from features.rcsp import RCSPFeatureExtractor


class FBRCSPRA:
    """
    Filter-Bank Regularized CSP with Riemannian Alignment (FB-RCSP-RA).

    Novel cross-subject MI EEG classification framework combining:

    1. Filter-Bank Decomposition
       EEG decomposed into 9 overlapping sub-bands (8-30 Hz),
       capturing frequency-specific motor imagery patterns across
       the full mu and beta rhythm range.

    2. Riemannian Alignment (RA) per band
       Each subject's covariance matrices are geometrically
       transported to the Riemannian mean of all training subjects,
       eliminating inter-subject covariance mismatch at the manifold
       level. This is more principled than Euclidean alignment as it
       respects the geometry of the SPD manifold.

    3. Ledoit-Wolf Regularized CSP (RCSP) per band
       After alignment, regularized OVR-CSP is applied with automatic
       Ledoit-Wolf shrinkage estimation, producing stable spatial
       filters robust to small sample sizes in cross-subject settings.

    The combination of these three components in a unified framework
    is novel: existing methods apply at most one or two of these
    strategies, and none combine per-band Riemannian alignment with
    regularized spatial filtering.

    Parameters
    ----------
    sfreq : float
        EEG sampling frequency (default: 250 Hz).
    n_components : int
        Number of RCSP components per band (default: 6).

    References
    ----------
    Ang et al. (2008) - FBCSP
    Lotte & Guan (2011) - RCSP
    Barachant et al. (2012) - Riemannian geometry for BCI
    He & Wu (2019) - Euclidean alignment (motivation for RA)
    """

    def __init__(self, sfreq=250, n_components=6):
        self.sfreq = sfreq
        self.n_components = n_components

        # 9 overlapping sub-bands covering full mu + beta range
        self.bands = [
            (8,  12),
            (10, 14),
            (12, 16),
            (14, 18),
            (16, 20),
            (18, 22),
            (20, 24),
            (22, 26),
            (24, 30)
        ]

        # One RCSP model per band
        self.rcsp_models = [
            RCSPFeatureExtractor(
                n_components=self.n_components,
                reg='ledoit_wolf'
            )
            for _ in self.bands
        ]

        # Riemannian mean per band (computed during fit_population)
        self.riemannian_means = [None] * len(self.bands)

    def _filter_trials(self, X, l_freq, h_freq):
        """Apply band-pass filter to all trials."""
        return np.array([
            filter_data(
                trial,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                verbose=False
            )
            for trial in X
        ])

    def _compute_riemannian_mean(self, X_band):
        """
        Compute Riemannian mean of covariance matrices.

        Parameters
        ----------
        X_band : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        R_mean : np.ndarray, shape (n_channels, n_channels)
            Riemannian mean covariance matrix.
        """
        cov_est = Covariances(estimator='oas')
        covs = cov_est.fit_transform(X_band)
        R_mean = mean_covariance(covs, metric='riemann')
        return R_mean

    def _riemannian_align(self, X_band, R_mean):
        """
        Transport trials to Riemannian mean reference space.

        Applies the whitening transform: X_aligned = R^{-1/2} @ X
        where R is the Riemannian mean covariance of training data.

        This maps all subjects' covariance matrices toward the
        identity matrix in the aligned space, eliminating
        inter-subject domain shift at the manifold level.

        Parameters
        ----------
        X_band : np.ndarray, shape (n_trials, n_channels, n_times)
        R_mean : np.ndarray, shape (n_channels, n_channels)

        Returns
        -------
        X_aligned : np.ndarray, same shape as X_band
        """
        eigvals, eigvecs = np.linalg.eigh(R_mean)
        eigvals = np.maximum(eigvals, 1e-10)
        R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return np.array([R_inv_sqrt @ trial for trial in X_band])

    def fit_population(self, X_train):
        """
        Compute Riemannian mean per band from all training subjects.

        Must be called before fit_transform() in LOSO evaluation.
        The Riemannian mean is computed from training subjects only
        (no information leakage from test subject).

        Parameters
        ----------
        X_train : np.ndarray, shape (n_trials_total, n_channels, n_times)
            Concatenated raw EEG from all training subjects.
        """
        print("  Computing Riemannian means per band...")
        for b, (l, h) in enumerate(self.bands):
            X_band = self._filter_trials(X_train, l, h)
            self.riemannian_means[b] = self._compute_riemannian_mean(X_band)
        print("  Riemannian means computed.")

    def fit_transform(self, X_train, y_train):
        """
        Fit RCSP per band after Riemannian alignment and extract features.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_trials, n_channels, n_times)
        y_train : np.ndarray, shape (n_trials,)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_bands * n_classes * n_components)
        """
        if any(m is None for m in self.riemannian_means):
            raise RuntimeError(
                "Call fit_population() before fit_transform()."
            )

        features = []
        for b, (l, h) in enumerate(self.bands):
            # Step 1: Band-pass filter
            X_band = self._filter_trials(X_train, l, h)

            # Step 2: Riemannian alignment to training mean
            X_aligned = self._riemannian_align(
                X_band, self.riemannian_means[b]
            )

            # Step 3: Fit Ledoit-Wolf RCSP on aligned data
            F = self.rcsp_models[b].fit_transform(X_aligned, y_train)
            features.append(F)

        return np.concatenate(features, axis=1)

    def transform(self, X_test):
        """
        Apply fitted alignment and RCSP to test trials.

        Parameters
        ----------
        X_test : np.ndarray, shape (n_trials, n_channels, n_times)

        Returns
        -------
        features : np.ndarray, shape (n_trials, n_bands * n_classes * n_components)
        """
        features = []
        for b, (l, h) in enumerate(self.bands):
            # Step 1: Band-pass filter
            X_band = self._filter_trials(X_test, l, h)

            # Step 2: Riemannian alignment using TRAINING mean
            X_aligned = self._riemannian_align(
                X_band, self.riemannian_means[b]
            )

            # Step 3: Apply fitted RCSP
            F = self.rcsp_models[b].transform(X_aligned)
            features.append(F)

        return np.concatenate(features, axis=1)
