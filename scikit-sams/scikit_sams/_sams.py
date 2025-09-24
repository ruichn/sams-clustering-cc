"""
Stochastic Approximation Mean-Shift (SAMS) clustering algorithm.

This module implements the SAMS clustering algorithm following scikit-learn
API conventions for seamless integration with the scikit-learn ecosystem.
"""

import numpy as np
import warnings
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted


class SAMSClustering(ClusterMixin, BaseEstimator):
    """
    Stochastic Approximation Mean-Shift (SAMS) clustering.
    
    SAMS is a fast approximation to the mean-shift clustering algorithm that
    achieves significant speedup while maintaining clustering quality through
    stochastic sampling of the data.
    
    Parameters
    ----------
    bandwidth : float or None, default=None
        The bandwidth of the kernel. If None, the bandwidth is estimated
        using a simple heuristic based on the data.
        
    sample_fraction : float, default=0.01
        The fraction of data points to use for stochastic approximation.
        Typical values range from 0.005 to 0.02.
        
    max_iter : int, default=200
        Maximum number of iterations for the mean-shift procedure.
        
    tol : float, default=1e-4
        Tolerance for convergence. The algorithm stops when the change
        in cluster centers is below this threshold.
        
    adaptive_sampling : bool, default=True
        Whether to use adaptive sampling that adjusts sample size based
        on local density.
        
    early_stop : bool, default=True
        Whether to enable early stopping when convergence is detected.
        
    random_state : int, RandomState instance or None, default=None
        Controls the random seed for reproducible results.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
        
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
        
    n_clusters_ : int
        Number of clusters found.
        
    bandwidth_ : float
        The bandwidth used for clustering.
        
    n_iter_ : int
        Number of iterations performed.
        
    Examples
    --------
    >>> from scikit_sams import SAMSClustering
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    >>> clustering = SAMSClustering(bandwidth=0.5, sample_fraction=0.02)
    >>> clustering.fit(X)
    SAMSClustering(bandwidth=0.5, sample_fraction=0.02)
    >>> clustering.labels_
    array([0, 0, 1, ...])
    
    References
    ----------
    Hyrien, O., & Baran, R. H. (2016). Fast Nonparametric Density-Based 
    Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift 
    Algorithm. PMC5417725.
    """
    
    def __init__(self, bandwidth=None, sample_fraction=0.01, max_iter=200, 
                 tol=1e-4, adaptive_sampling=True, early_stop=True, 
                 random_state=None):
        self.bandwidth = bandwidth
        self.sample_fraction = sample_fraction
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_sampling = adaptive_sampling
        self.early_stop = early_stop
        self.random_state = random_state
    
    def _estimate_bandwidth(self, X):
        """Estimate bandwidth using Silverman's rule of thumb."""
        n_samples, n_features = X.shape
        # Silverman's rule of thumb adapted for clustering
        return 1.06 * np.std(X, axis=0).mean() * (n_samples ** (-1.0 / 5))
    
    def _gaussian_kernel(self, x, xi, h):
        """Gaussian kernel function."""
        diff = x - xi
        return np.exp(-0.5 * np.sum(diff**2, axis=1) / (h**2))
    
    def _vectorized_gradient_batch(self, modes_batch, sample_data, h):
        """Vectorized gradient computation for performance."""
        if len(sample_data) == 0:
            return modes_batch.copy()  # Return unchanged if no samples
        
        # Compute all pairwise distances at once
        sq_dists = cdist(modes_batch, sample_data, metric='sqeuclidean')
        weights = np.exp(-0.5 * sq_dists / (h**2))
        
        # Vectorized weighted means - proper mean-shift update
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weight_sums = np.maximum(weight_sums, 1e-10)  # Avoid division by zero
        
        # Compute new mode positions (weighted average of sample points)
        new_modes = np.zeros_like(modes_batch)
        for i in range(len(modes_batch)):
            if weight_sums[i, 0] > 1e-10:
                new_modes[i] = np.sum(weights[i:i+1].T * sample_data, axis=0) / weight_sums[i, 0]
            else:
                new_modes[i] = modes_batch[i]  # Keep original position if no influence
        
        return new_modes
    
    def _adaptive_sample_size(self, X, base_fraction):
        """Compute adaptive sample size based on data characteristics."""
        if not self.adaptive_sampling:
            return max(1, int(len(X) * base_fraction))
        
        n_samples, n_features = X.shape
        
        # Adjust sample size based on dimensionality and data size
        dim_factor = min(2.0, 1.0 + n_features / 10.0)
        size_factor = min(2.0, np.log10(n_samples) / 2.0)
        
        adjusted_fraction = base_fraction * dim_factor * size_factor
        return max(1, min(n_samples, int(adjusted_fraction * n_samples)))
    
    def fit(self, X, y=None):
        """
        Fit the SAMS clustering algorithm.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to cluster.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Input validation - use validate_data to set n_features_in_
        X = self._validate_data(X, accept_sparse=False, dtype=[np.float64, np.float32])
        random_state = check_random_state(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Parameter validation
        if self.sample_fraction <= 0 or self.sample_fraction > 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        
        # Estimate bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth_ = self._estimate_bandwidth(X)
        else:
            if self.bandwidth <= 0:
                raise ValueError("bandwidth must be positive")
            self.bandwidth_ = self.bandwidth
        
        # Initialize modes at data points
        modes = X.copy()
        
        # Compute adaptive sample size
        sample_size = self._adaptive_sample_size(X, self.sample_fraction)
        
        # Main SAMS iteration
        for iteration in range(self.max_iter):
            # Stochastic sampling with replacement for small sample sizes
            replace = sample_size >= n_samples
            sample_indices = random_state.choice(
                n_samples, size=sample_size, replace=replace
            )
            sample_data = X[sample_indices]
            
            # Batch gradient computation
            new_modes = self._vectorized_gradient_batch(
                modes, sample_data, self.bandwidth_
            )
            
            # Check for convergence
            if self.early_stop:
                max_shift = np.max(np.linalg.norm(new_modes - modes, axis=1))
                if max_shift < self.tol:
                    break
            
            modes = new_modes
        
        self.n_iter_ = iteration + 1
        
        # Merge nearby modes (within bandwidth/2)
        merge_threshold = self.bandwidth_ / 2.0
        mode_distances = cdist(modes, modes)
        
        # Find clusters of nearby modes
        mode_labels = np.arange(len(modes))
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                if mode_distances[i, j] < merge_threshold:
                    # Merge mode j into mode i's cluster
                    mode_labels[mode_labels == mode_labels[j]] = mode_labels[i]
        
        # Get unique mode clusters and their representatives
        unique_mode_labels = np.unique(mode_labels)
        final_centers = []
        
        for label in unique_mode_labels:
            # Use the centroid of merged modes as final center
            mask = mode_labels == label
            center = np.mean(modes[mask], axis=0)
            final_centers.append(center)
        
        final_centers = np.array(final_centers)
        
        # Assign data points to nearest final centers
        distances = cdist(X, final_centers)
        self.labels_ = np.argmin(distances, axis=1)
        self.cluster_centers_ = final_centers
        self.n_clusters_ = len(final_centers)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=False, dtype=[np.float64, np.float32], reset=False)
        
        # Assign to nearest cluster center
        distances = cdist(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X, y=None):
        """
        Fit the algorithm and predict cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to cluster.
            
        y : Ignored
            Not used, present for API consistency.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        return self.fit(X, y).labels_
    
    def __sklearn_tags__(self):
        """Provide sklearn tags for the estimator (sklearn 1.6+ compatibility)."""
        tags = super().__sklearn_tags__()
        tags.requires_positive_X = False
        tags.requires_fit = True
        tags.no_validation = False
        return tags
    
    def _more_tags(self):
        """Provide additional tags for the estimator (backward compatibility)."""
        return {
            'requires_positive_X': False,
            'requires_fit': True,
            'no_validation': False,
            '_xfail_checks': {}
        }