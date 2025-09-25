"""
Standard Mean-Shift Implementation for Comparison with SAMS

This module provides a clean, educational implementation of the standard mean-shift
algorithm for fair comparison with SAMS clustering. It uses identical kernels,
bandwidth estimation, and convergence criteria to ensure valid performance comparisons.

Reference:
- Fukunaga, K., & Hostetler, L. (1975). The estimation of the gradient of a density 
  function, with applications in pattern recognition. IEEE Transactions on information theory.
- Cheng, Y. (1995). Mean shift, mode seeking, and clustering. IEEE transactions on 
  pattern analysis and machine intelligence.
"""

import numpy as np
from scipy.spatial.distance import cdist


class StandardMeanShift:
    """
    Standard Mean-Shift algorithm implementation for educational and comparison purposes.
    
    This implementation provides:
    - Clear, readable code for understanding the algorithm
    - Identical kernel and bandwidth computation to SAMS for fair comparison
    - Educational comments explaining each step
    - O(n²) complexity characteristic of standard mean-shift
    """
    
    def __init__(self, bandwidth=None, max_iter=300, tol=1e-4):
        """
        Initialize Standard Mean-Shift clusterer.
        
        Parameters:
        -----------
        bandwidth : float, optional
            Kernel bandwidth. If None, will be estimated from data.
        max_iter : int, default=300
            Maximum number of iterations.
        tol : float, default=1e-4
            Convergence tolerance (maximum shift between iterations).
        """
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
    
    def gaussian_kernel(self, x, xi, h):
        """
        Gaussian kernel function - identical to SAMS implementation.
        
        K(x) = exp(-0.5 * ||x-xi||² / h²)
        
        Parameters:
        -----------
        x : array-like, shape (1, n_features)
            Query point
        xi : array-like, shape (n_samples, n_features)  
            Data points
        h : float
            Bandwidth parameter
            
        Returns:
        --------
        weights : array-like, shape (n_samples,)
            Kernel weights for each data point
        """
        diff = x - xi
        return np.exp(-0.5 * np.sum(diff**2, axis=1) / (h**2))
    
    def compute_bandwidth(self, X):
        """
        Compute bandwidth using Silverman's rule - identical to SAMS.
        
        This ensures fair comparison by using the same bandwidth estimation
        method as SAMS clustering.
        """
        if self.bandwidth is None:
            n_samples, n_features = X.shape
            
            # Silverman's rule of thumb
            sample_std = np.std(X, axis=0).mean()
            self.bandwidth = sample_std * (n_samples**(-1.0 / (n_features + 4)))
            
        return self.bandwidth
    
    def fit_predict(self, X):
        """
        Perform standard mean-shift clustering.
        
        Algorithm:
        1. Initialize each data point as a mode
        2. For each mode, iteratively shift toward higher density:
           - Compute weighted mean of nearby points (using Gaussian kernel)
           - Move mode to weighted mean location
           - Repeat until convergence
        3. Assign clusters based on final mode positions
        
        Time Complexity: O(n² × iterations)
        - Each iteration processes all n modes
        - Each mode computation requires O(n) kernel evaluations
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data points
            
        Returns:
        --------
        labels : array-like, shape (n_samples,)
            Cluster labels for each point
        centers : array-like, shape (n_clusters, n_features)
            Final cluster centers (unique modes)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Compute bandwidth
        self.bandwidth = self.compute_bandwidth(X)
        
        # Initialize modes - each data point starts as its own mode
        modes = X.copy()
        
        print(f"Starting Standard Mean-Shift with {n_samples} points")
        print(f"Bandwidth: {self.bandwidth:.4f}")
        
        # Mean-shift iterations
        for iteration in range(self.max_iter):
            new_modes = np.zeros_like(modes)
            max_shift = 0
            
            # Update each mode (this is the O(n²) part)
            for i in range(n_samples):
                # Compute kernel weights for mode i against all data points
                weights = self.gaussian_kernel(modes[i].reshape(1, -1), X, self.bandwidth)
                
                # Weighted mean shift update
                if np.sum(weights) > 0:
                    # New mode position = weighted average of all points
                    weighted_points = X * weights.reshape(-1, 1)
                    new_modes[i] = np.sum(weighted_points, axis=0) / np.sum(weights)
                else:
                    # No nearby points - mode stays in place
                    new_modes[i] = modes[i]
                
                # Track maximum shift for convergence
                shift = np.linalg.norm(new_modes[i] - modes[i])
                max_shift = max(max_shift, shift)
            
            modes = new_modes
            
            # Check convergence
            if max_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        else:
            print(f"Reached maximum iterations ({self.max_iter})")
        
        # Assign cluster labels and find unique modes
        labels = self._assign_clusters(modes)
        centers = self._get_cluster_centers(modes, labels)
        
        return labels, centers
    
    def _assign_clusters(self, modes, distance_threshold=None):
        """
        Assign cluster labels based on mode proximity.
        
        Modes that are close together (within distance_threshold) are 
        considered to belong to the same cluster.
        """
        if distance_threshold is None:
            distance_threshold = self.bandwidth / 2
        
        n_points = len(modes)
        labels = np.full(n_points, -1)
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] == -1:
                labels[i] = cluster_id
                
                # Find nearby modes and assign to same cluster
                for j in range(i + 1, n_points):
                    if labels[j] == -1:
                        distance = np.linalg.norm(modes[i] - modes[j])
                        if distance <= distance_threshold:
                            labels[j] = cluster_id
                
                cluster_id += 1
        
        return labels
    
    def _get_cluster_centers(self, modes, labels):
        """
        Compute final cluster centers as the mean of modes in each cluster.
        """
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            cluster_modes = modes[labels == label]
            centers.append(np.mean(cluster_modes, axis=0))
        
        return np.array(centers)


def compare_implementations():
    """
    Educational function to compare StandardMeanShift vs sklearn.cluster.MeanShift
    
    This function demonstrates the differences in:
    - Performance (our O(n²) vs sklearn's optimizations)
    - Results (should be similar with same bandwidth)
    - Implementation details
    """
    from sklearn.cluster import MeanShift as SklearnMeanShift
    from sklearn.datasets import make_blobs
    import time
    
    # Generate test data
    X, y_true = make_blobs(n_samples=200, centers=3, n_features=2, 
                          random_state=42, cluster_std=1.5)
    
    print("=" * 60)
    print("COMPARISON: StandardMeanShift vs sklearn.cluster.MeanShift")
    print("=" * 60)
    
    # Test our implementation
    print("\n1. Our StandardMeanShift:")
    our_ms = StandardMeanShift(bandwidth=None)
    start_time = time.time()
    our_labels, our_centers = our_ms.fit_predict(X)
    our_time = time.time() - start_time
    
    print(f"   Time: {our_time:.3f}s")
    print(f"   Clusters: {len(our_centers)}")
    print(f"   Bandwidth: {our_ms.bandwidth:.4f}")
    
    # Test sklearn implementation
    print("\n2. sklearn MeanShift:")
    sklearn_ms = SklearnMeanShift(bandwidth=our_ms.bandwidth)  # Use same bandwidth
    start_time = time.time()
    sklearn_labels = sklearn_ms.fit_predict(X)
    sklearn_time = time.time() - start_time
    
    print(f"   Time: {sklearn_time:.3f}s")
    print(f"   Clusters: {len(sklearn_ms.cluster_centers_)}")
    print(f"   Bandwidth: {our_ms.bandwidth:.4f}")
    print(f"   Speedup: {our_time / sklearn_time:.1f}x faster")
    
    print(f"\nNote: sklearn MeanShift uses optimizations (KD-trees, early stopping)")
    print(f"      that make it faster than the educational O(n²) implementation.")


if __name__ == "__main__":
    compare_implementations()