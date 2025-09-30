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
            bandwidth = sample_std * (n_samples**(-1.0 / (n_features + 4)))
            
            # Ensure bandwidth is within reasonable bounds
            self.bandwidth = max(min(bandwidth, 10.0), 1e-6)
            
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
            # Vectorized mode update - compute all modes simultaneously
            # Compute pairwise distances: modes vs data points
            # modes: (n_samples, n_features), X: (n_samples, n_features)
            # Result: (n_samples, n_samples) where [i,j] = distance from mode_i to point_j
            
            diffs = modes[:, None, :] - X[None, :, :]  # Shape: (n_modes, n_points, n_features)
            sq_dists = np.sum(diffs**2, axis=2)        # Shape: (n_modes, n_points)
            
            # Gaussian kernel weights for all mode-point pairs  
            # Simplified weight calculation to avoid overflow
            weights = np.exp(-0.5 * sq_dists / (self.bandwidth**2))  # Shape: (n_modes, n_points)
            
            # Weighted mean shift update for all modes simultaneously
            weight_sums = np.sum(weights, axis=1, keepdims=True)  # Shape: (n_modes, 1)
            weight_sums = np.maximum(weight_sums, 1e-10)  # Avoid division by zero
            
            # Compute weighted averages: weights @ X for each mode
            # weights: (n_modes, n_points), X: (n_points, n_features)
            # Result: (n_modes, n_features)
            weighted_sums = np.sum(weights[:, :, None] * X[None, :, :], axis=1)  # (n_modes, n_features)
            new_modes = weighted_sums / weight_sums  # (n_modes, n_features)
            
            # Track maximum shift for convergence
            shifts = np.linalg.norm(new_modes - modes, axis=1)  # (n_modes,)
            max_shift = np.max(shifts)
            
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
        
        # Vectorized distance computation - all pairwise distances at once
        # modes: (n_points, n_features)
        diffs = modes[:, None, :] - modes[None, :, :]  # Shape: (n_points, n_points, n_features)
        distances = np.linalg.norm(diffs, axis=2)      # Shape: (n_points, n_points)
        
        # Create adjacency matrix for clustering
        # Points are connected if distance <= threshold
        adjacency = distances <= distance_threshold
        
        # Assign cluster labels using connected components
        labels = np.full(n_points, -1)
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] == -1:
                # Find all points connected to point i (breadth-first search)
                stack = [i]
                labels[i] = cluster_id
                
                while stack:
                    current = stack.pop()
                    # Find all unvisited neighbors
                    neighbors = np.where((adjacency[current]) & (labels == -1))[0]
                    for neighbor in neighbors:
                        labels[neighbor] = cluster_id
                        stack.append(neighbor)
                
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
    from sklearn.cluster import MeanShift as SklearnMeanShift, estimate_bandwidth
    from sklearn.datasets import make_blobs
    import time
    
    # Generate test data
    true_clusters = 3
    true_features = 64

    X, _ = make_blobs(n_samples=1000, centers=true_clusters, n_features=true_features, 
                          random_state=42, cluster_std=1.5)
    
    print("=" * 60)
    print("COMPARISON: StandardMeanShift vs sklearn.cluster.MeanShift")
    print("=" * 60)
    
    print(f"   True Clusters: {true_clusters}")
    print(f"   Dimensions: {true_features}")

    # Test our implementation
    print("\n1. Our StandardMeanShift:")
    our_ms = StandardMeanShift(bandwidth=None)
    start_time = time.time()
    _, our_centers = our_ms.fit_predict(X)
    our_time = time.time() - start_time
    
    print(f"   Time: {our_time:.3f}s")
    print(f"   Clusters: {len(our_centers)}")
    print(f"   Bandwidth: {our_ms.bandwidth:.4f}")
    
    # Test sklearn implementation
    print("\n2. sklearn MeanShift:")
    
    # Use our bandwidth with validation to prevent numerical warnings
    validated_bandwidth = our_ms.bandwidth
    if not np.isfinite(validated_bandwidth) or validated_bandwidth <= 0:
        validated_bandwidth = None  # Fall back to automatic
    else:
        validated_bandwidth = max(min(validated_bandwidth, 10.0), 1e-3)  # Clip to safe range
    
    sklearn_ms = SklearnMeanShift(bandwidth=validated_bandwidth)
    start_time = time.time()
    sklearn_ms.fit_predict(X)
    sklearn_time = time.time() - start_time
    
    print(f"   Time: {sklearn_time:.3f}s")
    print(f"   Clusters: {len(sklearn_ms.cluster_centers_)}")
    print(f"   Bandwidth: {validated_bandwidth if validated_bandwidth else estimate_bandwidth(X):.4f}")
    print(f"   Speedup: {our_time / sklearn_time:.1f}x faster")
    
    print(f"\nNote: sklearn MeanShift uses optimizations (KD-trees, early stopping)")
    print(f"      that make it faster than the educational O(n²) implementation.")


if __name__ == "__main__":
    compare_implementations()