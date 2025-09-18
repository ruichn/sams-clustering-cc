import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import time

class SAMS_Clustering:
    """
    Stochastic Approximation Mean-Shift (SAMS) Algorithm
    
    Fast nonparametric density-based clustering of large datasets using 
    stochastic approximation as described in Hyrien & Baran (2017).
    """
    
    def __init__(self, bandwidth=None, sample_fraction=0.01, max_iter=1000, 
                 tol=1e-4, kernel='gaussian', alpha1=None, alpha2=None):
        """
        Initialize SAMS clustering algorithm.
        
        Parameters:
        -----------
        bandwidth : float or None
            Bandwidth parameter for kernel density estimation. If None, 
            will be computed using data-driven method from paper.
        sample_fraction : float
            Fraction of data to sample at each iteration (0.1% - 1% recommended)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        kernel : str
            Kernel type ('gaussian' or 'epanechnikov')
        alpha1 : float or None
            Parameter for data-driven bandwidth selection. If None, uses σ^n^(-1/(p+4))
        alpha2 : float or None
            Parameter for pilot density sensitivity. If None, uses 1/p (Breiman et al.)
        """
        self.bandwidth = bandwidth
        self.sample_fraction = sample_fraction
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
    def gaussian_kernel(self, x, xi, h):
        """Gaussian kernel function"""
        diff = x - xi
        return np.exp(-0.5 * np.sum(diff**2, axis=1) / (h**2))
    
    def epanechnikov_kernel(self, x, xi, h):
        """Epanechnikov kernel function"""
        diff = x - xi
        dist_sq = np.sum(diff**2, axis=1) / (h**2)
        mask = dist_sq <= 1
        kernel_vals = np.zeros(len(dist_sq))
        kernel_vals[mask] = 0.75 * (1 - dist_sq[mask])
        return kernel_vals
    
    def kernel_function(self, x, xi, h):
        """Apply selected kernel function"""
        if self.kernel == 'gaussian':
            return self.gaussian_kernel(x, xi, h)
        elif self.kernel == 'epanechnikov':
            return self.epanechnikov_kernel(x, xi, h)
    
    def gradient_approximation(self, x, sample_data, h):
        """
        Compute gradient approximation using sampled data points.
        
        Parameters:
        -----------
        x : array-like, shape (d,)
            Current point
        sample_data : array-like, shape (n_sample, d)
            Sampled data points
        h : float
            Bandwidth parameter
            
        Returns:
        --------
        gradient : array-like, shape (d,)
            Gradient approximation
        """
        if len(sample_data) == 0:
            return np.zeros_like(x)
        
        # Reshape x for broadcasting
        x = x.reshape(1, -1)
        
        # Compute kernel weights
        weights = self.kernel_function(x, sample_data, h)
        
        # Avoid division by zero
        if np.sum(weights) == 0:
            return np.zeros(x.shape[1])
        
        # Weighted mean shift vector
        weighted_points = sample_data * weights.reshape(-1, 1)
        gradient = np.sum(weighted_points, axis=0) / np.sum(weights) - x.flatten()
        
        return gradient
    
    def compute_pilot_density(self, X, h_pilot):
        """
        Compute pilot density estimate for data-driven bandwidth selection.
        """
        n_samples, _ = X.shape
        pilot_densities = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Use kernel density estimation for pilot density
            weights = self.kernel_function(X[i].reshape(1, -1), X, h_pilot)
            pilot_densities[i] = np.mean(weights)
        
        return pilot_densities
    
    def compute_data_driven_bandwidth(self, X):
        """
        Compute data-driven bandwidth according to Hyrien & Baran (2017).
        
        Formula: hi = λi * α1
        where λi = (β^ / f~(yi))^α2
        and β^ = geometric mean of pilot densities
        """
        n_samples, n_features = X.shape
        
        # Set default parameters if not provided
        if self.alpha1 is None:
            # Rule of thumb: σ^n^(-1/(p+4))
            sample_var = np.var(X, axis=0).mean()
            self.alpha1 = (sample_var * n_samples)**(-1.0 / (n_features + 4))
        
        if self.alpha2 is None:
            # Breiman et al. recommendation: 1/p
            self.alpha2 = 1.0 / n_features
        
        # Use simple rule-of-thumb for pilot bandwidth
        h_pilot = 1.06 * np.std(X, axis=0).mean() * (n_samples**(-1.0/5))
        
        # Compute pilot density estimates
        pilot_densities = self.compute_pilot_density(X, h_pilot)
        
        # Avoid zero densities
        pilot_densities = np.maximum(pilot_densities, 1e-10)
        
        # Compute geometric mean (β^)
        beta_hat = np.exp(np.mean(np.log(pilot_densities)))
        
        # Compute adaptive bandwidth multipliers (λi)
        lambda_i = (beta_hat / pilot_densities) ** self.alpha2
        
        # Compute final bandwidths
        bandwidths = lambda_i * self.alpha1
        
        # Use median bandwidth for global bandwidth
        return np.median(bandwidths)
    
    def fit_predict(self, X):
        """
        Perform SAMS clustering on data X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        labels : array-like, shape (n_samples,)
            Cluster labels
        modes : array-like, shape (n_modes, n_features)
            Cluster centers (modes)
        """
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # Compute data-driven bandwidth if not provided
        if self.bandwidth is None:
            self.bandwidth = self.compute_data_driven_bandwidth(X)
            print(f"Data-driven bandwidth selected: {self.bandwidth:.4f}")
        
        # Initialize modes for each data point
        modes = X.copy()
        
        # Sample size for each iteration
        sample_size = max(1, int(self.sample_fraction * n_samples))
        
        print(f"Starting SAMS clustering with {n_samples} points")
        print(f"Sample size per iteration: {sample_size}")
        
        # Stochastic approximation iterations
        for iteration in range(self.max_iter):
            # Sample random subset of data
            sample_indices = np.random.choice(n_samples, sample_size, replace=False)
            sample_data = X[sample_indices]
            
            # Update each mode using stochastic approximation
            new_modes = np.zeros_like(modes)
            max_shift = 0
            
            for i in range(n_samples):
                # Compute gradient approximation
                gradient = self.gradient_approximation(modes[i], sample_data, 
                                                     self.bandwidth)
                
                # Robbins-Monro update with decreasing step size
                step_size = 1.0 / (iteration + 1)**0.6
                new_modes[i] = modes[i] + step_size * gradient
                
                # Track maximum shift for convergence
                shift = np.linalg.norm(new_modes[i] - modes[i])
                max_shift = max(max_shift, shift)
            
            modes = new_modes
            
            # Check convergence
            if max_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Cluster assignment: group points with similar modes
        labels = self._assign_clusters(modes)
        unique_modes = self._get_unique_modes(modes, labels)
        
        return labels, unique_modes
    
    def _assign_clusters(self, modes, distance_threshold=None):
        """Assign cluster labels based on mode similarity"""
        if distance_threshold is None:
            distance_threshold = self.bandwidth / 2
        
        n_points = len(modes)
        labels = np.full(n_points, -1)
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] == -1:  # Unassigned point
                # Start new cluster
                labels[i] = cluster_id
                
                # Find all points within distance threshold
                for j in range(i + 1, n_points):
                    if labels[j] == -1:
                        distance = np.linalg.norm(modes[i] - modes[j])
                        if distance <= distance_threshold:
                            labels[j] = cluster_id
                
                cluster_id += 1
        
        return labels
    
    def _get_unique_modes(self, modes, labels):
        """Get representative modes for each cluster"""
        unique_labels = np.unique(labels)
        unique_modes = []
        
        for label in unique_labels:
            cluster_modes = modes[labels == label]
            # Use centroid as representative mode
            unique_modes.append(np.mean(cluster_modes, axis=0))
        
        return np.array(unique_modes)


class StandardMeanShift:
    """
    Standard Mean-Shift algorithm for comparison.
    """
    
    def __init__(self, bandwidth=1.0, max_iter=1000, tol=1e-4):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
    
    def gaussian_kernel(self, x, xi, h):
        """Gaussian kernel function"""
        diff = x - xi
        return np.exp(-0.5 * np.sum(diff**2, axis=1) / (h**2))
    
    def fit_predict(self, X):
        """Standard mean-shift clustering"""
        X = np.array(X)
        n_samples, n_features = X.shape
        modes = X.copy()
        
        print(f"Starting standard Mean-Shift with {n_samples} points")
        
        for iteration in range(self.max_iter):
            new_modes = np.zeros_like(modes)
            max_shift = 0
            
            for i in range(n_samples):
                # Use all data points for gradient computation
                weights = self.gaussian_kernel(modes[i].reshape(1, -1), X, 
                                             self.bandwidth)
                
                if np.sum(weights) > 0:
                    weighted_points = X * weights.reshape(-1, 1)
                    new_modes[i] = np.sum(weighted_points, axis=0) / np.sum(weights)
                else:
                    new_modes[i] = modes[i]
                
                shift = np.linalg.norm(new_modes[i] - modes[i])
                max_shift = max(max_shift, shift)
            
            modes = new_modes
            
            if max_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Simple clustering based on mode proximity
        labels = self._assign_clusters(modes)
        unique_modes = self._get_unique_modes(modes, labels)
        
        return labels, unique_modes
    
    def _assign_clusters(self, modes, distance_threshold=None):
        """Assign cluster labels based on mode similarity"""
        if distance_threshold is None:
            distance_threshold = self.bandwidth / 2
        
        n_points = len(modes)
        labels = np.full(n_points, -1)
        cluster_id = 0
        
        for i in range(n_points):
            if labels[i] == -1:
                labels[i] = cluster_id
                for j in range(i + 1, n_points):
                    if labels[j] == -1:
                        distance = np.linalg.norm(modes[i] - modes[j])
                        if distance <= distance_threshold:
                            labels[j] = cluster_id
                cluster_id += 1
        
        return labels
    
    def _get_unique_modes(self, modes, labels):
        """Get representative modes for each cluster"""
        unique_labels = np.unique(labels)
        unique_modes = []
        
        for label in unique_labels:
            cluster_modes = modes[labels == label]
            unique_modes.append(np.mean(cluster_modes, axis=0))
        
        return np.array(unique_modes)


def generate_test_data(n_samples=1000, dataset_type='blobs'):
    """Generate test datasets as described in the paper"""
    
    if dataset_type == 'blobs':
        # Generate well-separated Gaussian clusters
        X, y_true = make_blobs(n_samples=n_samples, centers=4, n_features=2,
                              random_state=42, cluster_std=1.5, center_box=(-10, 10))
        
    elif dataset_type == 'circles':
        # Generate concentric circles
        X, y_true = make_circles(n_samples=n_samples, noise=0.1, factor=0.6,
                                random_state=42)
        
    elif dataset_type == 'mixed':
        # Generate mixed density clusters
        np.random.seed(42)
        
        # High density cluster
        cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 
                                                size=n_samples//3)
        
        # Medium density cluster
        cluster2 = np.random.multivariate_normal([5, 5], [[2, 0], [0, 2]], 
                                                size=n_samples//3)
        
        # Low density cluster
        cluster3 = np.random.multivariate_normal([-3, 4], [[0.5, 0], [0, 0.5]], 
                                                size=n_samples//3)
        
        X = np.vstack([cluster1, cluster2, cluster3])
        y_true = np.hstack([np.zeros(n_samples//3), 
                           np.ones(n_samples//3), 
                           np.full(n_samples//3, 2)])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y_true


if __name__ == "__main__":
    # Test the implementation
    print("Testing SAMS Clustering Implementation")
    print("=" * 50)
    
    # Generate test data
    X, y_true = generate_test_data(n_samples=500, dataset_type='blobs')
    
    print(f"Generated {len(X)} data points with {X.shape[1]} features")
    print(f"True number of clusters: {len(np.unique(y_true))}")