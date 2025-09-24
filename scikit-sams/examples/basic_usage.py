"""
Basic usage example for sklearn-sams.

This example demonstrates the basic functionality of the SAMSClustering
algorithm with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import MeanShift
import time

from scikit_sams import SAMSClustering


def basic_clustering_example():
    """Demonstrate basic SAMS clustering."""
    print("=" * 60)
    print("BASIC SAMS CLUSTERING EXAMPLE")
    print("=" * 60)
    
    # Generate synthetic data
    n_samples = 1000
    n_centers = 4
    X, y_true = make_blobs(
        n_samples=n_samples, 
        centers=n_centers, 
        cluster_std=1.0,
        random_state=42
    )
    
    print(f"Generated {n_samples} samples with {n_centers} true clusters")
    
    # Configure SAMS clustering
    sams = SAMSClustering(
        bandwidth=1.0,
        sample_fraction=0.02,
        max_iter=200,
        random_state=42
    )
    
    # Fit and predict
    start_time = time.time()
    labels = sams.fit_predict(X)
    sams_time = time.time() - start_time
    
    # Evaluate results
    ari = adjusted_rand_score(y_true, labels)
    silhouette = silhouette_score(X, labels)
    
    print(f"\\nSAMS Results:")
    print(f"  Clusters found: {sams.n_clusters_}")
    print(f"  Iterations: {sams.n_iter_}")
    print(f"  Time: {sams_time:.3f} seconds")
    print(f"  ARI: {ari:.3f}")
    print(f"  Silhouette Score: {silhouette:.3f}")
    
    return X, y_true, labels, sams


def performance_comparison():
    """Compare SAMS with standard Mean-Shift."""
    print("\\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: SAMS vs Mean-Shift")
    print("=" * 60)
    
    # Generate data
    X, y_true = make_blobs(n_samples=500, centers=3, random_state=42)
    
    # SAMS clustering
    sams = SAMSClustering(bandwidth=1.0, random_state=42)
    start_time = time.time()
    sams_labels = sams.fit_predict(X)
    sams_time = time.time() - start_time
    sams_ari = adjusted_rand_score(y_true, sams_labels)
    
    # Mean-Shift clustering
    ms = MeanShift(bandwidth=1.0)
    start_time = time.time()
    ms_labels = ms.fit_predict(X)
    ms_time = time.time() - start_time
    ms_ari = adjusted_rand_score(y_true, ms_labels)
    
    # Results
    speedup = ms_time / sams_time
    
    print(f"SAMS Results:")
    print(f"  Time: {sams_time:.3f}s")
    print(f"  Clusters: {sams.n_clusters_}")
    print(f"  ARI: {sams_ari:.3f}")
    
    print(f"\\nMean-Shift Results:")
    print(f"  Time: {ms_time:.3f}s") 
    print(f"  Clusters: {len(np.unique(ms_labels))}")
    print(f"  ARI: {ms_ari:.3f}")
    
    print(f"\\nSpeedup: {speedup:.1f}x")
    print(f"Quality retention: {(sams_ari/ms_ari*100):.1f}%")


def visualize_results(X, y_true, labels, sams):
    """Visualize clustering results."""
    if X.shape[1] != 2:
        print("Skipping visualization (data not 2D)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True clusters
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', alpha=0.7)
    axes[0].set_title('True Clusters')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # SAMS results
    axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.7)
    axes[1].scatter(
        sams.cluster_centers_[:, 0], 
        sams.cluster_centers_[:, 1],
        c='red', marker='x', s=200, linewidths=3,
        label='Cluster Centers'
    )
    axes[1].set_title(f'SAMS Clustering ({sams.n_clusters_} clusters)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def parameter_sensitivity():
    """Demonstrate parameter sensitivity."""
    print("\\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
    
    # Test different sample fractions
    sample_fractions = [0.005, 0.01, 0.02, 0.05]
    
    print("Sample Fraction Analysis:")
    print("Fraction | Clusters | ARI    | Time(s)")
    print("-" * 35)
    
    for frac in sample_fractions:
        sams = SAMSClustering(
            bandwidth=1.0, 
            sample_fraction=frac, 
            random_state=42
        )
        
        start_time = time.time()
        labels = sams.fit_predict(X)
        elapsed = time.time() - start_time
        
        ari = adjusted_rand_score(y_true, labels)
        
        print(f"{frac:8.3f} | {sams.n_clusters_:8d} | {ari:6.3f} | {elapsed:7.3f}")


if __name__ == "__main__":
    # Run examples
    X, y_true, labels, sams = basic_clustering_example()
    performance_comparison()
    parameter_sensitivity()
    
    # Visualization (requires matplotlib)
    try:
        visualize_results(X, y_true, labels, sams)
    except ImportError:
        print("\\nMatplotlib not available - skipping visualization")
    
    print("\\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)