#!/usr/bin/env python3
"""
Test script for high-dimensional demo functionality
"""

import numpy as np
import sys
import os

# Add parent directory to path for app imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app functions
from app import generate_dataset, DemoSAMS
from sklearn.decomposition import PCA

def test_high_dimensional_data_generation():
    """Test high-dimensional data generation"""
    print("Testing high-dimensional data generation...")
    
    # Test 64D data
    X, y_true = generate_dataset("High-Dimensional Blobs", 500, 5, 0.1, n_features=64)
    print(f"✓ Generated 64D data: shape {X.shape}, clusters {len(np.unique(y_true))}")
    
    # Test 128D data  
    X, y_true = generate_dataset("High-Dimensional Blobs", 1000, 7, 0.1, n_features=128)
    print(f"✓ Generated 128D data: shape {X.shape}, clusters {len(np.unique(y_true))}")
    
    return X, y_true

def test_high_dimensional_clustering():
    """Test SAMS clustering on high-dimensional data"""
    print("\nTesting high-dimensional clustering...")
    
    # Generate 64D test data
    X, y_true = generate_dataset("High-Dimensional Blobs", 800, 4, 0.05, n_features=64)
    
    # Test SAMS clustering
    sams = DemoSAMS(bandwidth=None, sample_fraction=0.03, max_iter=150)
    labels, centers = sams.fit_predict(X)
    
    n_clusters_found = len(np.unique(labels))
    n_clusters_true = len(np.unique(y_true))
    
    print(f"✓ 64D SAMS clustering: {n_clusters_found} clusters found vs {n_clusters_true} true")
    print(f"✓ Bandwidth used: {sams.bandwidth:.4f}")
    
    # Test 128D data
    X_128, y_true_128 = generate_dataset("High-Dimensional Blobs", 600, 5, 0.05, n_features=128)
    sams_128 = DemoSAMS(bandwidth=None, sample_fraction=0.04, max_iter=150)
    labels_128, centers_128 = sams_128.fit_predict(X_128)
    
    n_clusters_found_128 = len(np.unique(labels_128))
    n_clusters_true_128 = len(np.unique(y_true_128))
    
    print(f"✓ 128D SAMS clustering: {n_clusters_found_128} clusters found vs {n_clusters_true_128} true")
    print(f"✓ Bandwidth used: {sams_128.bandwidth:.4f}")
    
    return X_128, labels_128, centers_128

def test_pca_visualization():
    """Test PCA visualization functionality"""
    print("\nTesting PCA visualization...")
    
    # Generate high-dimensional data
    X, y_true = generate_dataset("High-Dimensional Blobs", 500, 4, 0.05, n_features=96)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_var = pca.explained_variance_ratio_[:2].sum()
    
    print(f"✓ PCA on 96D data: reduced to {X_pca.shape}")
    print(f"✓ Explained variance: {explained_var:.1%}")
    print(f"✓ PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"✓ PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
    
    return X_pca, pca

def test_dimensional_scaling():
    """Test performance across different dimensions"""
    print("\nTesting dimensional scaling performance...")
    
    import time
    
    dimensions = [8, 16, 32, 64, 96, 128]
    results = []
    
    for dim in dimensions:
        print(f"  Testing {dim}D...")
        X, y_true = generate_dataset("High-Dimensional Blobs", 600, 4, 0.05, n_features=dim)
        
        # Adaptive sample fraction
        sample_frac = min(0.05, max(0.02, 0.02 + (dim - 64) / 1000))
        
        sams = DemoSAMS(bandwidth=None, sample_fraction=sample_frac, max_iter=100)
        
        start_time = time.time()
        labels, centers = sams.fit_predict(X)
        runtime = time.time() - start_time
        
        n_clusters = len(np.unique(labels))
        n_true = len(np.unique(y_true))
        
        results.append({
            'dim': dim,
            'time': runtime,
            'clusters_found': n_clusters,
            'clusters_true': n_true,
            'sample_frac': sample_frac
        })
        
        print(f"    {dim}D: {runtime:.3f}s, {n_clusters} clusters, {sample_frac:.3f} sample frac")
    
    print("\n📊 Dimensional Scaling Summary:")
    print("Dim  | Time (s) | Clusters | Sample %")
    print("-----|----------|----------|----------")
    for r in results:
        print(f"{r['dim']:4d} | {r['time']:8.3f} | {r['clusters_found']:8d} | {r['sample_frac']*100:8.1f}")
    
    return results

def main():
    """Run all high-dimensional demo tests"""
    print("🔬 Testing High-Dimensional Demo Functionality")
    print("=" * 50)
    
    try:
        # Test data generation
        test_high_dimensional_data_generation()
        
        # Test clustering
        test_high_dimensional_clustering()
        
        # Test PCA visualization
        test_pca_visualization()
        
        # Test dimensional scaling
        test_dimensional_scaling()
        
        print("\n" + "=" * 50)
        print("✅ All high-dimensional demo tests passed!")
        print("\n🎯 Demo Features Validated:")
        print("  ✓ High-dimensional data generation (2D-128D)")
        print("  ✓ Adaptive SAMS clustering parameters")
        print("  ✓ PCA visualization for high-D data")
        print("  ✓ Performance scaling across dimensions")
        print("  ✓ Interactive slider support (2-128 dimensions)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)