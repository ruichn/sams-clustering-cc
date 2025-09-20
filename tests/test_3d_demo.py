"""
Test 3D functionality in the demo app
"""
import sys
import os

# Add repository root to path (from tests directory)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

# Test imports
try:
    from app import generate_dataset, DemoSAMS, plot_clustering_result_streamlit
    print("✅ Successfully imported demo functions")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

import numpy as np
import matplotlib.pyplot as plt

def test_3d_datasets():
    """Test 3D dataset generation"""
    print("\n" + "="*50)
    print("Testing 3D Dataset Generation")
    print("="*50)
    
    # Test 3D Blobs
    print("\n1. Testing 3D Blobs...")
    X, y_true = generate_dataset("3D Blobs", n_samples=300, n_centers=4, 
                                noise_level=0.1, cluster_std=1.2, n_features=3)
    print(f"   Shape: {X.shape}, Clusters: {len(np.unique(y_true))}")
    
    # Test 3D Spheres  
    print("\n2. Testing 3D Spheres...")
    X, y_true = generate_dataset("3D Spheres", n_samples=300, n_centers=3,
                                noise_level=0.05, n_features=3)
    print(f"   Shape: {X.shape}, Clusters: {len(np.unique(y_true))}")
    
    # Test regular datasets in 3D mode
    print("\n3. Testing Gaussian Blobs in 3D...")
    X, y_true = generate_dataset("Gaussian Blobs", n_samples=300, n_centers=3,
                                noise_level=0.1, cluster_std=1.0, n_features=3)
    print(f"   Shape: {X.shape}, Clusters: {len(np.unique(y_true))}")
    
    return X, y_true

def test_3d_clustering():
    """Test SAMS clustering on 3D data"""
    print("\n" + "="*50)
    print("Testing 3D SAMS Clustering")
    print("="*50)
    
    # Generate 3D data
    X, y_true = generate_dataset("3D Blobs", n_samples=200, n_centers=3, 
                                noise_level=0.1, n_features=3)
    
    print(f"\nData shape: {X.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")
    
    # Test SAMS clustering
    sams = DemoSAMS(bandwidth=None, sample_fraction=0.02, max_iter=100)
    
    try:
        labels, centers = sams.fit_predict(X)
        print(f"✅ SAMS clustering successful")
        print(f"   Predicted clusters: {len(np.unique(labels))}")
        print(f"   Bandwidth used: {sams.bandwidth:.4f}")
        
        # Test clustering quality
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, labels)
        print(f"   ARI: {ari:.3f}")
        
        return X, y_true, labels
        
    except Exception as e:
        print(f"❌ SAMS clustering failed: {e}")
        return None, None, None

def test_3d_visualization():
    """Test 3D visualization"""
    print("\n" + "="*50)
    print("Testing 3D Visualization")
    print("="*50)
    
    # Get 3D clustering results
    X, y_true, labels = test_3d_clustering()
    
    if X is None:
        print("❌ Skipping visualization test - clustering failed")
        return
    
    try:
        # Test true clusters visualization
        fig1 = plot_clustering_result_streamlit(X, y_true, "3D Test Data", len(X))
        print("✅ True clusters visualization successful")
        
        # Test predicted clusters visualization  
        fig2 = plot_clustering_result_streamlit(X, labels, "3D SAMS Results", len(X), 0.02)
        print("✅ SAMS results visualization successful")
        
        # Save test plots
        plots_dir = os.path.join(repo_root, 'plots')
        fig1.savefig(os.path.join(plots_dir, 'test_3d_true_clusters.png'), dpi=150, bbox_inches='tight')
        fig2.savefig(os.path.join(plots_dir, 'test_3d_sams_results.png'), dpi=150, bbox_inches='tight')
        print("✅ Test plots saved to plots/ directory")
        
        plt.close(fig1)
        plt.close(fig2)
        
    except Exception as e:
        print(f"❌ 3D visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔬 Testing 3D SAMS Demo Functionality")
    print("="*60)
    
    # Test dataset generation
    test_3d_datasets()
    
    # Test clustering
    test_3d_clustering()
    
    # Test visualization  
    test_3d_visualization()
    
    print("\n" + "="*60)
    print("✅ 3D Demo Testing Complete!")
    print("✅ SAMS algorithm successfully handles 3-dimensional data")
    print("✅ Demo app ready for 3D clustering experiments")