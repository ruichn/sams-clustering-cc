"""
Quick test to verify SAMS clustering works with 3D data
"""
import numpy as np
import sys
import os

# Add src directory to path (from tests directory)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
sys.path.append(src_dir)

from sams_clustering import SAMS_Clustering, StandardMeanShift
from sklearn.datasets import make_blobs

def test_3d_basic():
    """Basic 3D functionality test"""
    print("Testing basic 3D functionality...")
    
    # Generate simple 3D data
    X_3d, y_true = make_blobs(n_samples=300, centers=4, n_features=3, 
                              random_state=42, cluster_std=1.0)
    
    print(f"3D data shape: {X_3d.shape}")
    print(f"True clusters: {len(np.unique(y_true))}")
    
    # Test SAMS on 3D data
    sams = SAMS_Clustering(bandwidth=None, sample_fraction=0.05, max_iter=100)
    
    try:
        labels, centers = sams.fit_predict(X_3d)
        print(f"✅ SAMS SUCCESS: Found {len(np.unique(labels))} clusters")
        print(f"Cluster centers shape: {centers.shape}")
        return True
    except Exception as e:
        print(f"❌ SAMS FAILED: {str(e)}")
        return False

def test_3d_vs_2d():
    """Compare 3D vs 2D behavior"""
    print("\nComparing 3D vs 2D behavior...")
    
    # 2D data
    X_2d, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
    
    # 3D data  
    X_3d, _ = make_blobs(n_samples=200, centers=3, n_features=3, random_state=42)
    
    sams_2d = SAMS_Clustering(bandwidth=1.0, sample_fraction=0.05, max_iter=50)
    sams_3d = SAMS_Clustering(bandwidth=1.0, sample_fraction=0.05, max_iter=50)
    
    # Test 2D
    labels_2d, centers_2d = sams_2d.fit_predict(X_2d)
    print(f"2D: {len(np.unique(labels_2d))} clusters, centers shape: {centers_2d.shape}")
    
    # Test 3D
    labels_3d, centers_3d = sams_3d.fit_predict(X_3d)
    print(f"3D: {len(np.unique(labels_3d))} clusters, centers shape: {centers_3d.shape}")
    
    return True

if __name__ == "__main__":
    print("SAMS 3D Capability Test")
    print("=" * 40)
    
    success_basic = test_3d_basic()
    success_compare = test_3d_vs_2d()
    
    if success_basic and success_compare:
        print("\n✅ ALL TESTS PASSED - SAMS works with 3D data!")
    else:
        print("\n❌ Some tests failed")