#!/usr/bin/env python3
"""
Comprehensive testing of all demo simulation scenarios
"""
import sys
import os

# Add repository root to path for imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

from app import generate_dataset, DemoSAMS
import numpy as np
from sklearn.metrics import adjusted_rand_score

def test_standard_datasets():
    """Test all standard dataset types and dimensions"""
    print('🧪 STANDARD DATASET TESTING')
    print('='*50)
    
    scenarios = [
        # 2D scenarios
        {'type': 'Gaussian Blobs', 'n_samples': 1000, 'n_centers': 3, 'n_features': 2},
        {'type': 'Gaussian Blobs', 'n_samples': 1000, 'n_centers': 3, 'n_features': 3},
        {'type': 'Concentric Circles', 'n_samples': 1000, 'n_centers': 2, 'n_features': 2},
        {'type': 'Concentric Circles', 'n_samples': 1000, 'n_centers': 2, 'n_features': 3},
        {'type': 'Two Moons', 'n_samples': 1000, 'n_centers': 2, 'n_features': 2},
        {'type': 'Two Moons', 'n_samples': 1000, 'n_centers': 2, 'n_features': 3},
        {'type': 'Mixed Densities', 'n_samples': 1000, 'n_centers': 3, 'n_features': 2},
        {'type': 'Mixed Densities', 'n_samples': 1000, 'n_centers': 3, 'n_features': 3},
        # 3D scenarios
        {'type': '3D Blobs', 'n_samples': 800, 'n_centers': 4, 'n_features': 3},
        {'type': '3D Spheres', 'n_samples': 600, 'n_centers': 3, 'n_features': 3},
    ]
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f'\n{i}. Testing {scenario["type"]} ({scenario["n_features"]}D)...')
        
        try:
            # Generate dataset
            result = generate_dataset(
                scenario['type'], 
                scenario['n_samples'], 
                scenario['n_centers'], 
                0.1,  # noise_level
                1.0,  # cluster_std
                scenario['n_features']
            )
            
            X, y_true = result[:2]
            print(f'   ✅ Data: {X.shape}, Labels: {len(np.unique(y_true))} clusters')
            
            # Test SAMS clustering
            sams = DemoSAMS(bandwidth=None, sample_fraction=0.02, max_iter=30)
            labels, centers = sams.fit_predict(X)
            
            ari = adjusted_rand_score(y_true, labels)
            
            print(f'   ✅ SAMS: {len(np.unique(labels))} clusters, ARI={ari:.3f}')
            results.append((scenario['type'], scenario['n_features'], 'PASS', ari))
            
        except Exception as e:
            print(f'   ❌ FAILED: {str(e)}')
            results.append((scenario['type'], scenario['n_features'], 'FAIL', 0))
    
    return results

def test_image_segmentation():
    """Test all image segmentation scenarios"""
    print('\n🖼️ IMAGE SEGMENTATION TESTING')
    print('='*50)
    
    img_scenarios = [
        {'feature_type': 'intensity_only', 'size': (40, 40), 'regions': 4, 'expected_dim': 1},
        {'feature_type': 'position_only', 'size': (40, 40), 'regions': 4, 'expected_dim': 2},
        {'feature_type': 'intensity_position', 'size': (40, 40), 'regions': 4, 'expected_dim': 3},
        {'feature_type': 'intensity_gradient', 'size': (40, 40), 'regions': 4, 'expected_dim': 2},
        {'feature_type': 'intensity_position', 'size': (60, 60), 'regions': 3, 'expected_dim': 3},
        {'feature_type': 'intensity_position', 'size': (80, 80), 'regions': 5, 'expected_dim': 3},
    ]
    
    results = []
    
    for i, scenario in enumerate(img_scenarios, 1):
        print(f'\n{i}. Testing Image Segmentation ({scenario["feature_type"]}, {scenario["size"]})...')
        
        try:
            # Generate image dataset
            n_samples = scenario['size'][0] * scenario['size'][1]
            result = generate_dataset(
                'Image Segmentation',
                n_samples,
                scenario['regions'],
                0.05,  # noise_level
                feature_type=scenario['feature_type'],
                image_size=scenario['size']
            )
            
            X, y_true, image = result
            print(f'   ✅ Data: {X.shape}, Image: {image.shape}, Labels: {len(np.unique(y_true))} regions')
            
            # Verify expected dimensions
            if X.shape[1] != scenario['expected_dim']:
                raise ValueError(f'Expected {scenario["expected_dim"]}D features, got {X.shape[1]}D')
            
            # Test SAMS clustering (smaller sample for speed)
            sams = DemoSAMS(bandwidth=0.1, sample_fraction=0.03, max_iter=20)
            labels, centers = sams.fit_predict(X)
            
            ari = adjusted_rand_score(y_true, labels)
            
            print(f'   ✅ SAMS: {len(np.unique(labels))} clusters, ARI={ari:.3f}')
            results.append((scenario['feature_type'], scenario['size'], 'PASS', ari))
            
        except Exception as e:
            print(f'   ❌ FAILED: {str(e)}')
            results.append((scenario['feature_type'], scenario['size'], 'FAIL', 0))
    
    return results

def main():
    """Run comprehensive testing"""
    print('🔬 COMPREHENSIVE DEMO SIMULATION TESTING')
    print('='*70)
    
    # Test standard datasets
    standard_results = test_standard_datasets()
    
    # Test image segmentation
    image_results = test_image_segmentation()
    
    # Overall summary
    all_results = standard_results + image_results
    passed = sum(1 for r in all_results if r[2] == 'PASS')
    total = len(all_results)
    
    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    print(f'Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)')
    
    print(f'\nStandard Datasets: {sum(1 for r in standard_results if r[2] == "PASS")}/{len(standard_results)} passed')
    print(f'Image Segmentation: {sum(1 for r in image_results if r[2] == "PASS")}/{len(image_results)} passed')
    
    if passed == total:
        print('\n🎉 ALL DEMO SCENARIOS WORKING CORRECTLY!')
        print('✅ Ready for deployment!')
    else:
        print('\n⚠️ Some tests failed - check output above')
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)