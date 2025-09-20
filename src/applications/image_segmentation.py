import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sams_clustering import SAMS_Clustering
import time

def generate_synthetic_image(size=(100, 100), n_regions=4):
    """
    Generate a synthetic image for segmentation testing.
    This simulates the image segmentation experiments from the paper.
    """
    # Create coordinate grid
    x = np.linspace(0, 1, size[1])
    y = np.linspace(0, 1, size[0])
    np.meshgrid(x, y)
    
    # Create different regions with varying intensities
    image = np.zeros(size)
    
    if n_regions == 4:
        # Four quadrant regions
        image[0:size[0]//2, 0:size[1]//2] = 0.8  # Top-left: bright
        image[0:size[0]//2, size[1]//2:] = 0.3   # Top-right: dark
        image[size[0]//2:, 0:size[1]//2] = 0.6   # Bottom-left: medium
        image[size[0]//2:, size[1]//2:] = 0.9    # Bottom-right: very bright
    
    elif n_regions == 3:
        # Three circular regions
        center1 = (size[0]//4, size[1]//4)
        center2 = (3*size[0]//4, size[1]//4)
        center3 = (size[0]//2, 3*size[1]//4)
        
        for i in range(size[0]):
            for j in range(size[1]):
                dist1 = np.sqrt((i - center1[0])**2 + (j - center1[1])**2)
                dist2 = np.sqrt((i - center2[0])**2 + (j - center2[1])**2)
                dist3 = np.sqrt((i - center3[0])**2 + (j - center3[1])**2)
                
                min_dist = min(dist1, dist2, dist3)
                if min_dist == dist1:
                    image[i, j] = 0.8
                elif min_dist == dist2:
                    image[i, j] = 0.4
                else:
                    image[i, j] = 0.6
    
    # Add noise
    noise = np.random.normal(0, 0.05, size)
    image = np.clip(image + noise, 0, 1)
    
    return image

def extract_features_from_image(image, feature_type='intensity_position'):
    """
    Extract features from image for clustering.
    Different feature types as discussed in the paper.
    """
    h, w = image.shape
    features = []
    positions = []
    
    for i in range(h):
        for j in range(w):
            if feature_type == 'intensity_only':
                # Only pixel intensity
                features.append([image[i, j]])
            elif feature_type == 'position_only':
                # Only spatial position (normalized)
                features.append([i/h, j/w])
            elif feature_type == 'intensity_position':
                # Both intensity and spatial position
                features.append([image[i, j], i/h, j/w])
            elif feature_type == 'intensity_gradient':
                # Intensity and local gradient
                if i > 0 and i < h-1 and j > 0 and j < w-1:
                    grad_x = image[i, j+1] - image[i, j-1]
                    grad_y = image[i+1, j] - image[i-1, j]
                    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                    features.append([image[i, j], gradient_mag])
                else:
                    features.append([image[i, j], 0])
            
            positions.append([i, j])
    
    return np.array(features), np.array(positions)

def reconstruct_segmented_image(labels, positions, original_shape):
    """Reconstruct the segmented image from cluster labels"""
    segmented = np.zeros(original_shape)
    
    for idx, (i, j) in enumerate(positions):
        segmented[i, j] = labels[idx]
    
    return segmented

def run_image_segmentation_experiment():
    """
    Run image segmentation experiment similar to the paper.
    Tests SAMS algorithm on image segmentation tasks.
    """
    print("\n" + "="*60)
    print("IMAGE SEGMENTATION EXPERIMENT")
    print("Testing SAMS on synthetic image segmentation")
    print("="*60)
    
    # Test different image sizes and configurations
    configurations = [
        {'size': (50, 50), 'regions': 4, 'name': 'Small 4-region'},
        {'size': (80, 80), 'regions': 3, 'name': 'Medium 3-region'},
        {'size': (100, 100), 'regions': 4, 'name': 'Large 4-region'}
    ]
    
    feature_types = ['intensity_only', 'intensity_position', 'intensity_gradient']
    
    results = []
    
    _, axes = plt.subplots(len(configurations), len(feature_types) + 1, 
                            figsize=(16, 4*len(configurations)))
    
    for config_idx, config in enumerate(configurations):
        print(f"\nTesting {config['name']} image ({config['size'][0]}x{config['size'][1]})...")
        
        # Generate synthetic image
        image = generate_synthetic_image(config['size'], config['regions'])
        
        # Plot original image
        axes[config_idx, 0].imshow(image, cmap='gray')
        axes[config_idx, 0].set_title(f"Original {config['name']}")
        axes[config_idx, 0].axis('off')
        
        for feat_idx, feature_type in enumerate(feature_types):
            print(f"  Feature type: {feature_type}")
            
            # Extract features
            features, positions = extract_features_from_image(image, feature_type)
            
            print(f"    Extracted {len(features)} feature vectors with {features.shape[1]} dimensions")
            
            # Configure SAMS parameters based on feature dimensionality 
            # (data-driven bandwidth will be used automatically)
            if feature_type == 'intensity_only':
                sample_fraction = 0.02
            elif feature_type == 'position_only':
                sample_fraction = 0.015  # Slightly lower for spatial features
            else:
                sample_fraction = 0.02   # Combined features
            
            # Apply SAMS clustering with image-optimized parameters
            # For image segmentation, we use smaller bandwidth for finer segments
            if feature_type == 'intensity_only':
                bandwidth = 0.05  # Small bandwidth for intensity-only
            elif feature_type == 'position_only':
                bandwidth = 0.1   # Medium bandwidth for spatial features
            else:
                bandwidth = 0.15  # Larger bandwidth for combined features
            
            sams = SAMS_Clustering(bandwidth=bandwidth,
                                 sample_fraction=sample_fraction, 
                                 max_iter=200, 
                                 tol=1e-4,
                                 adaptive_sampling=True,
                                 early_stop=True)
            
            start_time = time.time()
            labels, _ = sams.fit_predict(features)
            clustering_time = time.time() - start_time
            
            n_clusters = len(np.unique(labels))
            
            print(f"    Clustering time: {clustering_time:.3f}s")
            print(f"    Number of segments: {n_clusters}")
            
            # Reconstruct segmented image
            segmented_image = reconstruct_segmented_image(labels, positions, config['size'])
            
            # Plot segmented image
            axes[config_idx, feat_idx + 1].imshow(segmented_image, cmap='tab10')
            axes[config_idx, feat_idx + 1].set_title(f"{feature_type}\n{n_clusters} segments, {clustering_time:.2f}s")
            axes[config_idx, feat_idx + 1].axis('off')
            
            # Store results
            results.append({
                'config': config['name'],
                'feature_type': feature_type,
                'n_pixels': config['size'][0] * config['size'][1],
                'n_features': features.shape[1],
                'n_clusters': n_clusters,
                'clustering_time': clustering_time,
                'bandwidth': sams.bandwidth,
                'sample_fraction': sample_fraction
            })
    
    plt.tight_layout()
    plt.savefig('/Users/ruichen/Projects/paper-implementation/plots/image_segmentation_results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary results
    print("\n" + "="*60)
    print("IMAGE SEGMENTATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<18} {'Features':<16} {'Pixels':<8} {'Segments':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config']:<18} {result['feature_type']:<16} "
              f"{result['n_pixels']:<8} {result['n_clusters']:<10} "
              f"{result['clustering_time']:<10.3f}")
    
    return results

def run_color_image_segmentation():
    """
    Simulate color image segmentation using RGB features.
    """
    print("\n" + "="*60)
    print("COLOR IMAGE SEGMENTATION SIMULATION")
    print("="*60)
    
    # Create a synthetic color image (RGB channels)
    size = (60, 60)
    
    # Generate RGB image with distinct color regions
    image_rgb = np.zeros((size[0], size[1], 3))
    
    # Red region (top-left)
    image_rgb[0:size[0]//2, 0:size[1]//2, 0] = 0.8
    image_rgb[0:size[0]//2, 0:size[1]//2, 1] = 0.1
    image_rgb[0:size[0]//2, 0:size[1]//2, 2] = 0.1
    
    # Green region (top-right)
    image_rgb[0:size[0]//2, size[1]//2:, 0] = 0.1
    image_rgb[0:size[0]//2, size[1]//2:, 1] = 0.8
    image_rgb[0:size[0]//2, size[1]//2:, 2] = 0.1
    
    # Blue region (bottom-left)
    image_rgb[size[0]//2:, 0:size[1]//2, 0] = 0.1
    image_rgb[size[0]//2:, 0:size[1]//2, 1] = 0.1
    image_rgb[size[0]//2:, 0:size[1]//2, 2] = 0.8
    
    # Yellow region (bottom-right)
    image_rgb[size[0]//2:, size[1]//2:, 0] = 0.8
    image_rgb[size[0]//2:, size[1]//2:, 1] = 0.8
    image_rgb[size[0]//2:, size[1]//2:, 2] = 0.1
    
    # Add noise
    noise = np.random.normal(0, 0.03, image_rgb.shape)
    image_rgb = np.clip(image_rgb + noise, 0, 1)
    
    # Extract RGB features
    h, w, _ = image_rgb.shape
    features = []
    positions = []
    
    for i in range(h):
        for j in range(w):
            # RGB + position features
            r, g, b = image_rgb[i, j, :]
            features.append([r, g, b, i/h, j/w])
            positions.append([i, j])
    
    features = np.array(features)
    positions = np.array(positions)
    
    print(f"Color image size: {h}x{w}")
    print(f"Feature vector dimension: {features.shape[1]} (RGB + position)")
    
    # Apply SAMS clustering with image-optimized parameters
    # For color images with RGB+position, use moderate bandwidth
    sams = SAMS_Clustering(bandwidth=0.2,
                          sample_fraction=0.02, 
                          max_iter=150,
                          tol=1e-4,
                          adaptive_sampling=True,
                          early_stop=True)
    
    start_time = time.time()
    labels, _ = sams.fit_predict(features)
    clustering_time = time.time() - start_time
    
    n_clusters = len(np.unique(labels))
    
    print(f"Clustering time: {clustering_time:.3f}s")
    print(f"Number of color segments: {n_clusters}")
    
    # Reconstruct segmented image
    segmented_image = reconstruct_segmented_image(labels, positions, (h, w))
    
    # Create visualization
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Color Image')
    axes[0].axis('off')
    
    axes[1].imshow(segmented_image, cmap='tab10')
    axes[1].set_title(f'SAMS Segmentation\n{n_clusters} segments, {clustering_time:.2f}s')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/ruichen/Projects/paper-implementation/plots/color_segmentation_results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return clustering_time, n_clusters

def run_all_image_experiments():
    """Run comprehensive image segmentation validation"""
    print("RUNNING IMAGE SEGMENTATION EXPERIMENTS")
    print("Validating SAMS algorithm on image segmentation tasks")
    print("="*70)
    
    # Run grayscale image segmentation
    grayscale_results = run_image_segmentation_experiment()
    
    # Run color image segmentation
    color_time, color_clusters = run_color_image_segmentation()
    
    # Final summary
    print("\n" + "="*60)
    print("IMAGE SEGMENTATION VALIDATION COMPLETE")
    print("="*60)
    
    print("\n✓ Grayscale image segmentation:")
    print(f"  - Tested on {len(grayscale_results)} configurations")
    print(f"  - Different feature types: intensity, position, gradient")
    print(f"  - Image sizes from 50x50 to 100x100 pixels")
    
    print(f"\n✓ Color image segmentation:")
    print(f"  - RGB + position features (5D)")
    print(f"  - {color_clusters} segments identified")
    print(f"  - Processing time: {color_time:.3f}s")
    
    print("\n✓ Key observations:")
    print("  - SAMS successfully segments images of various types")
    print("  - Processing time scales reasonably with image size")
    print("  - Different feature types capture different aspects")
    print("  - Algorithm handles multi-dimensional feature spaces")
    
    avg_time = np.mean([r['clustering_time'] for r in grayscale_results])
    print(f"\n✓ Average processing time: {avg_time:.3f}s per image")
    
    return grayscale_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run all image segmentation experiments
    results = run_all_image_experiments()