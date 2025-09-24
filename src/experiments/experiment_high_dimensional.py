import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os
# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sams_clustering import SAMS_Clustering

def generate_high_dimensional_data(n_samples=1000, n_features=128, n_centers=5, random_state=42):
    """Generate high-dimensional clustered data."""
    print(f"Generating {n_samples} samples with {n_features} dimensions and {n_centers} clusters...")
    
    # Generate high-dimensional blobs
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features, 
        centers=n_centers,
        cluster_std=2.0,
        center_box=(-10.0, 10.0),
        random_state=random_state
    )
    
    # Standardize the data (important for high dimensions)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_true, scaler

def evaluate_clustering(labels_true, labels_pred, X=None):
    """Evaluate clustering performance with multiple metrics."""
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    
    # Silhouette score (only if reasonable number of points and clusters)
    silhouette = None
    if X is not None and len(np.unique(labels_pred)) > 1 and len(X) <= 5000:
        try:
            silhouette = silhouette_score(X, labels_pred)
        except:
            silhouette = None
    
    n_clusters = len(np.unique(labels_pred))
    return ari, nmi, silhouette, n_clusters

def run_high_dimensional_experiment():
    """
    High-Dimensional Clustering Experiment: SAMS vs Mean-Shift on 128D data
    Tests the curse of dimensionality effects and SAMS performance scaling
    """
    print("\n" + "="*80)
    print("HIGH-DIMENSIONAL CLUSTERING EXPERIMENT: 128D DATA")
    print("Testing SAMS vs Mean-Shift performance on high-dimensional data")
    print("="*80)
    
    # Test configurations for different sample sizes
    configurations = [
        {'n_samples': 500, 'n_features': 128, 'n_centers': 3, 'name': 'Small 128D'},
        {'n_samples': 1000, 'n_features': 128, 'n_centers': 5, 'name': 'Medium 128D'},
        {'n_samples': 2000, 'n_features': 128, 'n_centers': 7, 'name': 'Large 128D'},
        {'n_samples': 1000, 'n_features': 64, 'n_centers': 5, 'name': 'Medium 64D'},
        {'n_samples': 1000, 'n_features': 256, 'n_centers': 5, 'name': 'Ultra High 256D'}
    ]
    
    sample_fractions = [0.01, 0.02]  # SAMS sampling rates
    
    all_results = []
    
    print(f"\n{'Dataset':<15} {'Dims':<5} {'Size':<6} {'SFrac':<6} {'SAMS ARI':<10} {'MS ARI':<10} {'SAMS Time':<11} {'MS Time':<10} {'Speedup':<8}")
    print("-" * 100)
    
    for config in configurations:
        print(f"\n--- Processing {config['name']} ---")
        
        # Generate high-dimensional data
        X, y_true, scaler = generate_high_dimensional_data(
            n_samples=config['n_samples'],
            n_features=config['n_features'], 
            n_centers=config['n_centers']
        )
        
        for sample_frac in sample_fractions:
            print(f"Testing with sample fraction: {sample_frac}")
            
            try:
                # SAMS Clustering
                print("  Running SAMS clustering...")
                sams = SAMS_Clustering(
                    bandwidth=None,  # Auto-estimate
                    sample_fraction=sample_frac,
                    max_iter=200,
                    tol=1e-4
                )
                
                start_time = time.time()
                labels_sams, centers_sams = sams.fit_predict(X)
                sams_time = time.time() - start_time
                
                # Evaluate SAMS
                ari_sams, nmi_sams, sil_sams, n_clusters_sams = evaluate_clustering(
                    y_true, labels_sams, X if len(X) <= 2000 else None
                )
                
                print(f"    SAMS: {n_clusters_sams} clusters, ARI={ari_sams:.3f}, Time={sams_time:.3f}s")
                
                # Mean-Shift Clustering (with timeout for large datasets)
                ms_time = None
                ari_ms = None
                nmi_ms = None
                sil_ms = None
                n_clusters_ms = None
                speedup = None
                
                # Only run MeanShift for smaller datasets to avoid excessive runtime
                if config['n_samples'] <= 1000 and config['n_features'] <= 128:
                    try:
                        print("  Running Mean-Shift clustering...")
                        ms = MeanShift(bandwidth=sams.bandwidth, max_iter=100)
                        
                        start_time = time.time()
                        labels_ms = ms.fit_predict(X)
                        ms_time = time.time() - start_time
                        
                        # Evaluate Mean-Shift
                        ari_ms, nmi_ms, sil_ms, n_clusters_ms = evaluate_clustering(
                            y_true, labels_ms, X if len(X) <= 2000 else None
                        )
                        
                        speedup = ms_time / sams_time if sams_time > 0 else 0
                        print(f"    MS: {n_clusters_ms} clusters, ARI={ari_ms:.3f}, Time={ms_time:.3f}s, Speedup={speedup:.1f}x")
                        
                    except Exception as e:
                        print(f"    Mean-Shift failed: {str(e)[:50]}...")
                        ms_time = float('inf')
                        speedup = float('inf')
                else:
                    print("    Skipping Mean-Shift (too large/slow for high-dimensional data)")
                    ms_time = "N/A"
                    speedup = "N/A"
                
                # Print results
                ms_ari_str = f"{ari_ms:.3f}" if ari_ms is not None else "N/A"
                ms_time_str = f"{ms_time:.3f}" if isinstance(ms_time, float) and ms_time != float('inf') else str(ms_time)
                speedup_str = f"{speedup:.1f}x" if isinstance(speedup, float) and speedup != float('inf') else str(speedup)
                
                print(f"{config['name']:<15} {config['n_features']:<5} {config['n_samples']:<6} {sample_frac:<6.2f} "
                      f"{ari_sams:<10.3f} {ms_ari_str:<10} {sams_time:<11.3f} "
                      f"{ms_time_str:<10} {speedup_str:<8}")
                
                # Store results
                all_results.append({
                    'config': config['name'],
                    'n_features': config['n_features'],
                    'n_samples': config['n_samples'],
                    'n_centers': config['n_centers'], 
                    'sample_fraction': sample_frac,
                    'sams_ari': ari_sams,
                    'ms_ari': ari_ms,
                    'sams_nmi': nmi_sams,
                    'ms_nmi': nmi_ms,
                    'sams_silhouette': sil_sams,
                    'ms_silhouette': sil_ms,
                    'sams_time': sams_time,
                    'ms_time': ms_time if isinstance(ms_time, float) else None,
                    'sams_clusters': n_clusters_sams,
                    'ms_clusters': n_clusters_ms,
                    'bandwidth': sams.bandwidth,
                    'speedup': speedup if isinstance(speedup, float) else None
                })
                
                # Create visualization for first configuration
                if config == configurations[0] and sample_frac == sample_fractions[0]:
                    visualize_high_dimensional_results(X, y_true, labels_sams, config)
                
            except Exception as e:
                print(f"  Error processing {config['name']}: {str(e)}")
                continue
    
    # Analysis and visualization
    analyze_high_dimensional_results(all_results)
    plot_high_dimensional_comparison(all_results)
    
    return all_results

def visualize_high_dimensional_results(X, y_true, labels_pred, config):
    """Visualize high-dimensional clustering results using PCA."""
    print(f"\nCreating PCA visualization for {config['name']}...")
    
    # Reduce to 2D using PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # True clusters
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10', alpha=0.7, s=20)
    ax1.set_title(f'True Clusters (PCA projection)\n{config["name"]} - {config["n_features"]}D data')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.grid(True, alpha=0.3)
    
    # SAMS results
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pred, cmap='tab10', alpha=0.7, s=20)
    n_clusters_found = len(np.unique(labels_pred))
    ax2.set_title(f'SAMS Clustering Results\n{n_clusters_found} clusters found')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax2.grid(True, alpha=0.3)
    
    # Add explained variance info
    total_var = pca.explained_variance_ratio_[:2].sum()
    fig.suptitle(f'High-Dimensional Clustering Visualization\n'
                 f'PCA explains {total_var:.1%} of total variance', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(repo_root, 'plots', f'high_dim_clustering_{config["n_features"]}d.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ High-dimensional clustering visualization saved to {save_path}")

def analyze_high_dimensional_results(results):
    """Analyze high-dimensional clustering results."""
    print("\n" + "="*80)
    print("HIGH-DIMENSIONAL CLUSTERING ANALYSIS")
    print("="*80)
    
    # Group results by dimensionality
    dim_groups = {}
    for r in results:
        dim = r['n_features']
        if dim not in dim_groups:
            dim_groups[dim] = []
        dim_groups[dim].append(r)
    
    print("\nPerformance by Dimensionality:")
    print("Dims | Avg SAMS ARI | Avg SAMS Time | Speedup | Sample Size Range")
    print("-" * 70)
    
    for dim in sorted(dim_groups.keys()):
        group = dim_groups[dim]
        
        sams_aris = [r['sams_ari'] for r in group]
        sams_times = [r['sams_time'] for r in group]
        valid_speedups = [r['speedup'] for r in group if r['speedup'] is not None and isinstance(r['speedup'], float)]
        sample_sizes = [r['n_samples'] for r in group]
        
        avg_ari = np.mean(sams_aris)
        avg_time = np.mean(sams_times) 
        avg_speedup = np.mean(valid_speedups) if valid_speedups else "N/A"
        size_range = f"{min(sample_sizes)}-{max(sample_sizes)}"
        
        speedup_str = f"{avg_speedup:.1f}x" if isinstance(avg_speedup, float) else str(avg_speedup)
        
        print(f"{dim:4d} | {avg_ari:12.3f} | {avg_time:13.3f} | {speedup_str:7} | {size_range}")
    
    # Dimensionality analysis
    print(f"\nDimensionality Effects:")
    
    # Find best performance across dimensions
    best_ari = max(r['sams_ari'] for r in results)
    best_config = next(r for r in results if r['sams_ari'] == best_ari)
    print(f"Best ARI: {best_ari:.3f} on {best_config['config']} ({best_config['n_features']}D)")
    
    # Time scaling analysis
    time_by_dim = {}
    for r in results:
        if r['n_samples'] == 1000:  # Compare same sample size
            time_by_dim[r['n_features']] = r['sams_time']
    
    if len(time_by_dim) > 1:
        print(f"\nTime Scaling (1000 samples):")
        for dim in sorted(time_by_dim.keys()):
            print(f"  {dim}D: {time_by_dim[dim]:.3f}s")
    
    # Sample fraction analysis
    print(f"\nSample Fraction Effects:")
    frac_analysis = {}
    for r in results:
        frac = r['sample_fraction']
        if frac not in frac_analysis:
            frac_analysis[frac] = []
        frac_analysis[frac].append(r['sams_ari'])
    
    for frac in sorted(frac_analysis.keys()):
        avg_ari = np.mean(frac_analysis[frac])
        print(f"  {frac:.2f}: Avg ARI = {avg_ari:.3f}")

def plot_high_dimensional_comparison(results):
    """Create visualization comparing high-dimensional clustering results."""
    
    # Prepare data for plotting
    dims = [r['n_features'] for r in results]
    sample_sizes = [r['n_samples'] for r in results]
    sams_aris = [r['sams_ari'] for r in results]
    sams_times = [r['sams_time'] for r in results]
    sample_fracs = [r['sample_fraction'] for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ARI vs Dimensionality
    scatter1 = ax1.scatter(dims, sams_aris, c=sample_sizes, cmap='viridis', s=80, alpha=0.7)
    ax1.set_xlabel('Number of Dimensions')
    ax1.set_ylabel('SAMS ARI Score')
    ax1.set_title('Clustering Quality vs Dimensionality')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Sample Size')
    
    # 2. Time vs Dimensionality  
    scatter2 = ax2.scatter(dims, sams_times, c=sample_sizes, cmap='plasma', s=80, alpha=0.7)
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_ylabel('SAMS Time (seconds)')
    ax2.set_title('Runtime vs Dimensionality')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Sample Size')
    
    # 3. ARI vs Sample Size (colored by dimensions)
    scatter3 = ax3.scatter(sample_sizes, sams_aris, c=dims, cmap='coolwarm', s=80, alpha=0.7)
    ax3.set_xlabel('Sample Size')
    ax3.set_ylabel('SAMS ARI Score')
    ax3.set_title('Clustering Quality vs Sample Size')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Dimensions')
    
    # 4. Sample Fraction Effect
    scatter4 = ax4.scatter(sample_fracs, sams_aris, c=dims, cmap='spring', s=80, alpha=0.7)
    ax4.set_xlabel('Sample Fraction')
    ax4.set_ylabel('SAMS ARI Score')
    ax4.set_title('Sample Fraction Effect on Quality')
    ax4.grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Dimensions')
    
    plt.tight_layout()
    
    # Save plot
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(repo_root, 'plots', 'high_dimensional_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ High-dimensional analysis plot saved to {save_path}")

if __name__ == "__main__":
    print("HIGH-DIMENSIONAL CLUSTERING EXPERIMENT")
    print("Testing SAMS performance on 128D and higher dimensional data")
    print("="*60)
    
    np.random.seed(42)
    results = run_high_dimensional_experiment()
    
    print("\n" + "="*80)
    print("HIGH-DIMENSIONAL EXPERIMENT COMPLETE")
    print("✓ Tested multiple dimensionalities (64D, 128D, 256D)")
    print("✓ Evaluated clustering quality and runtime scaling") 
    print("✓ Analyzed dimensionality effects and sample fraction impact")
    print("="*80)