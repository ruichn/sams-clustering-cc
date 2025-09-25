import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import sys
import os
# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sams_clustering import SAMS_Clustering, generate_test_data
from standard_meanshift import StandardMeanShift
from sklearn.cluster import MeanShift

def evaluate_clustering(labels_true, labels_pred):
    """Evaluate clustering performance"""
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    n_clusters = len(np.unique(labels_pred))
    return ari, nmi, n_clusters

def run_corrected_experiment_1():
    """
    CORRECTED Experiment 1: Fair Comparison of SAMS vs Standard Mean-Shift
    Following Hyrien & Baran (2017) methodology:
    - Same datasets for both algorithms
    - Same bandwidth for both algorithms  
    - Multiple trials for statistical significance
    - Proper evaluation metrics
    """
    print("\n" + "="*80)
    print("CORRECTED EXPERIMENT 1: SAMS vs Standard Mean-Shift Fair Comparison")
    print("Following paper methodology: same data, same bandwidth, multiple trials")
    print("="*80)
    
    # Test configurations following paper methodology
    configurations = [
        {'type': 'blobs', 'size': 1000, 'trials': 5, 'name': 'Gaussian Mixture'},
        {'type': 'mixed', 'size': 1000, 'trials': 5, 'name': 'Mixed Density'},
        {'type': 'blobs', 'size': 2000, 'trials': 3, 'name': 'Large Dataset'}
    ]
    
    sample_fractions = [0.01, 0.02]  # SAMS sampling rates to test
    
    all_results = []
    
    print(f"\n{'Dataset':<18} {'Size':<6} {'Sample%':<8} {'Trials':<7} {'SAMS ARI':<10} {'MS ARI':<10} {'SAMS Time':<11} {'MS Time':<9} {'Speedup':<8}")
    print("-" * 95)
    
    for config in configurations:
        for sample_frac in sample_fractions:
            trial_results = []
            
            print(f"\nRunning {config['trials']} trials for {config['name']} (n={config['size']}, sample={sample_frac*100:.0f}%)...")
            
            # Multiple trials for statistical significance (as done in paper)
            for trial in range(config['trials']):
                # Generate SAME dataset for both algorithms
                X, y_true = generate_test_data(n_samples=config['size'], 
                                             dataset_type=config['type'])
                
                # SAMS clustering - compute data-driven bandwidth
                sams = SAMS_Clustering(bandwidth=None, sample_fraction=sample_frac, 
                                     max_iter=200, tol=1e-4)
                
                start_time = time.time()
                labels_sams, _ = sams.fit_predict(X)
                sams_time = time.time() - start_time
                
                # For the first trial of each configuration, plot the clustering result
                if trial == 0:
                    plot_clustering_result(X, labels_sams, config, sample_frac)
                
                # Standard Mean-Shift - use SAME bandwidth and SAME dataset
                ms = StandardMeanShift(bandwidth=sams.bandwidth, max_iter=200)
                
                start_time = time.time()
                labels_ms, _ = ms.fit_predict(X)  # SAME dataset, SAME bandwidth
                ms_time = time.time() - start_time
                
                # Evaluate both on SAME ground truth
                ari_sams, nmi_sams, n_clusters_sams = evaluate_clustering(y_true, labels_sams)
                ari_ms, nmi_ms, n_clusters_ms = evaluate_clustering(y_true, labels_ms)
                
                speedup = ms_time / sams_time if sams_time > 0 else 0
                
                trial_results.append({
                    'sams_ari': ari_sams,
                    'ms_ari': ari_ms,
                    'sams_nmi': nmi_sams,
                    'ms_nmi': nmi_ms,
                    'sams_time': sams_time,
                    'ms_time': ms_time,
                    'sams_clusters': n_clusters_sams,
                    'ms_clusters': n_clusters_ms,
                    'bandwidth': sams.bandwidth,
                    'speedup': speedup
                })
                
                print(f"  Trial {trial+1}: SAMS ARI={ari_sams:.3f}, MS ARI={ari_ms:.3f}, "
                      f"Speedup={speedup:.1f}x, Bandwidth={sams.bandwidth:.3f}")
            
            # Compute statistics across trials
            sams_aris = [r['sams_ari'] for r in trial_results]
            ms_aris = [r['ms_ari'] for r in trial_results]
            sams_times = [r['sams_time'] for r in trial_results]
            ms_times = [r['ms_time'] for r in trial_results]
            speedups = [r['speedup'] for r in trial_results]
            
            avg_sams_ari = np.mean(sams_aris)
            avg_ms_ari = np.mean(ms_aris)
            avg_sams_time = np.mean(sams_times)
            avg_ms_time = np.mean(ms_times)
            avg_speedup = np.mean(speedups)
            avg_bandwidth = np.mean([r['bandwidth'] for r in trial_results])
            
            # Statistical significance
            std_sams_ari = np.std(sams_aris)
            std_ms_ari = np.std(ms_aris)
            
            print(f"{config['name']:<18} {config['size']:<6} {sample_frac*100:<8.0f} {config['trials']:<7} "
                  f"{avg_sams_ari:<10.3f} {avg_ms_ari:<10.3f} {avg_sams_time:<11.3f} "
                  f"{avg_ms_time:<9.3f} {avg_speedup:<8.1f}x")
            
            # Store aggregated results
            all_results.append({
                'config': config['name'],
                'size': config['size'],
                'sample_fraction': sample_frac,
                'trials': config['trials'],
                'sams_ari_mean': avg_sams_ari,
                'ms_ari_mean': avg_ms_ari,
                'sams_ari_std': std_sams_ari,
                'ms_ari_std': std_ms_ari,
                'sams_time_mean': avg_sams_time,
                'ms_time_mean': avg_ms_time,
                'speedup_mean': avg_speedup,
                'bandwidth_mean': avg_bandwidth,
                'clustering_error_diff': abs(avg_ms_ari - avg_sams_ari)
            })
    
    # Analysis and visualization
    analyze_comparison_results(all_results)
    
    return all_results

def analyze_comparison_results(results):
    """Analyze and visualize the fair comparison results"""
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS OF FAIR COMPARISON")
    print("="*80)
    
    # Overall statistics
    all_sams_ari = [r['sams_ari_mean'] for r in results]
    all_ms_ari = [r['ms_ari_mean'] for r in results]
    all_speedups = [r['speedup_mean'] for r in results]
    all_errors = [r['clustering_error_diff'] for r in results]
    
    print(f"\nOverall Results Across All Configurations:")
    print(f"Average SAMS ARI: {np.mean(all_sams_ari):.3f} ± {np.std(all_sams_ari):.3f}")
    print(f"Average MS ARI:   {np.mean(all_ms_ari):.3f} ± {np.std(all_ms_ari):.3f}")
    print(f"Average Speedup:  {np.mean(all_speedups):.1f}x ± {np.std(all_speedups):.1f}x")
    print(f"Average |ARI Difference|: {np.mean(all_errors):.3f} ± {np.std(all_errors):.3f}")
    
    # Paper's claim validation
    print(f"\nPaper Claims Validation:")
    similar_quality = np.mean(all_errors) < 0.05  # Within 5% ARI difference
    faster = np.mean(all_speedups) > 1.0
    
    print(f"✓ Similar clustering quality (|ΔARI| < 0.05): {'YES' if similar_quality else 'NO'}")
    print(f"✓ Faster than standard mean-shift: {'YES' if faster else 'NO'}")
    print(f"✓ Speedup factor: {np.mean(all_speedups):.1f}x average")
    
    # Create fair comparison visualization
    plot_fair_comparison(results)

def plot_clustering_result(X, labels, config, sample_frac):
    """Plots the scatter plot of data points colored by cluster assignment."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
    
    n_clusters = len(np.unique(labels))
    
    plt.title(f"SAMS Clustering Result for '{config['name']}'\n"
              f"(n={config['size']}, sample={sample_frac*100:.0f}%, Clusters={n_clusters})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)
    
    # Create short filename
    name_map = {
        'Gaussian Mixture': 'gaussian',
        'Mixed Density': 'mixed', 
        'Large Dataset': 'large'
    }
    short_name = name_map.get(config['name'], config['name'].lower().replace(' ', '_'))
    filename = f"exp1_sams_{short_name}.png"
    # Use relative path from repository root
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(repo_root, 'plots', filename)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ SAMS clustering visualization saved to {save_path}")

def plot_fair_comparison(results):
    """Create visualization of fair comparison results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    configs = [f"{r['config']} (n={r['size']})" for r in results]
    sams_aris = [r['sams_ari_mean'] for r in results]
    ms_aris = [r['ms_ari_mean'] for r in results]
    sams_stds = [r['sams_ari_std'] for r in results]
    ms_stds = [r['ms_ari_std'] for r in results]
    speedups = [r['speedup_mean'] for r in results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    # ARI Comparison with error bars
    ax1.bar(x - width/2, sams_aris, width, label='SAMS', alpha=0.8, 
            yerr=sams_stds, capsize=5)
    ax1.bar(x + width/2, ms_aris, width, label='Mean-Shift', alpha=0.8,
            yerr=ms_stds, capsize=5)
    ax1.set_xlabel('Dataset Configuration')
    ax1.set_ylabel('Adjusted Rand Index (ARI)')
    ax1.set_title('Clustering Quality: SAMS vs Mean-Shift\\n(Same data, same bandwidth)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.split('(')[0] for c in configs], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup analysis
    colors = ['green' if s > 1 else 'red' for s in speedups]
    ax2.bar(x, speedups, alpha=0.8, color=colors)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax2.set_xlabel('Dataset Configuration')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.set_title('SAMS Speedup over Standard Mean-Shift')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.split('(')[0] for c in configs], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ARI difference (clustering error difference)
    ari_diffs = [abs(r['ms_ari_mean'] - r['sams_ari_mean']) for r in results]
    ax3.bar(x, ari_diffs, alpha=0.8, color='orange')
    ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    ax3.set_xlabel('Dataset Configuration')
    ax3.set_ylabel('|ARI Difference|')
    ax3.set_title('Clustering Quality Difference\\n(Lower is better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.split('(')[0] for c in configs], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sample fraction effect
    sample_fracs = [r['sample_fraction']*100 for r in results]
    ax4.scatter(sample_fracs, speedups, c=sams_aris, cmap='viridis', s=100, alpha=0.8)
    ax4.set_xlabel('SAMS Sample Fraction (%)')
    ax4.set_ylabel('Speedup Factor (x)')
    ax4.set_title('Sample Fraction vs Speedup\\n(Color = SAMS ARI)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('SAMS ARI')
    
    plt.tight_layout()
    # Use relative path from repository root
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(repo_root, 'plots', 'experiment1_performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Fair comparison plot saved to plots/experiment1_performance_comparison.png")

if __name__ == "__main__":
    print("CORRECTED EXPERIMENTAL VALIDATION")
    print("Fixing issues with original Experiment 1")
    print("="*50)
    
    np.random.seed(42)
    results = run_corrected_experiment_1()
    
    print("\n" + "="*80)
    print("CORRECTED EXPERIMENT 1: COMPLETE")
    print("✓ Fair comparison with same datasets and bandwidths")
    print("✓ Multiple trials for statistical significance")
    print("✓ Proper evaluation following paper methodology")
    print("="*80)
