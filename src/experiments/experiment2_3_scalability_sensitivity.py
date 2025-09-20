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
from sams_clustering import SAMS_Clustering, StandardMeanShift, generate_test_data

def evaluate_clustering(labels_true, labels_pred):
    """Evaluate clustering performance"""
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    n_clusters = len(np.unique(labels_pred))
    
    # Clustering error rate as used in the paper
    error_rate = 1.0 - ari if ari >= 0 else 1.0
    
    return ari, nmi, n_clusters, error_rate

def run_corrected_experiment_2():
    """
    CORRECTED Experiment 2: Scalability Analysis
    Following Hyrien & Baran (2017) methodology:
    - Test both SAMS and standard mean-shift on same datasets
    - Keep sample fraction constant to isolate scalability effects
    - Multiple trials for statistical significance
    - Test computational complexity claims (O(n) per iteration)
    """
    print("\n" + "="*80)
    print("CORRECTED EXPERIMENT 2: Scalability Analysis")
    print("SAMS vs Standard Mean-Shift - Computational Scalability")
    print("="*80)
    
    # Dataset sizes following paper's large-scale approach
    dataset_sizes = [500, 1000, 2000, 5000, 10000]
    sample_fraction = 0.01  # Keep constant as per paper methodology
    trials_per_size = 3  # Multiple trials for statistical significance
    
    results = []
    
    print(f"\nTesting scalability with constant sample fraction: {sample_fraction*100:.0f}%")
    print(f"Multiple trials per size: {trials_per_size}")
    print(f"\n{'Size':<8} {'SAMS Time':<12} {'MS Time':<10} {'SAMS ARI':<11} {'MS ARI':<9} {'SAMS Error%':<12} {'MS Error%':<10} {'Speedup':<8}")
    print("-" * 88)
    
    for size in dataset_sizes:
        size_results = []
        
        print(f"\nTesting dataset size: {size}")
        
        for trial in range(trials_per_size):
            # Generate same dataset for both algorithms
            X, y_true = generate_test_data(n_samples=size, dataset_type='blobs')
            
            try:
                # SAMS clustering
                sams = SAMS_Clustering(bandwidth=None, sample_fraction=sample_fraction, 
                                     max_iter=200, tol=1e-4)
                
                start_time = time.time()
                labels_sams, _ = sams.fit_predict(X)
                sams_time = time.time() - start_time
                
                ari_sams, nmi_sams, clusters_sams, error_sams = evaluate_clustering(y_true, labels_sams)
                
                # Standard Mean-Shift on same dataset with same bandwidth
                ms = StandardMeanShift(bandwidth=sams.bandwidth, max_iter=200, tol=1e-4)
                
                start_time = time.time()
                labels_ms, _ = ms.fit_predict(X)
                ms_time = time.time() - start_time
                
                ari_ms, nmi_ms, clusters_ms, error_ms = evaluate_clustering(y_true, labels_ms)
                
                speedup = ms_time / sams_time if sams_time > 0 else 0
                
                size_results.append({
                    'size': size,
                    'sams_time': sams_time,
                    'ms_time': ms_time,
                    'sams_ari': ari_sams,
                    'ms_ari': ari_ms,
                    'sams_error': error_sams * 100,
                    'ms_error': error_ms * 100,
                    'speedup': speedup,
                    'bandwidth': sams.bandwidth
                })
                
                print(f"  Trial {trial+1}: SAMS={sams_time:.3f}s, MS={ms_time:.3f}s, Speedup={speedup:.1f}x")
                
            except Exception as e:
                print(f"  Trial {trial+1}: ERROR - {str(e)}")
                continue
        
        if size_results:  # Only process if we have valid results
            # Average across trials
            avg_sams_time = np.mean([r['sams_time'] for r in size_results])
            avg_ms_time = np.mean([r['ms_time'] for r in size_results])
            avg_sams_ari = np.mean([r['sams_ari'] for r in size_results])
            avg_ms_ari = np.mean([r['ms_ari'] for r in size_results])
            avg_sams_error = np.mean([r['sams_error'] for r in size_results])
            avg_ms_error = np.mean([r['ms_error'] for r in size_results])
            avg_speedup = np.mean([r['speedup'] for r in size_results])
            
            print(f"{size:<8} {avg_sams_time:<12.3f} {avg_ms_time:<10.3f} {avg_sams_ari:<11.3f} "
                  f"{avg_ms_ari:<9.3f} {avg_sams_error:<12.1f} {avg_ms_error:<10.1f} {avg_speedup:<8.1f}x")
            
            results.append({
                'size': size,
                'sams_time_mean': avg_sams_time,
                'ms_time_mean': avg_ms_time,
                'sams_ari_mean': avg_sams_ari,
                'ms_ari_mean': avg_ms_ari,
                'sams_error_mean': avg_sams_error,
                'ms_error_mean': avg_ms_error,
                'speedup_mean': avg_speedup,
                'valid_trials': len(size_results)
            })
    
    # Analyze computational complexity
    analyze_scalability_results(results)
    
    return results

def analyze_scalability_results(results):
    """Analyze computational complexity and scalability"""
    
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS")
    print("="*80)
    
    sizes = [r['size'] for r in results]
    sams_times = [r['sams_time_mean'] for r in results]
    ms_times = [r['ms_time_mean'] for r in results]
    speedups = [r['speedup_mean'] for r in results]
    
    # Analyze time complexity
    print(f"\nComputational Complexity Analysis:")
    
    # Fit linear model to check O(n) scaling for SAMS
    if len(sizes) >= 3:
        # Linear regression for time vs size
        sams_slope = np.polyfit(sizes, sams_times, 1)[0]
        ms_slope = np.polyfit(sizes, ms_times, 1)[0]
        
        print(f"SAMS time scaling: {sams_slope:.6f}s per data point")
        print(f"Mean-Shift time scaling: {ms_slope:.6f}s per data point")
        print(f"Relative scaling ratio: {ms_slope/sams_slope:.1f}x")
        
        # Check if SAMS approaches O(n) while MS approaches O(n²)
        # For true O(n²), we'd expect quadratic relationship
        ms_quad_fit = np.polyfit(sizes, ms_times, 2)
        quadratic_prediction = np.polyval(ms_quad_fit, sizes[-1])
        linear_prediction = ms_slope * sizes[-1]
        
        print(f"\nMean-Shift complexity analysis (largest dataset):")
        print(f"Linear model prediction: {linear_prediction:.3f}s")
        print(f"Quadratic model prediction: {quadratic_prediction:.3f}s")
        print(f"Actual time: {ms_times[-1]:.3f}s")
    
    # Paper claims validation
    print(f"\nPaper Claims Validation:")
    avg_speedup = np.mean(speedups)
    speedup_improving = len(speedups) > 1 and speedups[-1] > speedups[0]
    
    print(f"Average speedup: {avg_speedup:.1f}x")
    print(f"Speedup improving with size: {'YES' if speedup_improving else 'NO'}")
    print(f"Faster than mean-shift: {'YES' if avg_speedup > 1.0 else 'NO'}")
    
    # Plot scalability results
    plot_scalability_comparison(results)

def plot_scalability_comparison(results):
    """Create comprehensive scalability comparison plots"""
    
    sizes = [r['size'] for r in results]
    sams_times = [r['sams_time_mean'] for r in results]
    ms_times = [r['ms_time_mean'] for r in results]
    sams_aris = [r['sams_ari_mean'] for r in results]
    ms_aris = [r['ms_ari_mean'] for r in results]
    speedups = [r['speedup_mean'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Runtime comparison
    ax1.plot(sizes, sams_times, 'bo-', linewidth=2, markersize=8, label='SAMS')
    ax1.plot(sizes, ms_times, 'ro-', linewidth=2, markersize=8, label='Mean-Shift')
    ax1.set_xlabel('Dataset Size')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime Scalability Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Quality comparison
    ax2.plot(sizes, sams_aris, 'bo-', linewidth=2, markersize=8, label='SAMS ARI')
    ax2.plot(sizes, ms_aris, 'ro-', linewidth=2, markersize=8, label='Mean-Shift ARI')
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Adjusted Rand Index')
    ax2.set_title('Clustering Quality vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Speedup analysis
    ax3.plot(sizes, speedups, 'go-', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax3.set_xlabel('Dataset Size')
    ax3.set_ylabel('Speedup Factor (x)')
    ax3.set_title('SAMS Speedup vs Dataset Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Time complexity analysis
    if len(sizes) >= 3:
        # Fit models
        linear_fit_sams = np.polyfit(sizes, sams_times, 1)
        linear_fit_ms = np.polyfit(sizes, ms_times, 1)
        quad_fit_ms = np.polyfit(sizes, ms_times, 2)
        
        # Generate smooth curves
        size_range = np.linspace(min(sizes), max(sizes), 100)
        linear_pred_sams = np.polyval(linear_fit_sams, size_range)
        linear_pred_ms = np.polyval(linear_fit_ms, size_range)
        quad_pred_ms = np.polyval(quad_fit_ms, size_range)
        
        ax4.scatter(sizes, sams_times, color='blue', s=50, label='SAMS actual')
        ax4.scatter(sizes, ms_times, color='red', s=50, label='MS actual')
        ax4.plot(size_range, linear_pred_sams, 'b--', alpha=0.7, label='SAMS O(n) fit')
        ax4.plot(size_range, linear_pred_ms, 'r--', alpha=0.7, label='MS O(n) fit')
        ax4.plot(size_range, quad_pred_ms, 'r:', alpha=0.7, label='MS O(n²) fit')
        
        ax4.set_xlabel('Dataset Size')
        ax4.set_ylabel('Runtime (seconds)')
        ax4.set_title('Computational Complexity Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Use relative path from repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(repo_root, 'plots', 'corrected_experiment2_scalability.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Scalability analysis plot saved to plots/corrected_experiment2_scalability.png")

def run_corrected_experiment_3():
    """
    CORRECTED Experiment 3: Sample Fraction Sensitivity Analysis
    Following Hyrien & Baran (2017) methodology:
    - Focus on sample fraction as key parameter (paper's main contribution)
    - Test range 0.1% to 1% as specified in paper
    - Multiple dataset types and sizes
    - Analyze speed vs accuracy trade-off
    """
    print("\n" + "="*80)
    print("CORRECTED EXPERIMENT 3: Sample Fraction Sensitivity Analysis")
    print("Key parameter optimization following paper methodology")
    print("="*80)
    
    # Sample fractions as tested in the paper
    sample_fractions = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]  # 0.1% to 10%
    
    # Test on multiple dataset configurations
    test_configs = [
        {'type': 'blobs', 'size': 1000, 'name': 'Gaussian (n=1000)'},
        {'type': 'mixed', 'size': 1000, 'name': 'Mixed (n=1000)'},
        {'type': 'blobs', 'size': 2000, 'name': 'Gaussian (n=2000)'}
    ]
    
    all_results = []
    
    print(f"\nTesting sample fractions: {[f'{f*100:.1f}%' for f in sample_fractions]}")
    print(f"\n{'Dataset':<15} {'Size':<6} {'Sample%':<9} {'ARI':<8} {'Error%':<8} {'Time(s)':<9} {'vs MS ARI':<10} {'Speedup':<8}")
    print("-" * 85)
    
    for config in test_configs:
        print(f"\nTesting {config['name']}...")
        
        # Generate dataset once for all sample fraction tests
        X, y_true = generate_test_data(n_samples=config['size'], dataset_type=config['type'])
        
        # Get baseline mean-shift performance for comparison
        baseline_sams = SAMS_Clustering(bandwidth=None, sample_fraction=0.01, max_iter=200)
        labels_baseline, _ = baseline_sams.fit_predict(X.copy())  # Use copy to avoid side effects
        
        baseline_ms = StandardMeanShift(bandwidth=baseline_sams.bandwidth, max_iter=200)
        
        start_time = time.time()
        labels_ms, _ = baseline_ms.fit_predict(X)
        ms_time = time.time() - start_time
        
        ari_ms, _, _, error_ms = evaluate_clustering(y_true, labels_ms)
        
        for sample_frac in sample_fractions:
            try:
                # Test SAMS with different sample fractions
                sams = SAMS_Clustering(bandwidth=baseline_sams.bandwidth,  # Use same bandwidth for fair comparison
                                     sample_fraction=sample_frac, 
                                     max_iter=200, tol=1e-4)
                
                start_time = time.time()
                labels_sams, _ = sams.fit_predict(X)
                sams_time = time.time() - start_time
                
                ari_sams, nmi_sams, clusters_sams, error_sams = evaluate_clustering(y_true, labels_sams)
                
                # Compute relative performance
                ari_ratio = ari_sams / ari_ms if ari_ms > 0 else 0
                speedup = ms_time / sams_time if sams_time > 0 else 0
                
                print(f"{config['name']:<15} {config['size']:<6} {sample_frac*100:<9.1f} {ari_sams:<8.3f} "
                      f"{error_sams*100:<8.1f} {sams_time:<9.3f} {ari_ratio:<10.3f} {speedup:<8.1f}x")
                
                all_results.append({
                    'config': config['name'],
                    'size': config['size'],
                    'sample_fraction': sample_frac,
                    'sams_ari': ari_sams,
                    'ms_ari': ari_ms,
                    'sams_error': error_sams * 100,
                    'ms_error': error_ms * 100,
                    'sams_time': sams_time,
                    'ms_time': ms_time,
                    'ari_ratio': ari_ratio,
                    'speedup': speedup,
                    'bandwidth': baseline_sams.bandwidth
                })
                
            except Exception as e:
                print(f"{config['name']:<15} {config['size']:<6} {sample_frac*100:<9.1f} ERROR: {str(e)}")
                continue
    
    # Analyze sample fraction effects
    analyze_sample_fraction_effects(all_results)
    
    return all_results

def analyze_sample_fraction_effects(results):
    """Analyze the effect of sample fraction on performance"""
    
    print("\n" + "="*80)
    print("SAMPLE FRACTION ANALYSIS")
    print("="*80)
    
    # Group by configuration
    configs = list(set([r['config'] for r in results]))
    
    print(f"\nOptimal Sample Fraction Analysis:")
    
    for config in configs:
        config_results = [r for r in results if r['config'] == config]
        
        if not config_results:
            continue
            
        # Find optimal trade-offs
        best_quality = max(config_results, key=lambda x: x['sams_ari'])
        best_speed = max(config_results, key=lambda x: x['speedup'])
        
        # Find balanced trade-off (high ARI ratio with reasonable speedup)
        balanced = max([r for r in config_results if r['speedup'] > 1.0], 
                      key=lambda x: x['ari_ratio'], default=best_quality)
        
        print(f"\n{config}:")
        print(f"  Best quality: {best_quality['sample_fraction']*100:.1f}% "
              f"(ARI={best_quality['sams_ari']:.3f}, {best_quality['speedup']:.1f}x speedup)")
        print(f"  Best speed: {best_speed['sample_fraction']*100:.1f}% "
              f"(ARI={best_speed['sams_ari']:.3f}, {best_speed['speedup']:.1f}x speedup)")
        print(f"  Balanced: {balanced['sample_fraction']*100:.1f}% "
              f"(ARI ratio={balanced['ari_ratio']:.3f}, {balanced['speedup']:.1f}x speedup)")
    
    # Paper's recommendations validation
    paper_range_results = [r for r in results if 0.001 <= r['sample_fraction'] <= 0.01]  # 0.1% to 1%
    
    if paper_range_results:
        avg_ari_ratio = np.mean([r['ari_ratio'] for r in paper_range_results])
        avg_speedup = np.mean([r['speedup'] for r in paper_range_results])
        
        print(f"\nPaper's Recommended Range (0.1% - 1.0%):")
        print(f"Average ARI retention: {avg_ari_ratio:.3f} ({avg_ari_ratio*100:.1f}%)")
        print(f"Average speedup: {avg_speedup:.1f}x")
        print(f"Quality loss acceptable (>90% ARI): {'YES' if avg_ari_ratio > 0.9 else 'NO'}")
        print(f"Significant speedup achieved: {'YES' if avg_speedup > 2.0 else 'NO'}")
    
    # Create visualization
    plot_sample_fraction_analysis(results)

def plot_sample_fraction_analysis(results):
    """Create sample fraction analysis plots"""
    
    configs = list(set([r['config'] for r in results]))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['blue', 'red', 'green']
    
    for i, config in enumerate(configs):
        config_results = [r for r in results if r['config'] == config]
        config_results.sort(key=lambda x: x['sample_fraction'])
        
        sample_fracs = [r['sample_fraction'] * 100 for r in config_results]
        ari_ratios = [r['ari_ratio'] for r in config_results]
        speedups = [r['speedup'] for r in config_results]
        times = [r['sams_time'] for r in config_results]
        errors = [r['sams_error'] for r in config_results]
        
        color = colors[i % len(colors)]
        
        # ARI retention vs sample fraction
        ax1.plot(sample_fracs, ari_ratios, 'o-', color=color, label=config, linewidth=2, markersize=6)
        
        # Speedup vs sample fraction
        ax2.plot(sample_fracs, speedups, 'o-', color=color, label=config, linewidth=2, markersize=6)
        
        # Runtime vs sample fraction
        ax3.plot(sample_fracs, times, 'o-', color=color, label=config, linewidth=2, markersize=6)
        
        # Error rate vs sample fraction
        ax4.plot(sample_fracs, errors, 'o-', color=color, label=config, linewidth=2, markersize=6)
    
    # Customize plots
    ax1.set_xlabel('Sample Fraction (%)')
    ax1.set_ylabel('ARI Ratio (SAMS/MS)')
    ax1.set_title('Clustering Quality Retention')
    ax1.axhline(y=0.9, color='black', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    ax2.set_xlabel('Sample Fraction (%)')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.set_title('Speed Improvement')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    ax3.set_xlabel('Sample Fraction (%)')
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('SAMS Runtime vs Sample Fraction')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    ax4.set_xlabel('Sample Fraction (%)')
    ax4.set_ylabel('Clustering Error Rate (%)')
    ax4.set_title('Error Rate vs Sample Fraction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    # Use relative path from repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(repo_root, 'plots', 'corrected_experiment3_sample_fraction.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Sample fraction analysis plot saved to plots/corrected_experiment3_sample_fraction.png")

def run_all_corrected_experiments():
    """Run all corrected experiments"""
    print("CORRECTED EXPERIMENTAL VALIDATION")
    print("Fixing methodological issues in Experiments 2 & 3")
    print("="*70)
    
    np.random.seed(42)
    
    # Run corrected experiments
    print("\n" + "="*50)
    print("Running Corrected Experiment 2...")
    exp2_results = run_corrected_experiment_2()
    
    print("\n" + "="*50)
    print("Running Corrected Experiment 3...")
    exp3_results = run_corrected_experiment_3()
    
    # Final summary
    print("\n" + "="*80)
    print("CORRECTED EXPERIMENTS 2 & 3: COMPLETE")
    print("="*80)
    
    print("\n✓ Experiment 2 (Scalability):")
    print("  - Fair comparison with same datasets and bandwidth")
    print("  - Constant sample fraction to isolate scalability effects")
    print("  - Computational complexity analysis")
    
    print("\n✓ Experiment 3 (Sample Fraction):")
    print("  - Focus on key parameter from paper")
    print("  - Speed vs accuracy trade-off analysis")
    print("  - Paper's recommended range validation")
    
    return exp2_results, exp3_results

if __name__ == "__main__":
    exp2_results, exp3_results = run_all_corrected_experiments()