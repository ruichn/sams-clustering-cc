#!/usr/bin/env python3
"""Test script to verify experiment imports work correctly"""

import sys
import os

# Simulate being in src/experiments/ directory (from tests directory)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
experiment_dir = os.path.join(repo_root, 'src', 'experiments')
original_cwd = os.getcwd()

try:
    # Add src directory to path
    src_dir = os.path.join(repo_root, 'src')
    sys.path.append(src_dir)
    
    # Add experiments directory to path
    sys.path.append(experiment_dir)
    
    # Change to experiments directory temporarily
    os.chdir(experiment_dir)
    print(f"✅ Changed to directory: {os.getcwd()}")
    
    # Import from experiment1_basic_performance 
    from experiment1_basic_performance import evaluate_clustering
    print("✅ Successfully imported evaluate_clustering from experiment1")
    
    # Test the function
    import numpy as np
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    ari, nmi, n_clusters = evaluate_clustering(y_true, y_pred)
    print(f"✅ Function works: ARI={ari:.3f}, NMI={nmi:.3f}, clusters={n_clusters}")
    
finally:
    # Restore original directory
    os.chdir(original_cwd)