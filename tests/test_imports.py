#!/usr/bin/env python3
"""Test script to verify imports work correctly"""

import sys
import os

# Add src directory to path (from tests directory)
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
sys.path.append(src_dir)

print("=== Testing imports ===")
print(f"✅ Added to sys.path: {src_dir}")

try:
    from sams_clustering import SAMS_Clustering, StandardMeanShift, generate_test_data
    print("✅ Successfully imported SAMS_Clustering")
    print("✅ Successfully imported StandardMeanShift") 
    print("✅ Successfully imported generate_test_data")
    
    # Test instantiation
    sams = SAMS_Clustering()
    print("✅ Successfully instantiated SAMS_Clustering")
    
    ms = StandardMeanShift()
    print("✅ Successfully instantiated StandardMeanShift")
    
    # Test data generation
    X, y = generate_test_data(n_samples=100, dataset_type='blobs')
    print(f"✅ Successfully generated test data: {X.shape}, {y.shape}")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()