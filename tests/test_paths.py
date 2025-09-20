#!/usr/bin/env python3
"""Test script to verify path resolution in experiments"""

import sys
import os

# Test from repository root (from tests directory)
print("=== Testing from repository root ===")
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plots_dir = os.path.join(repo_root, 'plots')
print(f"✅ Repository root: {repo_root}")
print(f"✅ Plots directory: {plots_dir}")
print(f"✅ Plots directory exists: {os.path.exists(plots_dir)}")

# Test the path resolution logic used in experiments
print("\n=== Testing experiments path resolution ===")
# Simulate being in src/experiments/
fake_experiment_file = os.path.join(repo_root, 'src', 'experiments', 'fake_experiment.py')
exp_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(fake_experiment_file)))
exp_plots_dir = os.path.join(exp_repo_root, 'plots')
print(f"✅ Experiment repo root: {exp_repo_root}")
print(f"✅ Experiment plots directory: {exp_plots_dir}")
print(f"✅ Experiment plots directory exists: {os.path.exists(exp_plots_dir)}")
print(f"✅ Paths match: {repo_root == exp_repo_root}")

# Test import path
print("\n=== Testing import paths ===")
src_dir = os.path.join(repo_root, 'src')
print(f"✅ Src directory: {src_dir}")
print(f"✅ Src directory exists: {os.path.exists(src_dir)}")
sams_file = os.path.join(src_dir, 'sams_clustering.py')
print(f"✅ SAMS file: {sams_file}")
print(f"✅ SAMS file exists: {os.path.exists(sams_file)}")