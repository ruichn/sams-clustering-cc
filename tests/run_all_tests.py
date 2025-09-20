#!/usr/bin/env python3
"""
Comprehensive test runner for SAMS clustering implementation

This script runs all tests to validate:
1. Basic imports and path resolution
2. 2D clustering functionality  
3. 3D clustering functionality
4. Demo application functionality
5. Experiment file execution
"""

import sys
import os
import subprocess
import time

# Add repository root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

def run_test(test_name, test_file):
    """Run a single test and report results"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {test_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, 
                              cwd=os.path.dirname(test_file))
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({elapsed:.2f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {test_name} FAILED ({elapsed:.2f}s)")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ {test_name} FAILED - Exception: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 SAMS CLUSTERING - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define test suite
    tests = [
        ("Import Resolution", os.path.join(tests_dir, "test_imports.py")),
        ("Path Resolution", os.path.join(tests_dir, "test_paths.py")),
        ("Experiment Functions", os.path.join(tests_dir, "test_experiment.py")),
        ("3D Basic Capability", os.path.join(tests_dir, "test_3d_capability.py")),
        ("3D Demo Functionality", os.path.join(tests_dir, "test_3d_demo.py")),
    ]
    
    # Run tests
    results = {}
    total_tests = len(tests)
    
    for test_name, test_file in tests:
        if os.path.exists(test_file):
            results[test_name] = run_test(test_name, test_file)
        else:
            print(f"⚠️  {test_name} - Test file not found: {test_file}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(results.values())
    failed = total_tests - passed
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/total_tests*100:.1f}%")
    
    if failed == 0:
        print(f"\n🎉 ALL TESTS PASSED! SAMS implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())