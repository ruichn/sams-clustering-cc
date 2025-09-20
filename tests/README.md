# SAMS Clustering Tests

This directory contains comprehensive tests for the SAMS (Stochastic Approximation Mean-Shift) clustering implementation.

## Test Structure

### Core Tests
- **`test_imports.py`** - Validates that all SAMS modules import correctly
- **`test_paths.py`** - Tests path resolution for plots and source files  
- **`test_experiment.py`** - Validates experiment functions work correctly

### 3D Functionality Tests
- **`test_3d_capability.py`** - Basic 3D clustering functionality test
- **`test_3d_demo.py`** - Comprehensive 3D demo application testing
- **`experiment_3d_validation.py`** - Full 3D clustering validation experiment

### Test Data & Reports
- **`3D_VALIDATION_REPORT.txt`** - Detailed 3D validation results
- **`3d_validation_report.py`** - Report generation script

## Running Tests

### Run All Tests
```bash
cd tests
python run_all_tests.py
```

### Run Individual Tests
```bash
# Test imports and basic functionality
python test_imports.py

# Test 3D capabilities
python test_3d_capability.py

# Test demo functionality
python test_3d_demo.py

# Run comprehensive 3D experiment
python experiment_3d_validation.py
```

### Run From Repository Root
```bash
# All tests
python tests/run_all_tests.py

# Individual test
python tests/test_imports.py
```

## Test Coverage

### ✅ **Import Resolution**
- Validates SAMS_Clustering, StandardMeanShift imports
- Tests data generation functions
- Verifies module instantiation

### ✅ **Path Resolution** 
- Tests relative path resolution from different directories
- Validates plot output directory paths
- Confirms source file locations

### ✅ **2D Clustering**
- Basic SAMS functionality on 2D data
- Performance comparison with mean-shift
- Visualization generation

### ✅ **3D Clustering**
- SAMS algorithm on 3-dimensional data
- 3D dataset generation (blobs, spheres, extended 2D)
- 3D visualization with matplotlib
- Performance validation

### ✅ **Demo Application**
- Streamlit app functionality
- Dataset generation across dimensions
- UI parameter handling
- Plot generation and display

## Expected Results

When all tests pass, you should see:
```
✅ Import Resolution PASSED
✅ Path Resolution PASSED  
✅ Experiment Functions PASSED
✅ 3D Basic Capability PASSED
✅ 3D Demo Functionality PASSED

🎉 ALL TESTS PASSED! SAMS implementation is working correctly.
```

## Troubleshooting

### Import Errors
- Ensure you're running from the repository root or tests directory
- Check that the virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Path Errors
- All paths are automatically resolved relative to repository root
- Ensure plots/ directory exists
- Check file permissions

### 3D Test Failures
- Requires matplotlib with 3D support (`mpl_toolkits.mplot3d`)
- May need updated scipy/sklearn versions
- Check that test plots can be saved to plots/ directory

## Integration with CI/CD

These tests can be integrated into continuous integration pipelines:

```yaml
# Example GitHub Actions step
- name: Run SAMS Tests
  run: |
    pip install -r requirements.txt
    python tests/run_all_tests.py
```

## Test Development

When adding new features:
1. Add corresponding tests to appropriate test files
2. Update test runner if needed
3. Ensure paths are properly resolved from tests/ directory
4. Follow existing test patterns for consistency