# Experimental Results and Plots

This directory contains the generated plots from the SAMS algorithm validation experiments.

## Experiment Results

### 1. Dataset Performance Comparison (`experiment1_results.png`)
- **Left column**: True cluster labels for different dataset types
- **Middle column**: SAMS clustering results with ARI scores
- **Right column**: Standard Mean-Shift results for comparison
- **Datasets tested**: Blobs, Circles, Mixed density clusters

### 2. Scalability Analysis (`experiment2_scalability.png`)
- **Left plot**: Runtime vs Dataset Size - demonstrates O(n) scaling
- **Right plot**: Clustering Quality vs Dataset Size - maintains accuracy
- **Dataset sizes**: 100 to 5,000 data points
- **Key finding**: Linear scaling with dataset size

### 3. Parameter Sensitivity (`experiment3_sensitivity.png`)
- **Top row**: Bandwidth parameter effects on ARI and cluster count
- **Bottom row**: Sample fraction effects on ARI and runtime
- **Key insights**: 
  - Bandwidth controls cluster granularity
  - Sample fraction provides speed/accuracy trade-off

### 4. Image Segmentation (`image_segmentation_results.png`)
- **Rows**: Different image configurations (Small 4-region, Medium 3-region, Large 4-region)
- **Columns**: Original image + different feature extraction methods
- **Feature types**: Intensity-only, Intensity+Position, Intensity+Gradient
- **Demonstrates**: SAMS effectiveness on computer vision tasks

### 5. Color Image Segmentation (`color_segmentation_results.png`)
- **Left**: Original RGB color image with 4 distinct color regions
- **Right**: SAMS segmentation result using RGB+position features
- **Shows**: Multi-dimensional feature space handling (5D features)

## Key Experimental Findings

1. **Performance**: SAMS achieves competitive clustering quality with significant speedup
2. **Scalability**: Linear time complexity confirmed through empirical testing
3. **Versatility**: Effective for both standard clustering and image segmentation
4. **Parameter Robustness**: Clear guidelines for bandwidth and sample fraction selection

## Reproduction

To regenerate these plots:

```bash
# Activate virtual environment
source sams_env/bin/activate

# Run experiments
python experiments.py
python image_segmentation.py
```

All plots are saved with high DPI (300) for publication quality.