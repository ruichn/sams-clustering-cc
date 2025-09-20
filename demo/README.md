---
title: SAMS Clustering Demo
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
license: mit
---

# 🔬 SAMS Clustering Interactive Demo

An interactive web application for exploring the **Stochastic Approximation Mean-Shift (SAMS)** clustering algorithm from:

> Hyrien, O., & Baran, R. H. (2017). *Fast Nonparametric Density-Based Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift Algorithm*. PMC5417725.

## 🚀 Features

### 📊 **Interactive Data Generation**
- **Dataset Types**: Gaussian Blobs, Concentric Circles, Two Moons, Mixed Densities
- **Customizable Parameters**: Sample size (500-100,000), number of clusters, noise levels
- **Real-time Visualization**: Clean matplotlib plots with consistent styling

### ⚙️ **Algorithm Configuration**
- **SAMS Parameters**: Sample fraction, bandwidth selection, maximum iterations
- **Comparison Mode**: Optional scikit-learn mean-shift comparison
- **Performance Metrics**: ARI, NMI, Silhouette score, runtime analysis

### 📈 **Validated Performance**
- **74-106x speedup** over standard mean-shift clustering
- **91-99% quality retention** (ARI preservation)
- **Experiment-style plotting** using the same functions as research validation

## 🔧 Technical Implementation

- **Pure matplotlib plotting** for consistent visualization
- **Standalone SAMS implementation** optimized for demo use
- **Vectorized computations** for efficient performance
- **Interactive parameter controls** with real-time feedback

## 📍 Full Implementation

Complete research implementation with experimental validation: [github.com/ruichn/sams-clustering-cc](https://github.com/ruichn/sams-clustering-cc)

## 🎯 Usage

1. Select dataset type and parameters in the sidebar
2. Configure SAMS algorithm settings  
3. Click "Generate Data & Run Clustering"
4. Explore results with interactive visualizations
5. Compare performance metrics and runtime analysis