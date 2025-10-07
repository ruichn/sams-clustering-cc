"""
SAMS Clustering Demo - v2.1.0
Numerical stability improvements and enhanced UI
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

# Removed Plotly imports - using matplotlib only
# Import the main SAMS implementation
import sys
import os

# Add src directory to Python path for both local and Hugging Face environments
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from sams_clustering import SAMS_Clustering
from standard_meanshift import StandardMeanShift

# Page configuration
st.set_page_config(
    page_title="SAMS Clustering Demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("SAMS: Stochastic Approximation Mean-Shift Clustering Demo")
    st.markdown("Interactive demonstration of the SAMS clustering algorithm supporting arbitrary dimensions (1D-128D+).")
    st.markdown("""
    **Interactive simulation studies for:** *"Fast Nonparametric Density-Based Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift Algorithm"* by Hyrien & Baran (2016)
        
    Explore the SAMS algorithm with customizable parameters and compare performance against standard mean-shift clustering.
    """)
    
    # Sidebar for simulation parameters
    with st.sidebar:
        st.header("Simulation Parameters")
        
        # Data generation parameters
        st.subheader("Data Generation")
        
        # Data type (distribution shape)
        data_type = st.selectbox(
            "Data Type",
            ["Gaussian Blobs", "Mixed Densities", "Two Moons", "Image Segmentation"],
            help="Choose the distribution pattern/shape of the synthetic dataset"
        )
        
        # Number of dimensions (features)
        if data_type == "Image Segmentation":
            feature_type = st.selectbox(
                "Feature Extraction",
                ["Intensity + Position", "Intensity Only"],
                index=0,
                help="Choose feature extraction method for image segmentation"
            )
            # Map UI selection to internal format and dimensions
            if feature_type == "Intensity + Position":
                feature_type = "intensity_position"
                n_features = 3
            elif feature_type == "Intensity Only":
                feature_type = "intensity_only"
                n_features = 1
            elif feature_type == "Position Only":
                feature_type = "position_only"
                n_features = 2
            else:  # Intensity + Gradient
                feature_type = "intensity_gradient"
                n_features = 2
            st.info(f"**Image Segmentation** - {n_features}D clustering with {feature_type.replace('_', ' ')} features")
        else:
            n_features = st.number_input(
                "Number of Dimensions",
                min_value=1,
                value=2,
                step=1,
                help="Choose dimensionality. Algorithm supports arbitrary dimensions. Higher dimensions will use PCA for visualization."
            )
        
        # Sample size
        if data_type != "Image Segmentation":
            n_samples = st.number_input(
                "Sample Size (n)",
                min_value=500,
                value=1000,
                step=100,
                help="Number of data points to generate (no upper limit for scalability testing)"
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                image_size = st.selectbox(
                    "Image Size",
                    ["40x40", "60x60", "80x80"],
                    index=1,
                    help="Size of synthetic image to generate"
                )
            with col2:
                n_centers = st.slider(
                    "Image Regions",
                    min_value=3,
                    max_value=5,
                    value=4,
                    help="Number of distinct regions in the image"
                )
            # Parse image size
            img_width, img_height = map(int, image_size.split('x'))
            n_samples = img_width * img_height  # Override sample size for images
        
        # Warning for very large datasets
        if n_samples > 1000000:
            st.error(f"**Very Large Dataset**: {n_samples:,} points may require significant memory and processing time. Recommended sample fraction: ≤0.1%")
        elif n_samples > 100000:
            st.warning(f"**Large Dataset**: {n_samples:,} points selected. Consider using lower sample fractions (0.1-0.5%) for faster processing.")
        
        # Number of clusters (for applicable data types)
        if data_type in ["Gaussian Blobs", "Mixed Densities"]:
            n_centers = st.number_input(
                "Number of Clusters",
                min_value=2,
                value=4,
                step=1,
                help="Number of clusters in the dataset"
            )
        elif data_type == "Two Moons":
            n_centers = 2
        
        # Noise level
        noise_level = st.slider(
            "Noise Level",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.02,
            help="Amount of noise to add to the data"
        )
        
        # Cluster standard deviation (for blobs)
        if data_type == "Gaussian Blobs":
            cluster_std = st.slider(
                "Cluster Std Dev",
                min_value=0.5,
                max_value=2.5,
                value=1.2,
                step=0.1,
                help="Standard deviation of Gaussian clusters"
            )
        
        st.subheader("Algorithm Parameters")
        
        # SAMS Parameters
        with st.expander("🔬 SAMS Algorithm Parameters", expanded=True):
            # Method selection
            run_both_sams = st.checkbox(
                "Run both SAMS methods (fit_predict & fit)",
                value=True,
                help="Compare both SAMS implementations side-by-side"
            )

            # Bandwidth selection
            adaptive_bandwidth = st.checkbox(
                "Use adaptive bandwidth estimate",
                value=True,
                help="Adjust bandwidth per iteration"
            )
            sams_bandwidth_mode = st.radio(
                "Bandwidth Selection",
                ["Automatic (Silverman's rule)", "Manual"],
                help="Choose how to set the bandwidth parameter for SAMS",
                key="sams_bandwidth_mode"
            )
            
            if sams_bandwidth_mode == "Manual":
                sams_bandwidth = st.slider(
                    "Bandwidth",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.05,
                    help="Bandwidth parameter for kernel density estimation",
                    key="sams_bandwidth"
                )
            else:
                sams_bandwidth = None
            
            # Sample fraction parameter (SAMS only)
            sample_fraction = st.number_input(
                "Sample Fraction (%)",
                min_value=0.1,
                max_value=100.0,
                value=2.0,
                step=0.1,
                help="Percentage of data points to sample at each iteration. Lower values enable processing of very large datasets. 100% = standard mean-shift. This parameter is specific to SAMS algorithm.",
                key="sams_sample_fraction"
            ) / 100
            
            # Max iterations
            sams_max_iter = st.number_input(
                "Max Iterations",
                min_value=50,
                max_value=5000,
                value=300,
                step=50,
                help="Maximum number of iterations for SAMS",
                key="sams_max_iter"
            )
            
            # Advanced step size parameters (no nested expander)
            st.markdown("**Advanced Step Size Parameters:**")            
            alpha1_mode = st.radio(
                "Alpha1 (Scott's Rule Parameter)",
                ["Automatic", "Manual"],
                help="Alpha1 controls the base bandwidth scaling using Scott's rule",
                key="sams_alpha1_mode"
            )
            
            if alpha1_mode == "Manual":
                alpha1 = st.slider(
                    "Alpha1 Value",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.5,
                    step=0.01,
                    help="Manual alpha1 value for bandwidth computation",
                    key="sams_alpha1"
                )
            else:
                alpha1 = None
            
            alpha2_mode = st.radio(
                "Alpha2 (Dimension Factor)",
                ["Automatic", "Manual"],
                help="Alpha2 controls dimension-dependent bandwidth adjustment",
                key="sams_alpha2_mode"
            )
            
            if alpha2_mode == "Manual":
                alpha2 = st.slider(
                    "Alpha2 Value",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Manual alpha2 value for dimension adjustment",
                    key="sams_alpha2"
                )
            else:
                alpha2 = None
        
        # Comparison options
        st.subheader("Comparison")
        compare_standard = st.checkbox(
            "Include Standard Mean-Shift",
            value=True,
            help="Compare with our StandardMeanShift implementation"
        )

        standard_bandwidth_mode = None
        standard_bandwidth = None
        standard_max_iter = 300

        if compare_standard:
            # Standard Mean-Shift Parameters
            with st.expander("📊 Standard Mean-Shift Parameters", expanded=False):
                
                standard_max_iter = st.slider(
                    "Max Iterations",
                    min_value=50,
                    max_value=5000,
                    value=300,
                    step=25,
                    help="Maximum number of iterations for Standard Mean-Shift",
                    key="standard_max_iter"
                )

                standard_bandwidth_mode = st.radio(
                    "Bandwidth Selection",
                    ["Automatic (Silverman's rule)", "Use SAMS Bandwidth", "Manual"],
                    help="Choose how to set the bandwidth parameter for Standard Mean-Shift",
                    key="standard_bandwidth_mode"
                )
                
                if standard_bandwidth_mode == "Manual":
                    standard_bandwidth = st.slider(
                        "Bandwidth",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.5,
                        step=0.05,
                        help="Bandwidth parameter for Standard Mean-Shift",
                        key="standard_bandwidth"
                    )
        
        compare_sklearn = st.checkbox(
            "Include Scikit-Learn Mean-Shift",
            value=True,
            help="Compare with sklearn.cluster.MeanShift. Each algorithm uses its own optimal bandwidth estimation for fair comparison."
        )

        sklearn_bandwidth_mode = None
        sklearn_bandwidth = None
        sklearn_max_iter = 300

        if compare_sklearn:
            # Scikit-Learn Mean-Shift Parameters
            with st.expander("🔧 Scikit-Learn Mean-Shift Parameters", expanded=False):
                
                sklearn_max_iter = st.slider(
                    "Max Iterations",
                    min_value=50,
                    max_value=5000,
                    value=300,
                    step=25,
                    help="Maximum number of iterations for Scikit-Learn Mean-Shift",
                    key="sklearn_max_iter"
                )

                sklearn_bandwidth_mode = st.radio(
                    "Bandwidth Selection",
                    ["Automatic (estimate_bandwidth)", "Use Standard Mean-Shift Bandwidth", "Manual"] if compare_standard else ["Automatic (estimate_bandwidth)", "Use SAMS Bandwidth", "Manual"],
                    help="Choose how to set the bandwidth parameter for Scikit-Learn Mean-Shift",
                    key="sklearn_bandwidth_mode"
                )
                
                if sklearn_bandwidth_mode == "Manual":
                    sklearn_bandwidth = st.slider(
                        "Bandwidth",
                        min_value=0.1,
                        max_value=2.0,
                        value=0.5,
                        step=0.05,
                        help="Bandwidth parameter for Scikit-Learn Mean-Shift",
                        key="sklearn_bandwidth"
                    )
        
        # Random seed
        random_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="Seed for reproducible results"
        )
    
    # Main content area - full width
    st.subheader("Clustering Results")
    
    # Generate data button
    if st.button("🔄 Generate Data & Run Clustering", type="primary"):
        # Pass additional parameters for image segmentation
        extra_params = {}
        if data_type == "Image Segmentation":
            extra_params['feature_type'] = feature_type
            extra_params['image_size'] = (img_height, img_width)
        
        run_clustering_experiment(data_type, n_samples, n_centers, noise_level,
                                cluster_std if data_type == "Gaussian Blobs" else None,
                                sams_bandwidth, sample_fraction, sams_max_iter,
                                standard_bandwidth_mode, standard_bandwidth, standard_max_iter,
                                sklearn_bandwidth_mode, sklearn_bandwidth, sklearn_max_iter,
                                compare_standard, compare_sklearn,
                                random_seed, n_features, extra_params, alpha1, alpha2, adaptive_bandwidth, run_both_sams)

def run_clustering_experiment(data_type, n_samples, n_centers, noise_level, cluster_std,
                            sams_bandwidth, sample_fraction, sams_max_iter,
                            standard_bandwidth_mode, standard_bandwidth, standard_max_iter,
                            sklearn_bandwidth_mode, sklearn_bandwidth, sklearn_max_iter,
                            compare_standard, compare_sklearn,
                            random_seed, n_features=2, extra_params=None, alpha1=None, alpha2=None, adaptive_bandwidth=True, run_both_sams=True):
    """Run the complete clustering experiment"""
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Generate dataset
    with st.spinner("Generating dataset..."):
        if extra_params is None:
            extra_params = {}
        result = generate_dataset(data_type, n_samples, n_centers, noise_level, cluster_std, n_features, **extra_params)
        
        # Handle different return types (image segmentation returns 3 values)
        if len(result) == 3:
            X, y_true, original_image = result
        else:
            X, y_true = result
            original_image = None
    
    st.success(f"Generated {data_type} dataset with {n_samples:,} points")
    
    # Run clustering methods
    results = {}

    # SAMS clustering - run both methods or just fit_predict
    if run_both_sams:
        # Run fit_predict
        with st.spinner("Running SAMS fit_predict clustering..."):
            sams_fp = SAMS_Clustering(bandwidth=sams_bandwidth, sample_fraction=sample_fraction, max_iter=sams_max_iter, alpha1=alpha1, alpha2=alpha2, adaptive_bandwidth=adaptive_bandwidth)

            start_time = time.time()
            labels_sams_fp, centers_sams_fp = sams_fp.fit_predict(X)
            sams_fp_time = time.time() - start_time

            results['SAMS (fit_predict)'] = {
                'labels': labels_sams_fp,
                'centers': centers_sams_fp,
                'time': sams_fp_time,
                'bandwidth': sams_fp.bandwidth,
                'n_clusters': len(np.unique(labels_sams_fp))
            }

        # Run fit
        with st.spinner("Running SAMS fit clustering..."):
            sams_f = SAMS_Clustering(bandwidth=sams_bandwidth, sample_fraction=sample_fraction, max_iter=sams_max_iter, alpha1=alpha1, alpha2=alpha2, adaptive_bandwidth=adaptive_bandwidth)

            start_time = time.time()
            labels_sams_f, centers_sams_f = sams_f.fit(X)
            sams_f_time = time.time() - start_time

            results['SAMS (fit)'] = {
                'labels': labels_sams_f,
                'centers': centers_sams_f,
                'time': sams_f_time,
                'bandwidth': sams_f.bandwidth,
                'n_clusters': len(np.unique(labels_sams_f))
            }
    else:
        # Run only fit_predict
        with st.spinner("Running SAMS fit_predict clustering..."):
            sams = SAMS_Clustering(bandwidth=sams_bandwidth, sample_fraction=sample_fraction, max_iter=sams_max_iter, alpha1=alpha1, alpha2=alpha2, adaptive_bandwidth=adaptive_bandwidth)

            start_time = time.time()
            labels_sams, centers_sams = sams.fit_predict(X)
            sams_time = time.time() - start_time

            results['SAMS'] = {
                'labels': labels_sams,
                'centers': centers_sams,
                'time': sams_time,
                'bandwidth': sams.bandwidth,
                'n_clusters': len(np.unique(labels_sams))
            }
    
    # Standard Mean-Shift (if requested)
    if compare_standard:
        with st.spinner("Running Standard Mean-Shift..."):
            try:
                if standard_bandwidth_mode == "Use SAMS Bandwidth":
                    standard_bandwidth = sams.bandwidth

                standard_ms = StandardMeanShift(bandwidth=standard_bandwidth, max_iter=standard_max_iter)
                
                start_time = time.time()
                standard_labels, standard_centers = standard_ms.fit_predict(X)
                standard_time = time.time() - start_time
                
                results['Standard Mean-Shift'] = {
                    'labels': standard_labels,
                    'centers': standard_centers,
                    'time': standard_time,
                    'bandwidth': standard_ms.bandwidth,
                    'n_clusters': len(np.unique(standard_labels))
                }
            except Exception as e:
                st.warning(f"Standard Mean-Shift failed: {str(e)}")
    
    # Scikit-learn Mean-Shift (if requested)
    if compare_sklearn:
        with st.spinner("Running Scikit-Learn Mean-Shift..."):
            try:
                if sklearn_bandwidth_mode == "Use SAMS Bandwidth":
                    sklearn_bandwidth = sams.bandwidth
                elif sklearn_bandwidth_mode == "Use Standard Mean-Shift Bandwidth":
                    sklearn_bandwidth = standard_ms.bandwidth
                
                # Validate bandwidth for sklearn MeanShift to prevent numerical warnings
                if sklearn_bandwidth is not None:
                    # Ensure bandwidth is finite and within reasonable bounds
                    if not np.isfinite(sklearn_bandwidth) or sklearn_bandwidth <= 0:
                        sklearn_bandwidth = None  # Fall back to automatic estimation
                    else:
                        # Clip to reasonable range to prevent sklearn numerical issues
                        sklearn_bandwidth = max(min(sklearn_bandwidth, 10.0), 1e-3)
                
                print(f"Starting Scikit-learn Mean-Shift with {n_samples} points")
                
                ms = MeanShift(bandwidth=sklearn_bandwidth, max_iter=sklearn_max_iter)
                start_time = time.time()
                labels_ms = ms.fit_predict(X)
                ms_time = time.time() - start_time
                
                # Get the actual bandwidth used by MeanShift
                if sklearn_bandwidth is None:
                    # MeanShift used internal estimate_bandwidth()
                    actual_bandwidth = estimate_bandwidth(X)
                else:
                    actual_bandwidth = sklearn_bandwidth
                
                print(f"Bandwidth: {actual_bandwidth:.4f}")
                
                results['Scikit-Learn Mean-Shift'] = {
                    'labels': labels_ms,
                    'centers': ms.cluster_centers_,
                    'time': ms_time,
                    'bandwidth': actual_bandwidth,
                    'n_clusters': len(np.unique(labels_ms))
                }
            except Exception as e:
                st.warning(f"Scikit-Learn Mean-Shift failed: {str(e)}")
    
    # Display results
    display_results(X, y_true, results, data_type, sample_fraction, original_image)

def generate_synthetic_image(size=(60, 60), n_regions=4, noise_level=0.05):
    """Generate synthetic image for segmentation testing"""
    # Create different regions with varying intensities
    image = np.zeros(size)
    
    if n_regions == 4:
        # Four quadrant regions
        image[0:size[0]//2, 0:size[1]//2] = 0.8  # Top-left: bright
        image[0:size[0]//2, size[1]//2:] = 0.3   # Top-right: dark
        image[size[0]//2:, 0:size[1]//2] = 0.6   # Bottom-left: medium
        image[size[0]//2:, size[1]//2:] = 0.9    # Bottom-right: very bright
    
    elif n_regions == 3:
        # Three circular regions
        center1 = (size[0]//4, size[1]//4)
        center2 = (3*size[0]//4, size[1]//4)
        center3 = (size[0]//2, 3*size[1]//4)
        
        for i in range(size[0]):
            for j in range(size[1]):
                dist1 = np.sqrt((i - center1[0])**2 + (j - center1[1])**2)
                dist2 = np.sqrt((i - center2[0])**2 + (j - center2[1])**2)
                dist3 = np.sqrt((i - center3[0])**2 + (j - center3[1])**2)
                
                min_dist = min(dist1, dist2, dist3)
                if min_dist == dist1:
                    image[i, j] = 0.8
                elif min_dist == dist2:
                    image[i, j] = 0.4
                else:
                    image[i, j] = 0.6
    
    elif n_regions == 5:
        # Five random regions
        np.random.seed(42)
        regions = np.random.choice(5, size=size) * 0.2
        image = regions
    
    # Add noise
    noise = np.random.normal(0, noise_level, size)
    image = np.clip(image + noise, 0, 1)
    
    return image

def extract_features_from_image(image, feature_type='intensity_position'):
    """Extract features from image for clustering"""
    h, w = image.shape
    features = []
    
    for i in range(h):
        for j in range(w):
            if feature_type == 'intensity_only':
                # Only pixel intensity
                features.append([image[i, j]])
            elif feature_type == 'position_only':
                # Only spatial position (normalized)
                features.append([i/h, j/w])
            elif feature_type == 'intensity_position':
                # Both intensity and spatial position
                features.append([image[i, j], i/h, j/w])
            elif feature_type == 'intensity_gradient':
                # Intensity and local gradient
                if i > 0 and i < h-1 and j > 0 and j < w-1:
                    grad_x = image[i, j+1] - image[i, j-1]
                    grad_y = image[i+1, j] - image[i-1, j]
                    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                    features.append([image[i, j], gradient_mag])
                else:
                    features.append([image[i, j], 0])
    
    return np.array(features)

def generate_dataset(data_type, n_samples, n_centers, noise_level, cluster_std=None, n_features=2, feature_type=None, image_size=None):
    """Generate synthetic datasets based on user parameters"""
    
    if data_type == "Gaussian Blobs":
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=42
        )
        # Add noise
        X += np.random.normal(0, noise_level, X.shape)
                        
    elif data_type == "Mixed Densities":
        # Create clusters with different densities
        np.random.seed(42)
        
        # Generate n_centers clusters with varying densities
        clusters = []
        labels = []
        
        # Define density levels (variance) for different clusters
        density_levels = np.linspace(0.3, 2.0, n_centers)  # From high to low density
        
        # Generate cluster centers
        centers = np.random.uniform(-3, 3, (n_centers, n_features))
        
        # Distribute samples across clusters
        samples_per_cluster = [n_samples // n_centers] * n_centers
        samples_per_cluster[-1] += n_samples % n_centers  # Add remainder to last cluster
        
        for i in range(n_centers):
            n_cluster = samples_per_cluster[i]
            center = centers[i]
            density = density_levels[i]
            
            cov_matrix = np.eye(n_features) * density
            
            cluster_data = np.random.multivariate_normal(center, cov_matrix, size=n_cluster)
            clusters.append(cluster_data)
            labels.extend([i] * n_cluster)
        
        X = np.vstack(clusters)
        y_true = np.array(labels)
        
        # Add noise
        X += np.random.normal(0, noise_level, X.shape)
    
    elif data_type == "Image Segmentation":
        # Generate synthetic image
        if image_size is None:
            size = (60, 60)
        else:
            size = image_size
        
        image = generate_synthetic_image(size=size, n_regions=n_centers, noise_level=noise_level)
        
        # Extract features from image
        X = extract_features_from_image(image, feature_type)
        
        # Create ground truth labels based on pixel regions
        # For demo purposes, we'll create simple region-based labels
        h, w = image.shape
        y_true = []
        for i in range(h):
            for j in range(w):
                # Simple quadrant-based labeling for ground truth
                if n_centers == 4:
                    if i < h//2 and j < w//2:
                        y_true.append(0)  # Top-left
                    elif i < h//2 and j >= w//2:
                        y_true.append(1)  # Top-right
                    elif i >= h//2 and j < w//2:
                        y_true.append(2)  # Bottom-left
                    else:
                        y_true.append(3)  # Bottom-right
                else:
                    # For other region counts, use position-based labeling
                    y_true.append((i // (h // n_centers)) * n_centers + (j // (w // n_centers)))
        
        y_true = np.array(y_true)
        
        # Don't standardize image features as they have specific meaning
        # Return tuple with image for special visualization
        return X, y_true.astype(int), image
    
    else:
        # Default case: fallback to standard Gaussian blobs if no matching dataset type
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            cluster_std=cluster_std if cluster_std else 1.5,
            random_state=42
        )
        # Add noise
        X += np.random.normal(0, noise_level, X.shape)
        
    return X, y_true.astype(int)

def display_results(X, y_true, results, data_type, sample_fraction, original_image=None):
    """Display clustering results and comparisons"""
    
    # Special handling for image segmentation
    if data_type == "Image Segmentation" and original_image is not None:
        st.subheader("Original Image")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(original_image, cmap='gray')
        ax.set_title("Synthetic Image for Segmentation")
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
    
    # Main clustering visualization
    st.subheader("Clustering Results Comparison")
    fig = create_clustering_plot(X, y_true, results)
    st.pyplot(fig)
    
    # Performance metrics table
    st.subheader("Performance Metrics")
    metrics_df = calculate_metrics(X, y_true, results)
    st.dataframe(metrics_df, width='stretch')
    
    # Bottom section: Runtime comparison with experiment summary and performance comparison
    if len(results) > 1:
        st.subheader("Runtime Comparison")
        
        # Create three columns for the bottom section
        bottom_col1, bottom_col2, bottom_col3 = st.columns([1, 1, 2])
        
        with bottom_col1:
            st.markdown("**Experiment Summary**")
            st.info(f"""
            **Dataset:** {data_type}
            
            **Size:** {len(X):,} points
            
            **True Clusters:** {len(np.unique(y_true))}
            
            **Features:** {X.shape[1]}D
            """)
        
        with bottom_col2:
            st.markdown("**Performance Comparison**")
            # Get SAMS results (either single or both methods)
            sams_fit_predict = results.get('SAMS (fit_predict)') or results.get('SAMS')
            sams_fit = results.get('SAMS (fit)')
            standard_result = results.get('Standard Mean-Shift')
            sklearn_result = results.get('Scikit-Learn Mean-Shift')

            # Use fit_predict as the primary SAMS method for comparison
            primary_sams = sams_fit_predict if sams_fit_predict else sams_fit

            if primary_sams and (standard_result or sklearn_result):
                # Calculate speedups
                if standard_result:
                    speedup_standard = standard_result['time'] / primary_sams['time'] if primary_sams['time'] > 0 else 0
                    st.info(f"**SAMS vs Standard:**\n{speedup_standard:.1f}x faster")

                if sklearn_result:
                    speedup_sklearn = sklearn_result['time'] / primary_sams['time'] if primary_sams['time'] > 0 else 0
                    st.info(f"**SAMS vs Sklearn:**\n{speedup_sklearn:.1f}x faster")

            # Compare fit_predict vs fit if both are available
            if sams_fit_predict and sams_fit:
                speedup_methods = sams_fit['time'] / sams_fit_predict['time'] if sams_fit_predict['time'] > 0 else 0
                st.info(f"**fit_predict vs fit:**\n{speedup_methods:.1f}x faster")
        
        with bottom_col3:
            st.markdown("**Runtime Chart**")
            fig = create_runtime_plot(results)
            st.pyplot(fig)
            

def create_clustering_plot(X, y_true, results):
    """Create clustering visualization with support for high-dimensional data using PCA"""
    
    n_methods = len(results) + 1  # +1 for true clusters
    
    # Layout: prefer 1x4 if fits, otherwise 2x2
    if n_methods <= 4:
        n_cols = n_methods  # 1x4 or 1x3 or 1x2
        n_rows = 1
    else:
        n_cols = 2  # 2x2 for more than 4 methods
        n_rows = (n_methods + n_cols - 1) // n_cols
    
    is_1d = X.shape[1] == 1
    is_2d = X.shape[1] == 2
    is_3d = X.shape[1] == 3
    is_high_d = X.shape[1] > 3
    
    # For high-dimensional data, use PCA for visualization
    X_plot = X
    pca_info = ""
    if is_high_d:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_[:2].sum()
        pca_info = f" (PCA: {explained_var:.1%} variance explained)"
    
    # Adjust figure size based on layout
    if n_rows == 1:  # 1x4 layout
        fig_width = 4 * n_cols  # Scale width with number of columns
        fig_height = 8 if is_3d else 6
    else:  # 2x2 or other layouts
        fig_width = 12
        fig_height = (8 if is_3d else 6) * n_rows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Plot True clusters
    if is_3d and not is_high_d:
        ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_true, cmap='viridis', s=20, alpha=0.7)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
    else:
        ax = fig.add_subplot(n_rows, n_cols, 1)
        if is_high_d:
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_true, cmap='viridis', s=15, alpha=0.7)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        elif is_1d:
            # 1D visualization: feature value vs cluster ID (with jitter)
            y_jitter = y_true + np.random.normal(0, 0.1, len(y_true))
            scatter = ax.scatter(X[:, 0], y_jitter, c=y_true, cmap='viridis', s=20, alpha=0.8)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("True Cluster ID (with jitter)")
        else:  # is_2d
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10, alpha=0.8)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)
    
    true_clusters = len(np.unique(y_true))
    dim_info = f"{X.shape[1]}D" if is_high_d else ""
    ax.set_title(f"True Clusters {dim_info}(n={len(X):,}, k={true_clusters}){pca_info}")
    
    # Plot method results
    plot_idx = 2
    for method_name, result in results.items():
        if is_3d and not is_high_d:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=result['labels'], cmap='viridis', s=20, alpha=0.7)
            
            # Add cluster centers for 3D
            if result['centers'] is not None and len(result['centers']) > 0 and result['centers'].shape[1] == 3:
                ax.scatter(result['centers'][:, 0], result['centers'][:, 1], result['centers'][:, 2],
                          c='red', marker='x', s=100, linewidths=3, alpha=0.5)
            
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
        else:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            if is_high_d:
                scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=result['labels'], cmap='viridis', s=15, alpha=0.7)
                
                # Project cluster centers to 2D using same PCA
                if result['centers'] is not None and len(result['centers']) > 0:
                    centers_2d = pca.transform(result['centers'])
                    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                              c='red', marker='x', s=120, linewidths=4, alpha=0.5)
                
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
            elif is_1d:
                # 1D visualization: feature value vs cluster ID (with jitter)
                y_jitter = result['labels'] + np.random.normal(0, 0.1, len(result['labels']))
                scatter = ax.scatter(X[:, 0], y_jitter, c=result['labels'], cmap='viridis', s=20, alpha=0.8)
                
                # Add cluster centers if available
                if result['centers'] is not None and len(result['centers']) > 0:
                    ax.scatter(result['centers'][:, 0],
                              range(len(result['centers'])),  # Use cluster indices as y-positions
                              c='red', marker='x', s=120, linewidths=3, alpha=0.5)
                
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Cluster ID (with jitter)")
            else:  # is_2d
                scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'], cmap='viridis', s=10, alpha=0.8)
                
                # Add cluster centers for 2D
                if result['centers'] is not None and len(result['centers']) > 0:
                    ax.scatter(result['centers'][:, 0], result['centers'][:, 1],
                              c='red', marker='x', s=100, linewidths=3, alpha=0.5)
                
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
            ax.grid(True, alpha=0.3)
        
        n_clusters = result['n_clusters']
        runtime = result['time']
        ax.set_title(f"{method_name}\nClusters: {n_clusters}, Time: {runtime:.3f}s")
        
        plot_idx += 1
    
    plt.suptitle("Clustering Results: Data Distribution and Cluster Assignments", fontsize=16)
    plt.tight_layout()
    return fig

def plot_clustering_result_streamlit(X, labels, config_name, n_samples, sample_frac=None, return_fig=True):
    """Modified plot_clustering_result that returns figure for Streamlit with 3D support"""
    n_clusters = len(np.unique(labels))
    
    if sample_frac is not None:
        title = f"SAMS Clustering Result for '{config_name}'\n(n={n_samples:,}, sample={sample_frac*100:.0f}%, Clusters={n_clusters})"
    else:
        title = f"Data Distribution for '{config_name}'\n(n={n_samples:,} points, True Clusters={n_clusters})"
    
    # Check if data is 3D
    if X.shape[1] == 3:
        # 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=20, alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Cluster ID", shrink=0.5)
    elif X.shape[1] == 2:
        # 2D plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
    else:
        # 1D plot - use scatter plot with y-axis as cluster assignment
        fig, ax = plt.subplots(figsize=(8, 6))
        y_jitter = labels + np.random.normal(0, 0.1, len(labels))  # Add slight jitter for visibility
        scatter = ax.scatter(X[:, 0], y_jitter, c=labels, cmap='viridis', s=20, alpha=0.8)
        
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Cluster ID (with jitter)")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
    
    plt.tight_layout()
    return fig

def create_data_distribution_plot(X, y_true, data_type, n_samples):
    """Create data distribution plot using the streamlit version of plot_clustering_result"""
    return plot_clustering_result_streamlit(X, y_true, data_type, n_samples)

def create_individual_clustering_plot(X, labels, method_name, result_info):
    """Create clustering result plot using matplotlib"""
    
    # Get metrics for title
    n_clusters = result_info['n_clusters']
    runtime = result_info['time']
    bandwidth = result_info['bandwidth']
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data points colored by cluster assignment
    if X.shape[1] == 1:
        # 1D plot: feature value vs cluster ID with jitter
        y_jitter = labels + np.random.normal(0, 0.1, len(labels))
        scatter = ax.scatter(X[:, 0], y_jitter, c=labels, cmap='viridis', s=20, alpha=0.8)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Cluster ID (with jitter)")
    else:
        # 2D+ plot
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10, alpha=0.8)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    
    # Add cluster centers as red X markers
    if result_info.get('centers') is not None and len(result_info['centers']) > 0:
        centers = result_info['centers']
        if X.shape[1] == 1:
            # For 1D: plot centers on x-axis at y-positions corresponding to cluster indices
            ax.scatter(centers[:, 0], range(len(centers)), c='red', marker='x', s=120, linewidths=3, alpha=0.5, label='Centers')
        else:
            # For 2D+: plot centers normally
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, linewidths=3, alpha=0.5, label='Centers')
    
    # Set title and labels (already set above based on dimensionality)
    ax.set_title(f"{method_name} Clustering Result\n(n={len(X):,}, Clusters={n_clusters}, Time={runtime:.3f}s, BW={bandwidth:.4f})")
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    
    # Add legend if centers are present
    if result_info.get('centers') is not None and len(result_info['centers']) > 0:
        ax.legend()
    
    plt.tight_layout()
    return fig

def create_runtime_plot(results):
    """Create runtime comparison plot using matplotlib"""
    
    methods = list(results.keys())
    times = [results[method]['time'] for method in methods]
    
    # Color SAMS differently
    colors = ['#ff6b6b' if 'SAMS' in method else '#4ecdc4' for method in methods]
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create bar plot
    bars = ax.bar(methods, times, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom')
    
    # Set labels and title
    ax.set_title("Runtime Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel("Time (seconds)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def calculate_metrics(X, y_true, results):
    """Calculate clustering performance metrics"""

    metrics_data = []

    for method_name, result in results.items():
        labels = result['labels']

        # Calculate cluster sizes (sorted in descending order)
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_counts = sorted(counts, reverse=True)
        cluster_sizes_str = ', '.join([str(count) for count in sorted_counts])

        # Calculate metrics
        try:
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = 0
        except:
            ari = nmi = silhouette = 0

        metrics_data.append({
            'Method': method_name,
            'Clusters': result['n_clusters'],
            'Runtime (s)': f"{result['time']:.3f}",
            'Bandwidth': f"{result['bandwidth']:.4f}",
            'ARI': f"{ari:.3f}",
            'NMI': f"{nmi:.3f}",
            'Silhouette': f"{silhouette:.3f}",
            'Cluster Sizes': cluster_sizes_str
        })

    return pd.DataFrame(metrics_data)

def add_sidebar_info():
    """Add sidebar information"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("About SAMS")
        st.markdown("""
        **Key Algorithm Features:**
        - **O(n) complexity** per iteration
        - **Stochastic sampling** for speed
        - **Adaptive parameters** 
        - **Vectorized computation**
        
        **Validated Performance:**
        - **74-106x speedup** over mean-shift
        - **91-99% quality retention**
        - **Scales to large datasets**
        """)
        
        st.markdown("---")
        st.subheader("Reference")
        st.markdown("""
        Hyrien, O., & Baran, R. H. (2016). 
        *Fast Nonparametric Density-Based 
        Clustering of Large Data Sets Using a 
        Stochastic Approximation Mean-Shift Algorithm.*
        """)
        
        st.markdown("---")
        st.markdown("*Demo built with Streamlit & Plotly*")

if __name__ == "__main__":
    main()
    add_sidebar_info()