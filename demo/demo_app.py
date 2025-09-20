import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import MeanShift
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our SAMS implementation
from sams_clustering import SAMS_Clustering, StandardMeanShift

# Page configuration
st.set_page_config(
    page_title="SAMS Clustering Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔬 SAMS: Stochastic Approximation Mean-Shift Clustering Demo")
    st.markdown("""
    **Interactive simulation studies for:** *"Fast Nonparametric Density-Based Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift Algorithm"* by Hyrien & Baran (2017)
    
    Explore the SAMS algorithm with customizable parameters and compare performance against standard mean-shift clustering.
    """)
    
    # Sidebar for simulation parameters
    with st.sidebar:
        st.header("🎛️ Simulation Parameters")
        
        # Data generation parameters
        st.subheader("📊 Data Generation")
        
        # Dataset type
        dataset_type = st.selectbox(
            "Dataset Type",
            ["Gaussian Blobs", "Concentric Circles", "Two Moons", "Mixed Densities"],
            help="Choose the type of synthetic dataset to generate"
        )
        
        # Sample size
        n_samples = st.slider(
            "Sample Size (n)",
            min_value=500,
            max_value=100000,
            value=1000,
            step=100,
            help="Number of data points to generate"
        )
        
        # Number of clusters (for applicable datasets)
        if dataset_type in ["Gaussian Blobs", "Mixed Densities"]:
            n_centers = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=8,
                value=4,
                help="Number of clusters in the dataset"
            )
        else:
            n_centers = 2  # Fixed for circles and moons
        
        # Noise level
        noise_level = st.slider(
            "Noise Level",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Amount of noise to add to the data"
        )
        
        # Cluster standard deviation (for blobs)
        if dataset_type == "Gaussian Blobs":
            cluster_std = st.slider(
                "Cluster Std Dev",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Standard deviation of Gaussian clusters"
            )
        
        st.subheader("⚙️ Algorithm Parameters")
        
        # SAMS parameters
        st.markdown("**SAMS Configuration:**")
        sample_fraction = st.slider(
            "Sample Fraction (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Percentage of data points to sample at each iteration"
        ) / 100
        
        bandwidth_mode = st.radio(
            "Bandwidth Selection",
            ["Data-driven (automatic)", "Manual"],
            help="Choose how to set the bandwidth parameter"
        )
        
        if bandwidth_mode == "Manual":
            bandwidth = st.slider(
                "Bandwidth",
                min_value=0.05,
                max_value=2.0,
                value=0.5,
                step=0.05,
                help="Bandwidth parameter for kernel density estimation"
            )
        else:
            bandwidth = None
        
        max_iter = st.slider(
            "Max Iterations",
            min_value=50,
            max_value=500,
            value=200,
            step=25,
            help="Maximum number of iterations"
        )
        
        # Comparison options
        st.subheader("📈 Comparison")
        compare_methods = st.multiselect(
            "Methods to Compare",
            ["SAMS", "Standard Mean-Shift", "Scikit-Learn Mean-Shift"],
            default=["SAMS", "Standard Mean-Shift"],
            help="Select which clustering methods to compare"
        )
        
        # Random seed
        random_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="Seed for reproducible results"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Clustering Results")
        
        # Generate data button
        if st.button("🔄 Generate Data & Run Clustering", type="primary"):
            # Set random seed
            np.random.seed(random_seed)
            
            # Generate dataset
            with st.spinner("Generating dataset..."):
                X, y_true = generate_dataset(dataset_type, n_samples, n_centers, noise_level, 
                                           cluster_std if dataset_type == "Gaussian Blobs" else None)
            
            # Store in session state
            st.session_state.X = X
            st.session_state.y_true = y_true
            st.session_state.dataset_info = {
                'type': dataset_type,
                'n_samples': n_samples,
                'n_centers': n_centers,
                'noise': noise_level
            }
            
            # Run clustering methods
            results = {}
            
            for method in compare_methods:
                with st.spinner(f"Running {method}..."):
                    result = run_clustering_method(method, X, bandwidth, sample_fraction, max_iter)
                    results[method] = result
            
            st.session_state.results = results
            st.success("✅ Clustering completed!")
    
    # Display results if available
    if 'X' in st.session_state and 'results' in st.session_state:
        display_results(st.session_state.X, st.session_state.y_true, 
                       st.session_state.results, st.session_state.dataset_info, col1, col2)

def generate_dataset(dataset_type, n_samples, n_centers, noise_level, cluster_std=None):
    """Generate synthetic datasets based on user parameters"""
    
    if dataset_type == "Gaussian Blobs":
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=2,
            cluster_std=cluster_std,
            random_state=42
        )
        # Add noise
        X += np.random.normal(0, noise_level, X.shape)
        
    elif dataset_type == "Concentric Circles":
        X, y_true = make_circles(
            n_samples=n_samples,
            noise=noise_level,
            factor=0.6,
            random_state=42
        )
        
    elif dataset_type == "Two Moons":
        X, y_true = make_moons(
            n_samples=n_samples,
            noise=noise_level,
            random_state=42
        )
        
    elif dataset_type == "Mixed Densities":
        # Create clusters with different densities
        np.random.seed(42)
        
        # High density cluster
        n1 = n_samples // 3
        cluster1 = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], size=n1)
        
        # Medium density cluster  
        n2 = n_samples // 3
        cluster2 = np.random.multivariate_normal([4, 4], [[1.5, 0], [0, 1.5]], size=n2)
        
        # Low density cluster
        n3 = n_samples - n1 - n2
        cluster3 = np.random.multivariate_normal([-2, 3], [[2.5, 0], [0, 2.5]], size=n3)
        
        X = np.vstack([cluster1, cluster2, cluster3])
        y_true = np.hstack([np.zeros(n1), np.ones(n2), np.full(n3, 2)])
        
        # Add noise
        X += np.random.normal(0, noise_level, X.shape)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y_true.astype(int)

def run_clustering_method(method, X, bandwidth, sample_fraction, max_iter):
    """Run specified clustering method and return results"""
    
    start_time = time.time()
    
    if method == "SAMS":
        clusterer = SAMS_Clustering(
            bandwidth=bandwidth,
            sample_fraction=sample_fraction,
            max_iter=max_iter,
            tol=1e-4,
            adaptive_sampling=True,
            early_stop=True
        )
        labels, centers = clusterer.fit_predict(X)
        actual_bandwidth = clusterer.bandwidth
        
    elif method == "Standard Mean-Shift":
        # Use same bandwidth as SAMS if available
        if bandwidth is None:
            # Get bandwidth from a quick SAMS run
            temp_sams = SAMS_Clustering(bandwidth=None, sample_fraction=0.01, max_iter=10)
            temp_sams.fit_predict(X[:100])  # Quick estimation
            actual_bandwidth = temp_sams.bandwidth
        else:
            actual_bandwidth = bandwidth
            
        clusterer = StandardMeanShift(
            bandwidth=actual_bandwidth,
            max_iter=max_iter,
            tol=1e-4
        )
        labels, centers = clusterer.fit_predict(X)
        
    elif method == "Scikit-Learn Mean-Shift":
        if bandwidth is None:
            clusterer = MeanShift()
        else:
            clusterer = MeanShift(bandwidth=bandwidth)
        
        labels = clusterer.fit_predict(X)
        centers = clusterer.cluster_centers_
        actual_bandwidth = clusterer.bandwidth
    
    end_time = time.time()
    
    return {
        'labels': labels,
        'centers': centers,
        'time': end_time - start_time,
        'bandwidth': actual_bandwidth,
        'n_clusters': len(np.unique(labels))
    }

def display_results(X, y_true, results, dataset_info, col1, col2):
    """Display clustering results and comparisons"""
    
    with col1:
        # Create visualization
        fig = create_clustering_plot(X, y_true, results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        metrics_df = calculate_metrics(X, y_true, results)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Runtime comparison
        if len(results) > 1:
            st.subheader("⚡ Runtime Comparison")
            runtime_fig = create_runtime_plot(results)
            st.plotly_chart(runtime_fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Summary")
        
        # Dataset information
        st.info(f"""
        **Dataset:** {dataset_info['type']}
        
        **Size:** {dataset_info['n_samples']:,} points
        
        **True Clusters:** {dataset_info['n_centers']}
        
        **Noise Level:** {dataset_info['noise']:.2f}
        """)
        
        # Method comparison
        st.subheader("🎯 Method Comparison")
        
        for method_name, result in results.items():
            speedup = ""
            if 'SAMS' in results and method_name != 'SAMS':
                sams_time = results['SAMS']['time']
                method_time = result['time']
                if method_time > 0:
                    speedup_factor = method_time / sams_time
                    speedup = f"\n\n*SAMS is {speedup_factor:.1f}x faster*"
            
            st.success(f"""
            **{method_name}**
            
            🔍 **Clusters Found:** {result['n_clusters']}
            
            ⏱️ **Runtime:** {result['time']:.3f}s
            
            🎛️ **Bandwidth:** {result['bandwidth']:.4f}
            {speedup}
            """)
        
        # Download results
        st.subheader("💾 Export Results")
        if st.button("📊 Download Metrics CSV"):
            metrics_df = calculate_metrics(X, y_true, results)
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sams_clustering_results.csv",
                mime="text/csv"
            )

def create_clustering_plot(X, y_true, results):
    """Create interactive clustering visualization"""
    
    n_methods = len(results) + 1  # +1 for true clusters
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    subplot_titles = ["True Clusters"] + list(results.keys())
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # True clusters
    row, col = 1, 1
    fig.add_trace(
        go.Scatter(
            x=X[:, 0], 
            y=X[:, 1],
            mode='markers',
            marker=dict(color=y_true, colorscale='viridis', size=5),
            name="True",
            showlegend=False
        ),
        row=row, col=col
    )
    
    # Method results
    for i, (method_name, result) in enumerate(results.items()):
        row = ((i + 1) // n_cols) + 1
        col = ((i + 1) % n_cols) + 1
        
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1], 
                mode='markers',
                marker=dict(color=result['labels'], colorscale='viridis', size=5),
                name=method_name,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add cluster centers if available
        if result['centers'] is not None and len(result['centers']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=result['centers'][:, 0],
                    y=result['centers'][:, 1],
                    mode='markers',
                    marker=dict(color='red', symbol='x', size=12, line=dict(width=2)),
                    name=f"{method_name} Centers",
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Clustering Results Comparison",
        title_x=0.5
    )
    
    return fig

def create_runtime_plot(results):
    """Create runtime comparison plot"""
    
    methods = list(results.keys())
    times = [results[method]['time'] for method in methods]
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=times,
            text=[f"{t:.3f}s" for t in times],
            textposition='auto',
            marker_color=['#1f77b4' if 'SAMS' in method else '#ff7f0e' for method in methods]
        )
    ])
    
    fig.update_layout(
        title="Runtime Comparison",
        xaxis_title="Method",
        yaxis_title="Time (seconds)",
        height=400
    )
    
    return fig

def calculate_metrics(X, y_true, results):
    """Calculate clustering performance metrics"""
    
    metrics_data = []
    
    for method_name, result in results.items():
        labels = result['labels']
        
        # Calculate metrics
        try:
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
            silhouette = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0
        except:
            ari = nmi = silhouette = 0
        
        metrics_data.append({
            'Method': method_name,
            'Clusters Found': result['n_clusters'],
            'Runtime (s)': f"{result['time']:.3f}",
            'Bandwidth': f"{result['bandwidth']:.4f}",
            'ARI': f"{ari:.3f}",
            'NMI': f"{nmi:.3f}",
            'Silhouette Score': f"{silhouette:.3f}"
        })
    
    return pd.DataFrame(metrics_data)

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **SAMS Algorithm Features:**
    - O(n) complexity per iteration
    - Data-driven bandwidth selection
    - Stochastic approximation for speed
    - Adaptive sampling strategies
    
    **Paper Reference:**
    Hyrien, O., & Baran, R. H. (2017). Fast Nonparametric Density-Based Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift Algorithm.
    """)
    
    st.markdown("---")
    st.markdown("*Built with Streamlit & Plotly*")

if __name__ == "__main__":
    main()