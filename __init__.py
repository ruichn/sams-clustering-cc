"""
SAMS Clustering - Fast Nonparametric Density-Based Clustering

Implementation of the Stochastic Approximation Mean-Shift (SAMS) algorithm
described in Hyrien & Baran (2017).

Classes:
    SAMS_Clustering: Main SAMS clustering algorithm
    StandardMeanShift: Reference mean-shift implementation for comparison

Functions:
    generate_test_data: Generate synthetic datasets for testing
"""

from .sams_clustering import SAMS_Clustering, StandardMeanShift, generate_test_data

__version__ = "1.0.0"
__author__ = "Implementation of Hyrien & Baran (2017)"
__email__ = "your.email@example.com"

__all__ = [
    "SAMS_Clustering",
    "StandardMeanShift", 
    "generate_test_data"
]