"""
sklearn-sams: Scikit-learn compatible SAMS clustering algorithm

A standalone implementation of the Stochastic Approximation Mean-Shift (SAMS) 
clustering algorithm that follows scikit-learn API conventions.

Based on:
Hyrien, O., & Baran, R. H. (2016). Fast Nonparametric Density-Based 
Clustering of Large Data Sets Using a Stochastic Approximation Mean-Shift 
Algorithm. PMC5417725.
"""

from ._sams import SAMSClustering
from ._version import __version__

__all__ = ['SAMSClustering']