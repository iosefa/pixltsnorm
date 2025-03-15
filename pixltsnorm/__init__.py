"""
time_series_normalizer
======================

Lightweight library for bridging and harmonizing multi-sensor time-series data,
particularly NDVI from Landsat sensors.
"""

__version__ = '0.1.0'

from .harmonize import harmonize_series, chain_harmonization
from .outlier_filter import filter_outliers
from .regression import fit_regression, apply_regression
