"""
pixltsnorm
==========

Lightweight library for bridging and harmonizing multi-sensor time-series data,
particularly NDVI or similar measurements from multiple sensors.

Features:
- Single-pixel bridging (harmonize_series, chain_harmonization)
- Global bridging across entire dataset (global_bridging, chain_global_bridging)
- Outlier filtering and simple linear regression wrappers
"""

__version__ = "0.1.0"

from .harmonize import harmonize_series, chain_harmonization
from .outlier_filter import filter_outliers
from .regression import fit_regression, apply_regression
from .global_harmonize import global_bridging, chain_global_bridging