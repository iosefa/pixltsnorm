"""
pixltsnorm
==========

A Python library for pixel-based linear (and optionally seasonal) time-series 
normalization, bridging, and harmonization across multiple sensors.

Submodules:
  - harmonize            : Chain bridging logic (two-pass) for arrays.
  - dataframe_harmonize : Tools for DataFrame-based bridging.
  - models               : Contains `fit_linear` and `fit_seasonal` routines.
  - utils                : Utility functions for outlier filtering, time series extraction.
  - earth_engine         : (Subpackage) Earth Engine–specific routines (e.g., NDVI creation).

Exposed API:
  - Harmonizer
  - DataFrameHarmonizer
  - fit_linear
  - fit_seasonal
  - unify_and_extract_timeseries
  - filter_outliers
"""

from .harmonize import Harmonizer
from .dataframe_harmonize import DataFrameHarmonizer
from .models import fit_linear, fit_seasonal
from .utils import unify_and_extract_timeseries, filter_outliers
