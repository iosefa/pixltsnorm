"""
earth_engine
============

Subpackage for Earth Engine–specific code, such as utility functions
for Landsat data or GEE-based routines.

Imports:
  - create_reduce_region_function
  - addNDVI
  - addNBR
  - cloudMaskL457
  - scale_factors
"""

from .landsat import (
    create_reduce_region_function,
    addNDVI,
    addNBR,
    cloudMaskL457,
    scale_factors
)
