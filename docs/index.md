# pixltsnorm Documentation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/iosefa/PyForestScan/HEAD?labpath=docs%2Fexamples%2Fgetting-started-importing-preprocessing-dtm-chm.ipynb)
[![PyPI](https://img.shields.io/pypi/v/PyForestScan.svg)](https://pypi.org/project/PyForestScan/)
[![Docker Pulls](https://img.shields.io/docker/pulls/iosefa/pyforestscan?logo=docker&label=pulls)](https://hub.docker.com/r/iosefa/pyforestscan)
[![Contributors](https://img.shields.io/github/contributors/iosefa/PyForestScan.svg?label=contributors)](https://github.com/iosefa/PyForestScan/graphs/contributors)
[![Tests](https://img.shields.io/github/actions/workflow/status/iosefa/PyForestScan/main.yml?branch=main)](https://github.com/iosefa/PyForestScan/actions/workflows/main.yml)
[![Coverage](https://img.shields.io/codecov/c/github/iosefa/PyForestScan/main)](https://codecov.io/gh/iosefa/PyForestScan)

**Pixel-based Linear Time Series Normalizer**

## Overview

**pixltsnorm** is a lightweight Python library for bridging and **harmonizing** multi-sensor time-series data (e.g., NDVI, reflectance). It supports:

- **Linear transformations** (y = slope*x + intercept) to match one sensor's scale onto another.
- **Chain bridging** across multiple sensors (sensor0 → sensor1 → sensor2 → …).
- **Optional Earth Engine** submodules for GEE-based reflectance processing.

Use pixltsnorm to standardize multi-sensor time-series from, say, Landsat 5, 7, and 8, or to unify data from any other sensor or domain that needs a simple linear mapping.

## Features

1. **Outlier Filtering**  
   - Threshold-based filtering (|A - B| > threshold) before performing regression.

2. **Local Bridging**  
   - Fit transformations on a **per-pixel** or **per-subset** basis.

3. **Global Bridging**  
   - Perform a dataset-wide slope/intercept regression, combining all overlapping pixel/time pairs.

4. **Chaining**  
   - Link pairs of sensors (e.g., L5 → L7, L7 → L8) and produce a final (e.g. L5 → L8) transform.

5. **Extensible**  
   - Minimal code can be adapted to other scenarios: any numeric data that needs linear alignment.

## Example Notebooks

We provide sample Jupyter notebooks to help you get started:

- **[Quick Start: Global NDVI Harmonization](examples/global-ndvi-harmonization.ipynb)**
- **[Local (per-pixel) NDVI Harmonization](examples/per-pixel-ndvi-harmonization.ipynb)**
- **[Creating NDVI Timeseries with Earth Engine](examples/create-ndvi-timeseries-with-earthengine.ipynb)**

To run these notebooks:

```bash
pip install jupyter
```

## References

This library is inspired by techniques described in:
	Roy, D. P., Kovalskyy, V., Zhang, H. K., Vermote, E., Yan, L., Kumar, S. S., & Egorov, A. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity. Remote Sensing of Environment, 185, 57–70.
