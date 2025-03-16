# pixltsnorm

**Pixel-based Linear Time Series Normalizer**

![Normalized Landsat NDVI Time Series](./docs/images/example.png)

`pixltsnorm` is a small, focused Python library that:

- **Bridges** or **harmonizes** numeric time-series data (e.g., reflectance, NDVI, etc.) across multiple sensors or sources.  
- **Fits** simple linear transformations (`y = slope*x + intercept`) to map one sensor’s scale onto another.  
- **Chains** transformations to handle indirect overlaps (sensor0 → sensor1 → …).  
- **Filters** outliers using threshold-based filtering before fitting linear models.

Although originally inspired by NDVI normalization across different Landsat sensors, **pixltsnorm** is domain-agnostic. You can use it for any numeric time-series that needs linear alignment.

---

## Features

1. **Outlier Filtering**  
   - Removes large disagreements in overlapping time-series pairs, based on a simple threshold for |A - B|.

2. **Local (Pixel-Level) Linear Bridging**  
   - Regress one sensor’s measurements onto another’s (e.g., a single pixel’s time series).  
   - Produces an easy-to-apply transform function for new data.

3. **Global Bridging**  
   - Follows the approach of Roy et al. (2016): gather all overlapping values across the entire dataset, fit one “universal” slope/intercept.  
   - Useful if you need *scene-wide* or *region-wide* continuity between two or more sensors (e.g., L5 → L7 → L8).

4. **Chaining**  
   - Allows any number of sensors to be combined in sequence, producing a single transform from the first sensor to the last.

5. **Lightweight**  
   - Minimal dependencies: `numpy`, `scikit-learn`, and optionally `pandas`.

6. **Earth Engine Submodule**  
   - A dedicated `earth_engine` subpackage provides GEE-specific helpers (e.g., for Landsat) that you can incorporate in your Earth Engine workflows.

---

## Basic Usage

### Harmonize Two Sensors (Pixel-Level Example)

```python
import numpy as np
from pixltsnorm.harmonize import harmonize_series

# Suppose sensorA_values and sensorB_values have overlapping data
sensorA = np.array([0.0, 0.2, 0.8, 0.9])
sensorB = np.array([0.1, 0.25, 0.7, 1.0])

results = harmonize_series(sensorA, sensorB, outlier_threshold=0.2)
print("Slope:", results['coef'])
print("Intercept:", results['intercept'])

# Transform new data from sensorA scale -> sensorB scale
transform_func = results['transform']
new_data = np.array([0.3, 0.4, 0.5])
mapped = transform_func(new_data)
print("Mapped values:", mapped)
```

### Chaining Multiple Sensors

```python
from pixltsnorm.harmonize import chain_harmonization
import numpy as np

# Suppose we have 4 different sensors that partially overlap:
sensor0 = np.random.rand(10)
sensor1 = np.random.rand(10)
sensor2 = np.random.rand(10)
sensor3 = np.random.rand(10)

chain_result = chain_harmonization([sensor0, sensor1, sensor2, sensor3])
print("Pairwise transforms:", chain_result['pairwise'])
print("Overall slope (sensor0->sensor3):", chain_result['final_slope'])
print("Overall intercept (sensor0->sensor3):", chain_result['final_intercept'])

# Apply sensor0 -> sensor3 transform
sensor0_on_sensor3_scale = (chain_result['final_slope'] * sensor0 
                            + chain_result['final_intercept'])
print("sensor0 mapped onto sensor3 scale:", sensor0_on_sensor3_scale)
```

### Global Bridging

```python
import pandas as pd
from pixltsnorm.global_harmonize import chain_global_bridging

# Suppose we have three DataFrames: df_l5, df_l7, df_l8
# Each has row=pixels, columns=dates (plus 'lon','lat').
# The approach merges all overlapping values across the region/time:
result = chain_global_bridging(df_l5, df_l7, df_l8, outlier_thresholds=(0.2, 0.2))

# We get a single slope/intercept for L5->L7, L7->L8, plus the chain L5->L8
print("Global bridging L5->L7 =>", result["L5->L7"]["coef"], result["L5->L7"]["intercept"])
print("Global bridging L7->L8 =>", result["L7->L8"]["coef"], result["L7->L8"]["intercept"])
print("Chained L5->L8 =>", result["L5->L8"]["coef"], result["L5->L8"]["intercept"])
```

### Earth Engine Submodule

```python
from pixltsnorm.earth_engine import create_reduce_region_function, addNDVI, cloudMaskL457

# Use these GEE-based helpers inside your Earth Engine scripts
```

Please see the docs and example notebooks for more examples.

---

## Installation

1. Clone or download this repository.  
2. (Optional) Create and activate a virtual environment.  
3. Install in editable mode:

```bash
pip install -e .
```

Then you can do:

```python
import pixltsnorm
```

and access the library’s functionality.

---

## Acknowledgements

- **Joseph Emile Honour Percival** performed the initial research in 2021 during his PhD at **Kyoto University**, where the pixel-level time-series normalization idea was first applied to multi-sensor analysis.  
- The global bridging logic is inspired by Roy et al. (2016), which outlines regression-based continuity for Landsat sensors across large areas.

Roy, David P., V. Kovalskyy, H. K. Zhang, Eric F. Vermote, L. Yan, S. S. Kumar, and A. Egorov. "Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity." Remote sensing of Environment 185 (2016): 57-70.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.