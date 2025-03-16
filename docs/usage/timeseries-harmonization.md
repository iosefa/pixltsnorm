# usage/timeseries-harmonization.md

This guide provides an overview of **time-series harmonization** in **pixltsnorm**, detailing how multiple sensors can be aligned onto a single reference scale. Whether your objective is to unify Landsat 5, 7, and 8 NDVI datasets, or to calibrate other remote-sensing measurements, pixltsnorm offers linear and seasonal-based transformations to standardize time-series data.

---

## 1. Motivation

Different sensors often exhibit **systematic scale differences** due to instrument calibration, spectral response variations, or distinct atmospheric correction methods. **Harmonizing** these sensors is crucial for consistent comparisons or combined analyses. Time-series harmonization:

- **Removes** or **reduces** inter-sensor offsets and biases.
- **Ensures** a shared “reference” scale (the “target” sensor).
- **Facilitates** merged time-series or multi-sensor synergy.

---

## 2. Pairwise Bridging and Chaining

Pixltsnorm harmonizes sensors through **pairwise** transformations:

1. **Outlier Removal**: A threshold-based approach excludes data points where \(|A - B|\) exceeds a specified limit.  
2. **Linear or Seasonal**: Fit a **linear** slope/intercept or apply **seasonal decomposition** (for two sensors) before regression.  
3. **Two-Pass Chaining**: For multiple sensors, pixltsnorm composes these pairwise fits so that each sensor references the **target** sensor.

---

## 3. Approaches

1. **Global**:  
   - Flattens all overlapping data points into one dataset.  
   - Derives a single slope/intercept for each sensor → target.  
   - Particularly suitable if biases are consistent across space/time.  
   - See [Global Harmonization](./timeseries-harmonization-global.md) for details.

2. **Local**:  
   - Assigns a distinct slope/intercept **per row** (e.g., a “pixel” dimension in DataFrames).  
   - Accommodates spatially varying biases.  
   - For more information, see [Local Harmonization](./timeseries-harmonization-local.md).

---

## 4. Basic Usage Example

### 4.1 Simple Linear Chaining (Arrays)

```python
import numpy as np
from pixltsnorm.harmonize import Harmonizer

# Example sensor arrays
s0 = np.array([1.2, 1.4, 1.5, 2.0, ...])
s1 = np.array([2.1, 2.3, 2.6, 2.8, ...])
s2 = np.array([4.0, 4.3, 4.2, 4.5, ...])  # let's designate s2 as the target

harm = Harmonizer(method='linear', outlier_threshold=0.2)
harm.fit([s0, s1, s2], target_index=2)

for i, (slope, intercept) in enumerate(harm.transforms_):
    print(f"Sensor {i} -> target: slope={slope:.3f}, intercept={intercept:.3f}")
```

In this scenario, s0 → s1 → s2 bridging yields final slope/intercept for each sensor → s2.

### 4.2 Seasonal (Two-Sensor Only)

If your data show periodic patterns (e.g., strong seasonality in NDVI), you can specify `method='seasonal_decompose'`. This removes each sensor’s seasonal component and fits a linear model on the residuals, then reintroduces the target’s seasonal pattern.

```python
harm_seas = Harmonizer(method='seasonal_decompose', period=12, outlier_threshold=0.1)
harm_seas.fit([s0, s1])  # only valid for exactly 2 sensors

# transform new data from sensor0 at time=3 (index in the time array, for instance)
harmonized_value = harm_seas.transform(sensor_index=0, x=1.7, t=3)
```

---

## 5. Additional Reading and References

- **Linear Regression**:  
  Freedman, D., Pisani, R., & Purves, R. (2007). *Statistics*.  
  Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis*.

- **Time-Series & Seasonal**:  
  Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*.

- **Multi-Sensor NDVI**:  
  Roy, D. P., Kovalskyy, V., Zhang, H., Vermote, E., Yan, L., Kumar, S. S., & Egorov, A. (2016). Characterization of Landsat-7 to Landsat-8 NDVI continuity. *Remote Sensing of Environment, 185*, 57–70.

---

## 6. Next Steps

- **Global Harmonization**: Learn how to flatten your data and derive a single scale-shift in [Global Timeseries Harmonization](./timeseries-harmonization-global.md).  
- **Local Harmonization**: Explore row-by-row bridging for spatial variability in [Local Timeseries Harmonization](./timeseries-harmonization-local.md).  
- **Earth Engine**: For Landsat NDVI extraction and advanced remote-sensing workflows, see [usage/landsat-timeseries-creation.md](./landsat-timeseries-creation.md).

By following these principles, pixltsnorm helps unify multi-sensor time-series in a straightforward, flexible manner, enabling robust cross-sensor comparisons and integrated analyses.
