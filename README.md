# pixltsnorm

**Pixel-based Linear Time Series Normalizer**

`pixltsnorm` is a small, focused Python library that:

- **Bridges** or **harmonizes** numeric time-series data (e.g., reflectance, NDVI, etc.) across multiple sensors or sources.  
- **Fits** simple linear transformations (y = slope*x + intercept) to map one sensor’s scale onto another.  
- **Chains** transformations to handle indirect overlaps (sensor0 → sensor1 → …).  
- **Filters** outliers using threshold-based filtering before fitting linear models.

Although originally inspired by NDVI normalization across different Landsat sensors, **pixltsnorm** is domain-agnostic. You can use it for any numeric time-series that needs linear alignment.

---

## Features

1. **Outlier Filtering**  
   - Removes large disagreements in overlapping time-series pairs, based on a simple threshold for |A - B|.

2. **Linear Bridging**  
   - Regress one sensor’s measurements onto another’s (y = slope*x + intercept).  
   - Produces an easy-to-apply transform function.

3. **Chaining**  
   - Allows any number of sensors to be combined in sequence, producing a single transform from the first sensor to the last.

4. **Lightweight**  
   - Minimal dependencies: `numpy`, `scikit-learn`, and optionally `pandas`.

---

## Basic Usage

### Harmonize Two Sensors

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

# Suppose we have 4 different sensors that partially overlap:
# sensor0 -> sensor1 -> sensor2 -> sensor3
sensor0 = np.random.rand(10)
sensor1 = np.random.rand(10)
sensor2 = np.random.rand(10)
sensor3 = np.random.rand(10)

chain_result = chain_harmonization([sensor0, sensor1, sensor2, sensor3])
print("Pairwise transforms:", chain_result['pairwise'])
print("Overall slope (sensor0->sensor3):", chain_result['final_slope'])
print("Overall intercept (sensor0->sensor3):", chain_result['final_intercept'])

# Apply sensor0 -> sensor3 transform
sensor0_on_sensor3_scale = chain_result['final_slope'] * sensor0 + chain_result['final_intercept']
print("sensor0 mapped onto sensor3 scale:", sensor0_on_sensor3_scale)
```

## Installation

1. Clone or download this repository.  
2. (Optional) Create and activate a virtual environment.  
3. Install in editable mode:

```bash
pip install -e .
```

Then you can `import pixltsnorm` (or whichever name you gave the folder) in your scripts.

---

## Acknowledgements

- **Joseph Emile Honour Percival** performed the initial research during his PhD at **Kyoto University**, where the pixel-level time-series normalization idea was first applied to multi-sensor NDVI analysis.  
- The linear bridging approach for different Landsat sensors is based on work published by Roy et al. (2016), which outlines regression-based continuity between Landsat 7 and 8.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.