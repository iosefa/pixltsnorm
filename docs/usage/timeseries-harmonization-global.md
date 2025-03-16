# Global Time Series Harmonization 

This document explains **Global Time-Series Harmonization** within **pixltsnorm**, wherein a **single** slope/intercept transform is derived for each pair of sensors (e.g., sensor A → sensor B). Global bridging means all overlapping data points across space and time are merged before fitting a regression, yielding a universal calibration for the entire dataset.

---

## 1. Rationale

1. **Single Bias Correction**  
   - If differences between sensors are **relatively uniform** across a large region, one global slope/intercept can sufficiently reconcile them.
2. **Computational Simplicity**  
   - Global bridging reduces the problem to a straightforward linear (or seasonal) fit. 
   - Minimizes storage, as you keep just one transform for each sensor → target.

### Typical Applications

- **Mosaics or Composites**  
  Combining many Landsat scenes from multiple sensors – one "global" correction ensures consistent reflectance or NDVI across the mosaic.
- **Large-Area Studies**  
  Regions where sensor differences are assumed not to vary with location or environment (e.g., uniform topography, low spatial complexity).

---

## 2. Data Requirements

1. **Overlapping Observations**:  
   - For each pair of sensors, you need areas / times where both sensors measure the same location under comparable conditions.
2. **Filtering**:  
   - Outlier thresholding, cloud masking, and optional snow or terrain screening.  
   - Possibly discard extremely high reflectance or saturated pixels.
3. **Optional Seasonal Adjustments**:  
   - For strictly linear bridging, no time index or seasonality needed.  
   - If seasonal decomposition is included (two-sensor scenario), read [usage/timeseries-harmonization.md](./timeseries-harmonization.md) for further details.

---

## 3. Basic Workflow

```python
from pixltsnorm.harmonize import Harmonizer

# Suppose we have two sensors covering a large region:
sensorA_data = ... # e.g. reflectance or NDVI from Landsat 5/7
sensorB_data = ... # e.g. reflectance or NDVI from Landsat 8

harm = Harmonizer(method='linear', outlier_threshold=0.2)
harm.fit([sensorA_data, sensorB_data], target_index=1)

# The final transforms are:
transform_A, transform_B = harm.transforms_
print(transform_A, transform_B)
```
- **transform_A**: slope/intercept mapping sensor A → sensor B’s scale  
- **transform_B**: identity = (1.0, 0.0) if sensor B is the target.

In global mode, each sensor is reduced to **one** slope and intercept across the entire region. No row-by-row variation occurs.

---

## 4. Example: NDVI Harmonization for a Large Region

1. **Combine** all times / scenes where sensor A and sensor B overlap.  
2. **Remove** clouds, outliers, and non-overlapping days.  
3. **Compute** NDVI from red & NIR reflectance.  
4. **Flatten** these NDVI pairs into a single 1D array – all points from your region of interest.  
5. **Fit** the bridging:
   ```python
   # sensorA_ndvi, sensorB_ndvi: arrays of 1D NDVI after merging
   harm_ndvi = Harmonizer(method='linear', outlier_threshold=0.1)
   harm_ndvi.fit([sensorA_ndvi, sensorB_ndvi])
   # Done! One slope/intercept for sensor A -> B.
   ```

---

## 5. Considerations and Caveats

1. **Spatial Variability**  
   - If different areas have distinct biases (e.g., terrain, aerosol fields, land cover differences), a global slope/intercept may be inadequate.  
   - In that case, see [Local Time-Series Harmonization](./timeseries-harmonization-local.md).

2. **High Biomass Saturation**  
   - For NDVI or red/NIR reflectance in dense vegetation, both sensors might saturate differently. This can complicate a global linear approach.

3. **Cloud and Snow**  
   - Omissions or commission errors in cloud/snow masks can introduce biases. The threshold-based outlier filter helps mitigate these issues, but the user should carefully review the data screening.

---

## 6. Further Reading

- **Spectral Differences and Linear Transformations**  
  Roy, D. P. et al., *Characterization of Landsat-7 to Landsat-8 Reflective Wavelength and NDVI Continuity*, *Remote Sensing of Environment*, 185, 57–70 (2016).  
- **Landsat Radiometric Calibration**
  Markham, B. L., and Helder, D. L. *Forty-year calibrated record of earth-reflected radiance from Landsat: A review*, *Remote Sensing of Environment*, 122, 30-40 (2012).
- **Seasonal Decomposition**  
  Box, G. E. P., et al., *Time Series Analysis: Forecasting and Control*, 5th ed. (2015).

---

## 7. Summary

Global time-series harmonization assigns **one** slope/intercept transform for each sensor pair. This is ideal when the study region is homogeneous enough (or the user is comfortable) that a single bias correction applies. Users seeking finer spatial detail or pixel-specific calibrations should explore the “local” approach described in [usage/timeseries-harmonization-local.md](./timeseries-harmonization-local.md).  
