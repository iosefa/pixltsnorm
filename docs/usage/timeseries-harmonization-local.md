# Local Time Series Harmonization

In **Local Time-Series Harmonization**, each pixel (or row) is assigned a **unique slope/intercept** to reconcile differences across sensors, rather than using a single, global correction. This allows for **spatially varying** calibrations, which can be crucial when factors such as topography, land cover, or aerosol fields vary strongly across the study region.

---

## 1. Why Local Harmonization?

1. **Spatial Variability**  
   - Some regions (e.g., mountainous terrain, heterogeneous soils/vegetation) exhibit sensor biases that differ from one pixel to the next.  
   - One global calibration slope/intercept might be insufficient if reflectance differences vary spatially.

2. **Per-Pixel Correctness**  
   - By deriving a slope/intercept for each pixel, you can correct local divergences that a “one-size-fits-all” approach would miss.  
   - Particularly beneficial for high-resolution imagery or studies analyzing diverse landscapes.

3. **Finer-Grained Analysis**  
   - Enables detection of subtle local sensor differences (e.g., adjacency to water or complex canopy geometry).

---

## 2. Basic Approach

1. **Flatten Overlapping Columns**:  
   - For each pair of DataFrames, gather a set of **row-wise** time-series data from both sensors.  
2. **Row-by-Row Bridging**:  
   - Fit a **linear** or **seasonal** model on the data in each row, obtaining slope and intercept for that row alone.  
3. **Chaining**:  
   - For multiple sensors, a left pass (target sensor → left neighbors) and a right pass (target sensor → right neighbors) ensures all are eventually mapped to the chosen reference scale.

---

## 3. How to Use in `pixltsnorm`

Pixltsnorm provides the `DataFrameHarmonizer` class with `approach='local'`. Each row (pixel) in your DataFrames gets its own slope/intercept. For example:

```python
from pixltsnorm.dataframe_harmonize import DataFrameHarmonizer

# Suppose df_l5, df_l7, df_l8 each contain NDVI or reflectance columns over time (row= pixel)
dfs = [df_l5, df_l7, df_l8]

harm_df = DataFrameHarmonizer(
    method='linear',        # or 'seasonal_decompose' if exactly 2 sensors
    approach='local',       # local bridging
    outlier_threshold=0.2
)

harm_df.fit(dfs, target_index=2)  # pick df_l8 as the reference
print(harm_df.transforms_[0])     # => e.g., { 'slope': array([...]), 'inter': array([...]) }

# Convert them all to the df_l8 scale row-by-row:
harmonized_list = harm_df.get_harmonized_dfs(dfs)
df_l5_h, df_l7_h, df_l8_h = harmonized_list
```

The result: **df_l5_h** and **df_l7_h** are now consistent with **df_l8_h** on a **row-by-row** basis.

---

## 4. Example Workflow

1. **Data Preparation**  
   - Each DataFrame has date/time-based columns plus an index representing pixels/rows.  
   - Filter out invalid data (clouds, saturations).  
2. **Set Approach = 'local'**  
   - Tells `DataFrameHarmonizer` to do a row-by-row bridging.  
3. **Check Fits**  
   - Each row obtains a slope array and an intercept array stored in `harm_df.transforms_`.  
   - If any rows have insufficient overlap or become entirely outliers, bridging may fail for those specific rows (leading to NaN slope/intercept).  
4. **Apply**  
   - Transform each row’s data to the target sensor scale with `harm_df.transform_df(...)` or `harm_df.get_harmonized_dfs(...)`.

---

## 5. When to Consider Local Mode

- **Heterogeneous Landscapes**  
  Regions with strong variation in land cover, topography, or atmospheric conditions – requiring a distinct calibration for each pixel.  
- **Per-Pixel Accuracy**  
  If your analysis depends on sub-regional or local studies (e.g., single fields, forest stands, water bodies).

### Trade-offs

- **Computational Overhead**  
  - Fitting a slope/intercept for each pixel can be expensive for large images.  
- **Data Requirements**  
  - Each row must have sufficient overlap in time. If many rows are clouded or missing data, local bridging might yield partial coverage.  
- **Statistical Robustness**  
  - Row-level fits can be noisy if only a few valid overlapping observations exist for a pixel.

---

## 6. Seasonal Option

If `method='seasonal_decompose'`, **two** sensors can be decomposed row-by-row:
1. **Extract** the time series for each row’s columns.  
2. **Apply** a seasonal decomposition (e.g., monthly) if enough time points exist.  
3. **Fit** a linear transform on the deseasonalized signals.  
4. **Re-add** the target’s seasonal signature.

This is more computationally intense and requires a consistent, large time series per row.

---

## 7. Further Reading

1. **Local vs. Global Sensor Differences**  
   Roy, D. P. et al., *Characterization of Landsat-7 to Landsat-8 Reflective Wavelength and NDVI Continuity*, *Remote Sensing of Environment* 185, 57–70 (2016).
2. **Robust Row-wise NDVI**  
   Gao, F. et al., *Bidirectional NDVI and atmospherically resistant BRDF inversion for vegetation canopy*, *IEEE Transactions on Geoscience and Remote Sensing* 40, 1269–1278 (2002).

---

## 8. Summary

Local harmonization in pixltsnorm offers **pixel-level** bridging, modeling each row’s differences with a unique slope/intercept. This is especially valuable in highly variable landscapes, or whenever a single global correction is insufficient. While more computationally demanding, it provides **maximal accuracy** for each row/pixel. For broad, relatively uniform domains, see [usage/timeseries-harmonization-global.md](./timeseries-harmonization-global.md).
