# Timeseries Creation with Landsat

In this guide, we demonstrate how to use the **`earth_engine`** subpackage in **pixltsnorm** to create and export **Landsat-based NDVI** (and optionally NBR) time series. Supported satellite missions available for routines in `pixltsnorm.earth_engine` are listed in [Supported Satellite Missions](usage/earth-engine/supported-missions.md).

---

## Overview

The **`earth_engine`** subpackage provides helper functions for:

- **Creating Earth Engine reduceRegion tasks** that query pixel-level reflectance values over time.  
- **Masking clouds** from Landsat data based on band-specific QA bits.  
- **Adding NDVI (Normalized Difference Vegetation Index)** and **NBR (Normalized Burn Ratio)** bands to a Landsat image collection.  

With these tools, you can easily extract a multi-month (or multi-year) **reflectance time series** for your region of interest, then feed the resulting data into the **pixltsnorm** bridging or harmonization pipeline.

---

## Prerequisites

1. **Earth Engine Account**  
   You need a Google Earth Engine account set up.  
2. **earth_engine** Subpackage  
   Make sure you have installed `pixltsnorm` and can import the `earth_engine` subpackage.  
3. **Authentication**  
   You must authenticate your Earth Engine environment (via `earthengine authenticate` or within a Python notebook environment using `ee.Authenticate()`).

---

## Typical Workflow

1. **Initialize Earth Engine**  
   ```python
   import ee

   ee.Authenticate()
   ee.Initialize()
   ```

2. **Define ROI & Time Range**  
   ```python
   roi = ee.Geometry.Rectangle([ -120.0, 35.0, -119.8, 35.2 ])
   start_date = '2020-01-01'
   end_date   = '2020-12-31'
   ```

3. **Get Landsat Collection & Preprocess**  
   - For example, Landsat 5, 7, or 8 collections can be filtered and masked.  
   - The earth_engine subpackage provides functions like:
     - `cloudMaskL457` or `cloudMaskL8` (depending on your sensor).
     - `addNDVI`, `addNBR` to add NDVI/NBR bands to each image.  

   ```python
   from pixltsnorm.earth_engine.landsat import (cloudMaskL457, addNDVI, addNBR)

   # Filter Landsat 7 (ETM+) for our date range & region
   l7_collection = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
       .filterBounds(roi)
       .filterDate(start_date, end_date)
       .map(cloudMaskL457)   # remove clouds
       .map(addNDVI)         # add NDVI band
       .map(addNBR)          # add NBR band
   )
   ```

4. **Extract Time Series**  
   - A typical approach is using a `reduceRegion` or `reduceRegions` operation to extract the NDVI or NBR for each date.  
   - The subpackage may provide helper functions (like `create_reduce_region_function`) to standardize how you reduce images to a single pixel or region.  

   ```python
   from pixltsnorm.earth_engine.landsat import create_reduce_region_function

   # Suppose we want to reduce each image over 'roi' using the 'NDVI' band
   reduce_func = create_reduce_region_function(
       geometry=roi,
       band_names=['NDVI','NBR'],
       scale=30,
       statistics_type='mean'
   )

   # This will map a reduceRegion call over each image
   l7_reduced = l7_collection.map(reduce_func)
   ```

5. **Export or Convert to Pandas**  
   - Once you have a collection of reduced values, you can either:
     - Export it to your Google Drive or GEE Assets as a table.  
     - Retrieve it in Python to create a `pandas.DataFrame`.  
   - The exact flow depends on how you prefer to store or analyze your data.  

   ```python
   # Example to extract the data in Python (small ROI/time range)
   def get_ndvi_values(img):
       d = img.get('system:time_start')
       val = img.get('NDVI_mean')  # from our reduceRegion
       return [ee.Date(d).format('YYYY-MM-dd'), val]

   # convert to a list
   ndvi_list = l7_reduced.map(get_ndvi_values).getInfo()
   ```

---

## Key Functions

1. **`cloudMaskL457(image: ee.Image) -> ee.Image`**  
   Masks out clouds/shadows from Landsat 4/5/7 using QA bits.

2. **`cloudMaskL8(image: ee.Image) -> ee.Image`**  
   Similar cloud/shadow masking logic for Landsat 8.

3. **`addNDVI(image: ee.Image) -> ee.Image`**  
   Adds a band called `"NDVI"` to the image, computed from the red & NIR bands.

4. **`addNBR(image: ee.Image) -> ee.Image`**  
   Adds a band called `"NBR"`, often used to assess burn severity (NIR vs. SWIR2).

5. **`create_reduce_region_function(...)`**  
   Returns a function that can be mapped over an `ee.ImageCollection` to reduce each image to stats over a given geometry. Useful for building consistent time series from many images.

---

## Additional Notes

- **Band Scaling**  
  Depending on the Landsat Collection you use (e.g., Collection 2 Level-2), you might need to account for scaling factors on reflectance or temperature bands.  
- **Combining with `pixltsnorm`**  
  Once you’ve created your NDVI (or reflectance) time series for each sensor, you can apply the normal bridging/harmonizer logic to unify them onto a single scale.

---

## Conclusion

The **`pixltsnorm.earth_engine`** subpackage is designed to simplify time-series creation with Google Earth Engine - turning an image collection into a time series. It provides functions to perform cloud masking, NDVI/NBR band creation, and easy reduceRegion scaffolding. After retrieving your final NDVI data, feed it into **pixltsnorm** bridging methods (e.g., `Harmonizer`, `DataFrameHarmonizer`) to align multiple Landsat missions or unify Landsat with another sensor.

Feel free to explore the example notebook [**Creating NDVI Timeseries with Earth Engine**](../examples/create-ndvi-timeseries-with-earthengine.ipynb) for a **hands-on** tutorial covering these steps in code.
