"""
landsat.py
==========

Helper functions for working with Landsat imagery in Google Earth Engine (ee).
Includes:
- create_reduce_region_function(...) for per-pixel statistical reduction
- addNDVI(...) and addNBR(...) band-creation functions
- cloudMaskL457(...) for cloud masking in Landsat 4-7, plus L8
- scale_factors(...) for optical band scaling

Example usage:
    import ee
    from pixltsnorm.earth_engine.landsat import (
        create_reduce_region_function,
        addNDVI,
        addNBR,
        cloudMaskL457,
        scale_factors
    )

    # Then call these functions when building ee.ImageCollections
"""

import ee


def create_reduce_region_function(geometry,
                                  reducer=ee.Reducer.mean(),
                                  scale=30,
                                  crs='EPSG:4326',
                                  bestEffort=True,
                                  maxPixels=1e13,
                                  tileScale=4):
    """Creates a region reduction function.

    Creates a region reduction function intended to be used as the input function
    to ee.ImageCollection.map() for reducing pixels intersecting a provided region
    to a statistic for each image in a collection. See ee.Image.reduceRegion()
    documentation for more details.

    Args:
      geometry:
        An ee.Geometry that defines the region over which to reduce data.
      reducer:
        Optional; An ee.Reducer that defines the reduction method.
      scale:
        Optional; A number that defines the nominal scale in meters of the
        projection to work in.
      crs:
        Optional; An ee.Projection or EPSG string ('EPSG:5070') that defines
        the projection to work in.
      bestEffort:
        Optional; A Boolean indicator for whether to use a larger scale if the
        geometry contains too many pixels at the given scale for the operation
        to succeed.
      maxPixels:
        Optional; A number specifying the maximum number of pixels to reduce.
      tileScale:
        Optional; A number representing the scaling factor used to reduce
        aggregation tile size; using a larger tileScale (e.g. 2 or 4) may enable
        computations that run out of memory with the default.

    Returns:
      A function that accepts an ee.Image and reduces it by region, according to
      the provided arguments.
    """

    def reduce_region_function(img):
        """Applies the ee.Image.reduceRegion() method.

        Args:
          img:
            An ee.Image to reduce to a statistic by region.

        Returns:
          An ee.Feature that contains properties representing the image region
          reduction results per band and the image timestamp formatted as
          milliseconds from Unix epoch (included to enable time series plotting).
        """

        stat = img.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            crs=crs,
            bestEffort=bestEffort,
            maxPixels=maxPixels,
            tileScale=tileScale)

        return ee.Feature(geometry, stat).set(
            {
                'millis': img.date().millis()
            }
        )

    return reduce_region_function


def addNDVI(image):
    spacecraft_id = ee.String(image.get('SPACECRAFT_ID'))
    ndvi_bands = ee.Algorithms.If(
        spacecraft_id.equals('LANDSAT_5'),
        ['SR_B4', 'SR_B3'],
        ee.Algorithms.If(
            spacecraft_id.equals('LANDSAT_7'),
            ['SR_B4', 'SR_B3'],
            ['SR_B5', 'SR_B4']  # for Landsat 8
        )
    )
    return image.addBands(image.normalizedDifference(ndvi_bands).rename('NDVI'))


def addNBR(image):
    spacecraft_id = ee.String(image.get('SPACECRAFT_ID'))
    nbr_bands = ee.Algorithms.If(
        spacecraft_id.equals('LANDSAT_5'),
        ['SR_B7', 'SR_B4'],
        ee.Algorithms.If(
            spacecraft_id.equals('LANDSAT_7'),
            ['SR_B7', 'SR_B4'],
            ['SR_B7', 'SR_B5']  # for Landsat 8
        )
    )
    return image.addBands(image.normalizedDifference(nbr_bands).rename('NBR'))


def cloudMaskL457(image, spacecraft_id):
    cloud = ee.Image(0)

    if spacecraft_id in ['LANDSAT_5', 'LANDSAT_7']:
        qa = image.select(['QA_PIXEL'])
        basic_mask = qa.bitwiseAnd(1 << 1).Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4))
        confidence_mask = qa.bitwiseAnd(3 << 8).gte(1 << 8).And(qa.bitwiseAnd(3 << 10).gte(1 << 10))  # Modified line

        cloud = basic_mask.And(confidence_mask)

    elif spacecraft_id == 'LANDSAT_8':
        qa = image.select(['QA_PIXEL'])
        basic_mask = qa.bitwiseAnd(1 << 1).Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4)).Or(qa.bitwiseAnd(1 << 2))
        confidence_mask = qa.bitwiseAnd(3 << 8).gte(1 << 8).And(qa.bitwiseAnd(3 << 10).gte(1 << 10))  # Modified line

        cloud = basic_mask.And(confidence_mask)

    mask = cloud.eq(0)

    return image.updateMask(mask)


def scale_factors(image):
    optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    image = image.addBands(optical_bands, None, True)
    return image
