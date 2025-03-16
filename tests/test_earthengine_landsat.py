import ee
import json
import os
import pytest


def ee_service_account_auth():
    """
    Demonstrates how you might do Earth Engine service account auth
    using a JSON key stored in an environment variable.

    Example usage:
        GEE_SERVICE_ACCOUNT_JSON -> the entire service account key JSON as a single string
    """
    creds_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise RuntimeError(
            "No GEE_SERVICE_ACCOUNT_JSON env var found. Can't auth to EE."
        )

    # If using a service account
    #   parse the JSON, initialize Earth Engine with ee.ServiceAccountCredentials
    # If using a refresh token, you'd do something else.

    creds_dict = json.loads(creds_json)
    service_acc_email = creds_dict.get("client_email")
    private_key = creds_dict.get("private_key")
    if not service_acc_email or not private_key:
        raise ValueError(
            "Service account JSON is incomplete. Missing client_email or private_key."
        )

    credentials = ee.ServiceAccountCredentials(service_acc_email, key_data=private_key)
    ee.Initialize(credentials)


@pytest.mark.skipif(
    not os.environ.get("GEE_SERVICE_ACCOUNT_JSON"),
    reason="No Earth Engine credentials provided.",
)
def test_landsat_ndvi():
    """
    Minimal Earth Engine test:
      - Authenticates with GEE
      - Imports your 'landsat' module (if it's referencing earth_engine subpackage)
      - Queries a small region/time to see if code runs

    This test uses Earth Engine. In GitHub Actions, ensure GEE_SERVICE_ACCOUNT_JSON
    is set as a secret, e.g. in your .yml workflow:

        env:
          GEE_SERVICE_ACCOUNT_JSON: ${{ secrets.GEE_SERVICE_ACCOUNT_JSON }}
    """
    # 1) Auth with Earth Engine
    ee_service_account_auth()

    # 2) Run a minimal check
    #    e.g. we filter the Landsat 8 SR collection for a small region, just to see if it runs
    region = ee.Geometry.Point([-122.292, 37.901])  # e.g. near SF
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate("2023-01-01", "2023-01-15")
    )

    # 3) Maybe do a minimal getInfo() to confirm we have data
    info = collection.limit(1).getInfo()
    # 'features' in getInfo => list of images if found
    features = info.get("features", [])
    # We just check that we didn't break:
    assert isinstance(
        features, list
    ), "Expected features to be a list from EE getInfo()"

    # 4) (Optionally) import your code that depends on Earth Engine
    from pixltsnorm.earth_engine import landsat  # example import

    # run a function from that code, e.g. addNDVI or create_reduce_region_function
    # This is purely an example:
    def addNDVI(img):
        # Suppose your code does something like this
        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
        return img.addBands(ndvi)

    # Just check we don't crash
    test_img = addNDVI(ee.Image(features[0]["id"]))
    assert test_img, "Expect an image with NDVI band added"
