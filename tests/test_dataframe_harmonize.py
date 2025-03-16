import pytest
import pandas as pd
from pixltsnorm.dataframe_harmonize import DataFrameHarmonizer


def test_dataframe_harmonizer_global():
    """
    Test DataFrameHarmonizer with approach='global'.
    """
    # Create two small sample DataFrames
    dfA = pd.DataFrame({"time1": [0.1, 0.2, 0.3], "time2": [0.15, 0.25, 0.35]})
    # Suppose dfB = slope=1.5, intercept=+0.05
    dfB = dfA * 1.5 + 0.05

    harm_df = DataFrameHarmonizer(
        method="linear", approach="global", outlier_threshold=0.6
    )
    harm_df.fit([dfA, dfB], target_index=1)

    # transforms_[0] => slope, intercept from dfA->dfB
    slope, intercept = harm_df.transforms_[0]
    # Expect ~1.5, 0.05
    assert pytest.approx(slope, 0.1) == 1.5
    assert pytest.approx(intercept, 0.1) == 0.05


def test_dataframe_harmonizer_local():
    """
    Test DataFrameHarmonizer with approach='local'.
    Each row gets separate slope/intercept.
    """
    dfA = pd.DataFrame({"time1": [0.1, 0.3], "time2": [0.2, 0.4]})
    # For row 0 => slope=1.2 intercept=0.0
    # For row 1 => slope=2.0 intercept=0.1
    # We'll just do this by row
    dfB = pd.DataFrame(
        {
            "time1": [0.1 * 1.2 + 0.0, 0.3 * 2.0 + 0.1],  # [0.12, 0.7]
            "time2": [0.2 * 1.2 + 0.0, 0.4 * 2.0 + 0.1],  # [0.24, 0.9]
        }
    )

    harm_df = DataFrameHarmonizer(
        method="linear", approach="local", outlier_threshold=0.6
    )
    harm_df.fit([dfA, dfB], target_index=1)

    # transforms_[0] is a dict => {slope: arr, inter: arr}
    local_transform = harm_df.transforms_[0]
    slope_arr = local_transform["slope"]
    inter_arr = local_transform["inter"]

    # row 0 => slope ~1.2, intercept ~0
    assert pytest.approx(slope_arr[0], 0.05) == 1.2
    assert pytest.approx(inter_arr[0], 0.05) == 0.0

    # row 1 => slope ~2.0, intercept ~0.1
    assert pytest.approx(slope_arr[1], 0.05) == 2.0
    assert pytest.approx(inter_arr[1], 0.05) == 0.1


def test_dataframe_harmonizer_transform_dfs():
    """
    Check that get_harmonized_dfs() yields expected results.
    """
    dfA = pd.DataFrame({"t1": [0.0, 1.0], "t2": [0.5, 1.5]})
    dfB = dfA * 2 + 0.05  # slope=2, intercept=0.05 for every row

    harm_df = DataFrameHarmonizer(method="linear", approach="global")
    harm_df.fit([dfA, dfB], target_index=1)
    out_dfs = harm_df.get_harmonized_dfs([dfA, dfB])
    dfA_h, dfB_h = out_dfs

    # dfB_h should be ~ dfB (the target remains basically the same)
    assert pytest.approx(dfB_h["t1"][0], 0.001) == dfB["t1"][0]
    # dfA_h should match dfB after the transform
    # Let's check row[0, col 't1'] => original 0.0 => mapped => 2.0*0.0 + 0.05 => 0.05
    # We can confirm from the output DF
    assert pytest.approx(dfA_h["t1"][0], 0.01) == 0.05
