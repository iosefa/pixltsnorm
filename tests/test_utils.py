import pytest
import pandas as pd
import numpy as np
from pixltsnorm.utils import unify_and_extract_timeseries, filter_outliers


def test_filter_outliers():
    a = np.array([0.1, 0.2, 0.6, 0.9])
    b = np.array([0.12, 0.22, 0.55, 1.2])
    # threshold=0.2 => differences bigger than 0.2 are removed
    # The difference for last pair => abs(0.9 - 1.2) = 0.3 => out
    fa, fb = filter_outliers(a, b, threshold=0.2)
    # We expect first 3 pairs remain
    assert len(fa) == 3
    assert len(fb) == 3


def test_unify_extract_timeseries():
    # Create DataFrames with date-based columns
    df1 = pd.DataFrame(
        {
            "2000-01": [1.0, 2.0],
            "2000-02": [1.1, 2.1],
            "lon": [-95.0, -95.2],
            "lat": [36.0, 36.2],
        }
    )
    df2 = pd.DataFrame(
        {
            "2000-02": [1.2, 2.2],
            "2000-01": [1.0, 2.0],
            "lon": [-95.5, -95.7],
            "lat": [36.5, 36.7],
        }
    )

    # unify date columns, row_index=0 => first row of each
    (arr_list, dates, time_axis) = unify_and_extract_timeseries(
        [df1, df2], row_index=0, skip_cols=("lon", "lat"), date_format="%Y-%m"
    )
    # We expect union of columns => [2000-01, 2000-02]
    # for row_index=0 => df1 => [1.0, 1.1], df2 => [1.0, 1.2], reindexed
    # Let's check shapes
    assert len(arr_list) == 2  # two DF arrays
    assert len(dates) == 2
    assert list(dates.strftime("%Y-%m")) == ["2000-01", "2000-02"]
    # time_axis => [0,1]
    assert len(time_axis) == 2

    # Check actual array content
    assert np.allclose(arr_list[0], [1.0, 1.1])  # from df1 row 0
    assert np.allclose(arr_list[1], [1.0, 1.2])  # from df2 row 0
