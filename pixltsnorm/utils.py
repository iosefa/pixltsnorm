import numpy as np
import pandas as pd


def unify_and_extract_timeseries(dfs, row_index=0, skip_cols=("lon","lat"), date_format="%Y-%m"):
    """
    Unify the date columns across multiple DataFrames by:
      1) Identifying columns not in `skip_cols`.
      2) Taking the union of those columns from all DataFrames.
      3) Reindexing each DataFrame to that union (missing => NaN).
      4) Extracting row_index's time series from each DataFrame.
      5) Returning the union columns (converted to datetime) and a time_axis.

    Args:
        dfs (list[pd.DataFrame]): list of DataFrames (e.g. [df_l5, df_l7, df_l8]).
        row_index (int): which row to extract NDVI / data from.
        skip_cols (tuple or set): columns to exclude (e.g. "lon","lat").
        date_format (str): a strftime format to parse the columns to datetime.

    Returns:
        tuple: (list_of_arrays, dates, time_axis)
          - list_of_arrays: list of np.ndarrays, each array is the row data from a DataFrame
          - dates: pd.DatetimeIndex (converted from union columns)
          - time_axis: np.arange(len(dates))

    Example:
        df_l5, df_l7, df_l8 = ...
        arrays, dates, time_axis = unify_and_extract_timeseries([df_l5, df_l7, df_l8])
    """
    columns_list = []
    for df in dfs:
        valid_cols = [c for c in df.columns if c not in skip_cols]
        columns_list.append(valid_cols)

    union_cols = sorted(set().union(*columns_list))

    reindexed_dfs = []
    for df in dfs:
        reindexed_df = df.reindex(columns=union_cols + list(skip_cols))
        reindexed_dfs.append(reindexed_df)

    list_of_arrays = []
    for rdf in reindexed_dfs:
        arr = rdf.iloc[row_index, :-len(skip_cols)].values
        list_of_arrays.append(arr)

    dates = pd.to_datetime(union_cols, format=date_format, errors="coerce")
    time_axis = np.arange(len(dates))

    return list_of_arrays, dates, time_axis


def filter_outliers(sensor_a_values, sensor_b_values, threshold=0.2):
    """
    Remove pairs where abs(sensor_a - sensor_b) > threshold.

    Args:
        sensor_a_values (array-like): data array for sensor A
        sensor_b_values (array-like): data array for sensor B
        threshold (float): difference threshold

    Returns:
        tuple of arrays: (filtered_a, filtered_b)
    """
    a = np.array(sensor_a_values)
    b = np.array(sensor_b_values)

    diff = np.abs(a - b)
    mask = diff <= threshold
    return a[mask], b[mask]