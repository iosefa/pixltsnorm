"""
Global Harmonization Module
===========================

This module provides a global (scene-wide) approach to bridging or harmonizing
two or three sensors. For example, L5->L7 and L7->L8, then chain them for L5->L8.

Functions:
- global_bridging(dfA, dfB, outlier_threshold)
- chain_global_bridging(df_l5, df_l7, df_l8, outlier_thresholds)

Usage:
    from global_harmonize import chain_global_bridging
    result = chain_global_bridging(df_l5, df_l7, df_l8)
    # => Produces global slope/intercept for each pair and their chain.
"""


import numpy as np
import pandas as pd

from .harmonize import harmonize_series


def global_bridging(dfA, dfB, outlier_threshold=0.2):
    """
    Perform a global bridging between two sensors (A -> B) by:
      1) finding overlapping date columns
      2) flattening all pixel/time data pairs
      3) removing NaNs
      4) calling harmonize_series for outlier filtering & regression

    Args:
        dfA (pd.DataFrame): Numeric data for sensor A, shape [N x (M+2)]
                            where last 2 columns might be [lon, lat].
                            The data columns are typically named like "YYYY-MM".
        dfB (pd.DataFrame): Numeric data for sensor B, same structure as dfA
        outlier_threshold (float): threshold for removing pairs
                                   where abs(A - B) > threshold

    Returns:
        dict: {
            "coef": slope (float),
            "intercept": intercept (float),
            "filtered_a": np.array (after outliers removed),
            "filtered_b": np.array (after outliers removed),
            "n_initial": number_of_pairs_before_outlier_filter,
            "n_used": number_of_pairs_after_outlier_filter
        }

    Example usage:
        result = global_bridging(dfA, dfB)
        slope, intercept = result["coef"], result["intercept"]
    """

    # 1) Find columns that are numeric data columns (excluding "lon" and "lat" if present)
    setA_cols = set(dfA.columns) - {"lon", "lat"}
    setB_cols = set(dfB.columns) - {"lon", "lat"}

    # Overlap
    overlap_cols = sorted(setA_cols.intersection(setB_cols))
    if not overlap_cols:
        raise ValueError("No overlapping date columns between the two sensors. "
                         "Cannot do global bridging.")

    # 2) Subset both DataFrames to only those columns
    dfA_sub = dfA[overlap_cols]
    dfB_sub = dfB[overlap_cols]

    # 3) Flatten
    arrA = dfA_sub.values.flatten()  # shape => N*M flattened
    arrB = dfB_sub.values.flatten()

    # 4) Remove NaNs
    mask = (~np.isnan(arrA)) & (~np.isnan(arrB))
    arrA = arrA[mask]
    arrB = arrB[mask]

    # We'll store how many we had initially
    n_initial = len(arrA)

    # 5) Use harmonize_series to handle outlier filtering + regression
    #    This returns slope/intercept plus the filtered arrays.
    #    By default, outlier_threshold=0.2 => filter pairs where abs(A - B) > 0.2
    result = harmonize_series(arrA, arrB, outlier_threshold=outlier_threshold)

    slope = result["coef"]
    intercept = result["intercept"]
    filtered_a = result["filtered_a"]
    filtered_b = result["filtered_b"]
    n_used = len(filtered_a)

    return {
        "coef": slope,
        "intercept": intercept,
        "filtered_a": filtered_a,
        "filtered_b": filtered_b,
        "n_initial": n_initial,
        "n_used": n_used
    }


def chain_global_bridging(df_l5, df_l7, df_l8, outlier_thresholds=(0.2, 0.2)):
    """
    Perform a global bridging for L5->L7 and L7->L8, then chain them
    so we get final L5->L8 = (a*c)*L5 + (a*d + b).

    Args:
        df_l5, df_l7, df_l8 (pd.DataFrame): data for each sensor,
            each shaped [N x (M+2)] with date columns + [lon, lat].
        outlier_thresholds (tuple): (threshold_5_7, threshold_7_8) for outlier filtering

    Returns:
        dict with:
          "L5->L7": { "coef": c, "intercept": d, ... }
          "L7->L8": { "coef": a, "intercept": b, ... }
          "L5->L8": { "coef": x, "intercept": y }
    """

    # 1) L5->L7
    res_5_7 = global_bridging(df_l5, df_l7, outlier_threshold=outlier_thresholds[0])
    c, d = res_5_7["coef"], res_5_7["intercept"]

    # 2) L7->L8
    res_7_8 = global_bridging(df_l7, df_l8, outlier_threshold=outlier_thresholds[1])
    a, b = res_7_8["coef"], res_7_8["intercept"]

    # 3) Chain => L5->L8
    x = a * c
    y = a * d + b

    return {
        "L5->L7": res_5_7,
        "L7->L8": res_7_8,
        "L5->L8": {
            "coef": x,
            "intercept": y
        }
    }