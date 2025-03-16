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


def chain_global_bridging(dfs, target_index=None, outlier_threshold=0.2):
    """
    A two-pass bridging approach mapping each DataFrame in `dfs` onto
    the scale of `dfs[target_index]` (or onto the last DataFrame if no
    `target_index` is provided).

    It returns a dictionary with:
      - "transforms": the (slope, intercept) for each DFS -> target
      - "pairwise_left": bridging results for the left pass
      - "pairwise_right": bridging results for the right pass
      - "harmonized": a list of the actual transformed DataFrames

    Steps:
      1) Verify adjacency overlap for all (dfs[i], dfs[i+1]) to ensure bridging is possible
         (i.e., they share at least one date column excluding 'lon'/'lat').
      2) Initialize transforms[target_index] = (1.0, 0.0) as the identity transform.
      3) Left pass: for i in descending order from `target_index` down to 1:
         - bridging (i-1 -> i) in forward direction
         - chain with i->target => (i-1)->target
      4) Right pass: for i in ascending order from `target_index` up to n-2:
         - bridging (i -> i+1) => (i+1)->target
      5) Finally, construct a "harmonized" list by applying each sensor's slope/intercept
         to its DataFrame columns.

    Requirements/Assumptions:
      - `dfs` is in an order where bridging each adjacency (i->i+1) is valid in the
        forward direction (commonly chronological).
      - Overlap is checked. If any adjacency shares no date columns, a ValueError is raised.
      - The bridging operation must find at least one valid data point; otherwise a ValueError
        is raised (no overlap or all outliers).
      - Bridging is performed by a global_bridging(...) function, returning
        { "coef": slope, "intercept": intercept, "filtered_a","filtered_b","n_initial","n_used" }.

    Args:
        dfs (list of pd.DataFrame):
            Each DataFrame has date-based columns plus optional 'lon','lat'.
        target_index (int or None, optional):
            Index in [0..len(dfs)-1]. If None, defaults to the last DF in `dfs`.
        outlier_threshold (float, optional):
            Threshold for removing outlier pairs in bridging. Default 0.2.

    Returns:
        dict with:
         - "transforms":  length==len(dfs). Each transforms[i] = (slope_i, intercept_i)
           meaning "apply y = slope_i*x + intercept_i to map DF i's scale -> DF[target_index]."
         - "pairwise_left":  list of ((i-1, i), bridging_result) from the left pass
         - "pairwise_right": list of ((j, j+1), bridging_result) from the right pass
         - "harmonized": the actual list of transformed DataFrames

    Raises:
        ValueError:
          - If any adjacent pair has zero overlap in columns,
          - Or bridging yields n_used=0 for an adjacency.
    """
    n = len(dfs)
    if n < 2:
        raise ValueError("Must have at least two DataFrames for chain bridging.")

    # If target_index is None, default to the last DataFrame in the list
    if target_index is None:
        target_index = n - 1

    if not (0 <= target_index < n):
        raise ValueError(f"target_index {target_index} out of range [0..{n-1}]")

    # 1) Check adjacency overlap
    skip_cols = {'lon', 'lat'}
    for i in range(n - 1):
        cols_a = set(dfs[i].columns)   - skip_cols
        cols_b = set(dfs[i+1].columns) - skip_cols
        overlap = cols_a.intersection(cols_b)
        if not overlap:
            raise ValueError(f"No overlapping date columns between dfs[{i}] and dfs[{i+1}]. "
                             "Cannot chain adjacency bridging.")

    # Initialize the slope/intercept for each sensor -> target
    transforms = [None]*n
    transforms[target_index] = (1.0, 0.0)  # The target sensor maps to itself by identity

    pairwise_left = []
    pairwise_right = []

    # 2) Left pass: from i=target_index down to i=1
    for i in range(target_index, 0, -1):
        bridging_res = global_bridging(dfs[i-1], dfs[i], outlier_threshold=outlier_threshold)
        pairwise_left.append(((i-1, i), bridging_res))

        if bridging_res.get("n_used", 0) < 1:
            raise ValueError(f"No valid overlap bridging dfs[{i-1}]->dfs[{i}] in left pass.")

        a_i = bridging_res["coef"]
        b_i = bridging_res["intercept"]

        slope_i, inter_i = transforms[i]  # i->target
        slope_new = a_i * slope_i
        inter_new = a_i * inter_i + b_i
        transforms[i-1] = (slope_new, inter_new)

    # 3) Right pass: from i=target_index up to i=n-2
    for i in range(target_index, n - 1):
        bridging_res = global_bridging(dfs[i], dfs[i+1], outlier_threshold=outlier_threshold)
        pairwise_right.append(((i, i+1), bridging_res))

        if bridging_res.get("n_used", 0) < 1:
            raise ValueError(f"No valid overlap bridging dfs[{i}]->dfs[{i+1}] in right pass.")

        a_i = bridging_res["coef"]
        b_i = bridging_res["intercept"]

        slope_i, inter_i = transforms[i]  # i->target
        slope_new = slope_i * a_i
        inter_new = a_i * inter_i + b_i
        transforms[i+1] = (slope_new, inter_new)

    # 4) Create transformed copies of each DataFrame
    harmonized = []
    for i, df in enumerate(dfs):
        slope_i, intercept_i = transforms[i]
        new_df = df.copy()

        # Apply slope/intercept to all numeric columns except 'lon','lat'
        numeric_cols = [c for c in new_df.columns if c not in skip_cols]
        new_df[numeric_cols] = new_df[numeric_cols].apply(lambda col: col*slope_i + intercept_i)

        harmonized.append(new_df)

    return {
        "transforms":      transforms,
        "pairwise_left":   pairwise_left,
        "pairwise_right":  pairwise_right,
        "harmonized": harmonized
    }
