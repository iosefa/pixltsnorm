"""
dataframe_harmonize.py
======================

This module provides a class (`DataFrameHarmonizer`) for bridging and aligning
multiple pandas DataFrames onto a common reference scale. It offers:

- **Two-pass Chaining**: Uses a left pass and right pass to apply pairwise
  adjacencies in a sequence, eventually mapping each DataFrame to the chosen
  target DataFrame's scale.

- **Global vs. Local**: 
  - *Global approach*: Flatten and compute a single slope/intercept for the entire dataset.
  - *Local approach*: Fit a separate slope/intercept for each row (e.g., per-pixel),
    storing arrays of transforms for more fine-grained harmonization.

- **Method**:
  - `'linear'` (multiple DataFrames supported): Simple linear calibration for each adjacency.
  - `'seasonal_decompose'` (only valid for exactly two DataFrames): 
    Not fully implemented for local bridging, but can be adapted if needed.

Use this class if you have multiple DataFrames (e.g., NDVI data from different
sensors) and want to align them onto one final sensor's scale, either at a 
whole-dataset (global) or row-by-row (local) level. The chaining logic ensures
that each intermediate adjacency (df[i]→df[i+1]) is composed into a final 
transform (df[i]→target).
"""


import numpy as np

from .harmonize import Harmonizer


class DataFrameHarmonizer:
    """
    A class for chaining (two-pass) bridging of multiple DataFrames onto
    a single target DataFrame's scale. Supports:
      - 'global' approach: flatten all overlapping columns to one array per adjacency
      - 'local' approach: do a per-row bridging (each row gets its own slope/intercept)

    Also supports:
      - method='linear' for any number of DataFrames
      - method='seasonal_decompose' if exactly 2 DataFrames (n=2)
        (raises NotImplementedError for multi-sensor seasonal)

    After fit(), call get_harmonized_dfs() to apply final transforms
    (either single slope/intercept or per-row arrays) to each DataFrame.

    Two-pass bridging steps:
      1) LEFT pass: from i=target_index down to i=1
         - adjacency (i-1, i)
         - bridging => slope/intercept
         - compose with transforms[i]->target => transforms[i-1]->target
      2) RIGHT pass: from i=target_index up to i=n-2
         - adjacency (i, i+1)
         - bridging => slope/intercept
         - compose with i->target => (i+1)->target
    """

    def __init__(self, method='linear', period=None,
                 outlier_threshold=0.2, approach='global'):
        """
        Args:
            method (str): 'linear' or 'seasonal_decompose'
            period (int or None): if using 'seasonal_decompose' (only valid for 2 DF)
            outlier_threshold (float): threshold for outlier filter
            approach (str): 'global' or 'local'
        """
        self.method = method
        self.period = period
        self.outlier_threshold = outlier_threshold
        self.approach = approach  # 'global' or 'local'

        self._skip_cols = {'lon','lat'}
        self.transforms_ = None
        self.target_index_ = None
        self.pairwise_left_ = []
        self.pairwise_right_ = []

    def fit(self, dfs, target_index=None):
        """
        Fit the linear chain harmonization model using the specified DataFrames.

        This method performs a two-pass (left and right) adjustment of the provided
        dataframes in order to harmonize them based on a specified target index. The
        adjustment can be performed globally or locally depending on the approach
        selected. Multi-sensor chaining with certain methods may have limitations,
        as noted in the functionality.

        :param dfs: A list of pandas DataFrames that need to be harmonized. Each DataFrame
            represents sensor data or different datasets that require alignment.
        :param target_index: An optional integer specifying the index of the target
            DataFrame that serves as the reference for harmonization. Defaults to the last
            DataFrame in the list.

        :return: Returns self (the fitted instance of the harmonization object).

        :raises ValueError:
            - If the number of provided DataFrames is less than two.
            - If the provided target_index is out of the valid range [0..n-1].

        :raises NotImplementedError: If multi-sensor chaining with the
            `seasonal_decompose` method is attempted and the number of DataFrames
            exceeds two.
        """
        n = len(dfs)
        if n < 2:
            raise ValueError("Need at least two DataFrames for chain harmonization.")

        if target_index is None:
            target_index = n-1
        if not (0 <= target_index < n):
            raise ValueError(f"target_index {target_index} out of range [0..{n-1}]")

        if self.method=='seasonal_decompose' and n>2:
            raise NotImplementedError(
                "multi-sensor chaining with method='seasonal_decompose' not supported."
            )

        self.target_index_ = target_index
        # Initialize transforms
        if self.approach=='global':
            # single slope/intercept per DF
            self.transforms_ = [None]*n
            self.transforms_[target_index] = (1.0, 0.0)
        else:
            # local => store slope/intercept arrays
            # figure out how many rows from target DF
            n_rows = len(dfs[target_index])
            slope_arr = np.ones(n_rows, dtype=float)
            inter_arr = np.zeros(n_rows, dtype=float)
            self.transforms_ = [None]*n
            self.transforms_[target_index] = {"slope": slope_arr, "inter": inter_arr}

        self.pairwise_left_.clear()
        self.pairwise_right_.clear()

        # LEFT pass
        for i in range(target_index, 0, -1):
            bridging_res = self._bridge_pair(dfs[i-1], dfs[i])
            self.pairwise_left_.append(((i-1,i), bridging_res))
            # bridging_res => { "coef", "intercept" }, shape=() or shape=[n_rows]
            self._compose_into(i-1, i, bridging_res)

        # RIGHT pass
        for i in range(target_index, n-1):
            bridging_res = self._bridge_pair(dfs[i], dfs[i+1])
            self.pairwise_right_.append(((i, i+1), bridging_res))
            self._compose_into(i+1, i, bridging_res, reverse=False)

        return self

    def get_harmonized_dfs(self, dfs):
        """
        Transforms a list of dataframes using previously fitted transformations. This method should
        only be called after the fit method has been executed and transformations have been prepared.

        :param dfs: A list of pandas DataFrame objects to be harmonized.
        :type dfs: list[pandas.DataFrame]
        :return: A list of transformed pandas DataFrame objects, with transformations defined by the
            previously fitted process.
        :rtype: list[pandas.DataFrame]
        :raises RuntimeError: If the fit method has not been called prior to using this method.
        """
        if self.transforms_ is None:
            raise RuntimeError("Call .fit() before get_harmonized_dfs().")

        out_list = []
        for i, df in enumerate(dfs):
            out_list.append(self._transform_df(i, df))
        return out_list

    # ---------------------------
    # Internal bridging logic
    # ---------------------------
    def _bridge_pair(self, dfA, dfB):
        """
        Bridge two dataframes based on the specified approach.

        This method determines whether to bridge the two input dataframes using the
        global or local approach as specified by the `self.approach` attribute. If
        the approach is set to 'global', the `_bridge_global` method is called. For
        any other value of the approach, the `_bridge_local` method is invoked.

        :param dfA: The first dataframe to be bridged.
        :type dfA: pandas.DataFrame
        :param dfB: The second dataframe to be bridged.
        :type dfB: pandas.DataFrame
        :return: The resulting dataframe after applying the bridging method.
        :rtype: pandas.DataFrame
        """
        if self.approach=='global':
            return self._bridge_global(dfA, dfB)
        else:
            return self._bridge_local(dfA, dfB)

    def _bridge_global(self, dfA, dfB):
        """
        Performs global harmonization between two dataframes by calculating the overlap, cleaning
        the data, and applying a predefined harmonization method. This operation ensures alignment
        of distributions and removes discrepancies based on a specified method and parameters.

        :param dfA: First DataFrame containing data to be harmonized.
        :type dfA: pandas.DataFrame
        :param dfB: Second DataFrame containing data to be harmonized.
        :type dfB: pandas.DataFrame

        :return: A dictionary containing the calculated coefficients for slope and intercept
                 after harmonization.
        :rtype: dict

        :raises ValueError: If there is no valid data left after removing NaN values during
                            the global adjacency validation step.
        """
        overlap = self._overlap_columns(dfA, dfB)
        arrA, arrB = self._flatten_and_clean(dfA[overlap], dfB[overlap])
        if len(arrA)==0:
            raise ValueError("No valid data after removing NaNs (global adjacency).")

        # If n=2 or method='linear' => no problem
        small_harm = Harmonizer(method=self.method,
                                period=self.period,
                                outlier_threshold=self.outlier_threshold)
        small_harm.fit([arrA, arrB], target_index=1)
        slope, intercept = small_harm.transforms_[0]
        return {"coef": slope, "intercept": intercept}

    def _bridge_local(self, dfA, dfB):
        """
        Executes a local bridging operation between two dataframes (dfA and dfB) by calculating
        linear regression coefficients (slopes and intercepts) for each row based on overlapping
        columns. The function assumes that both dataframes have the same number of rows and
        performs outlier filtering and single-sensor bridging based on the data.

        :param dfA: First dataframe used in the bridging operation.
        :type dfA: pandas.DataFrame
        :param dfB: Second dataframe used in the bridging operation.
        :type dfB: pandas.DataFrame
        :return: Dictionary with two keys: "coef", containing an array of slopes calculated for
                 each row, and "intercept", containing an array of intercepts calculated for each row.
        :rtype: dict
        :raises ValueError: If the number of rows in dfA and dfB does not match.
        """
        overlap = self._overlap_columns(dfA, dfB)
        n_rows = len(dfA)
        if len(dfB)!=n_rows:
            raise ValueError("local bridging requires DF A and B have same number of rows")

        slopes = np.full(n_rows, np.nan, dtype=float)
        intercepts = np.full(n_rows, np.nan, dtype=float)

        # For each row i
        for i in range(n_rows):
            rowA = dfA[overlap].iloc[i].values.astype(float)
            rowB = dfB[overlap].iloc[i].values.astype(float)

            # Remove NaNs
            mask = (~np.isnan(rowA)) & (~np.isnan(rowB))
            rowA_valid = rowA[mask]
            rowB_valid = rowB[mask]
            if len(rowA_valid)==0:
                # remain NaN
                continue

            # outlier filter => remove pairs where |A-B|>threshold
            rowA_f, rowB_f = self._filter_outliers(rowA_valid, rowB_valid)
            if len(rowA_f)==0:
                continue  # remain NaN

            # Now we do a single-sensor bridging => OLS or seasonal if exactly 2 DF
            slope_i, intercept_i = self._fit_single_pair(rowA_f, rowB_f)
            slopes[i] = slope_i
            intercepts[i] = intercept_i

        return {"coef": slopes, "intercept": intercepts}

    def _fit_single_pair(self, arrA, arrB):
        """
        Fits a pair of data arrays to a specified method, calculating either a simple linear
        relationship or raising an error for methods not fully implemented.

        :param arrA: The first input array for fitting, representing independent variable data.
                     Must be of compatible shape to `arrB`.
        :type arrA: numpy.ndarray
        :param arrB: The second input array for fitting, representing dependent variable data.
                     Must be of compatible shape to `arrA`.
        :type arrB: numpy.ndarray
        :return: A tuple `(slope, intercept)` when the fitting method is 'linear'; the coefficients
                 describing the linear relationship between input arrays. Raises an exception for
                 unsupported methods.
        :rtype: tuple[float, float]
        :raises ValueError: If the specified `method` attribute is not recognized or supported.
        :raises NotImplementedError: If the method is 'seasonal_decompose' or any feature-dependent
                                      functionality is attempted that is not implemented.
        """
        # If method='linear'
        if self.method=='linear':
            # OLS: B ~ slope*A + intercept
            X = np.column_stack((np.ones_like(arrA), arrA))
            beta = np.linalg.lstsq(X, arrB, rcond=None)[0]
            intercept, slope = beta[0], beta[1]
            return slope, intercept
        elif self.method=='seasonal_decompose':
            # For local bridging => row's data => might do a decomposition =>
            # but you need a period. This might be short or missing.
            # We'll do a minimal approach: We'll skip implementing a real decomposition
            # for every row. Or you can do an actual statsmodels seasonal_decompose if you want.
            raise NotImplementedError("local bridging + seasonal_decompose not fully implemented.")
        else:
            raise ValueError(f"Unknown method='{self.method}'")

    def _filter_outliers(self, arrA, arrB):
        """
        Filters outliers from the input arrays based on the specified threshold.

        This method compares the absolute difference between corresponding elements
        of two input arrays and filters out the elements that exceed the outlier
        threshold. Only elements satisfying the condition are returned.

        :param arrA: The first input NumPy array to be filtered.
        :type arrA: numpy.ndarray
        :param arrB: The second input NumPy array to be filtered.
        :type arrB: numpy.ndarray
        :return: A tuple containing two NumPy arrays with outliers removed. The first
                 array corresponds to the filtered elements of arrA, and the second
                 array corresponds to the filtered elements of arrB.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        diffs = np.abs(arrA - arrB)
        mask = diffs <= self.outlier_threshold
        return arrA[mask], arrB[mask]

    def _compose_into(self, idxA, idxB, bridging_res, reverse=True):
        """
        Composes transformation coefficients into the transformation sequence. This
        method updates the transformation parameters for idxA based on the given
        parameters for idxB and the bridging coefficients. It handles both global
        and local approaches for transformations and determines whether to apply
        the composition in reverse order.

        :param idxA: Index of the transformation to be updated.
        :type idxA: int
        :param idxB: Index of the transformation to be used for reference.
        :type idxB: int
        :param bridging_res: Bridging coefficients containing "coef" (slope) and
            "intercept" values for the transformation operation.
        :type bridging_res: dict
        :param reverse: Indicator for whether the composition should be applied
            in reverse order. Defaults to True.
        :type reverse: bool, optional
        :return: None, the method updates the respective transformation in-place.
        """
        a_i = bridging_res["coef"]  # float or array
        b_i = bridging_res["intercept"]  # float or array

        # Are we in global or local approach?
        if self.approach == 'global':
            slopeB, interB = self.transforms_[idxB]  # a 2-tuple
            if reverse:
                new_slope = a_i * slopeB
                new_inter = a_i * interB + b_i
            else:
                new_slope = slopeB * a_i
                new_inter = a_i * interB + b_i
            self.transforms_[idxA] = (new_slope, new_inter)

        else:
            # local => transforms_[idxB] is a dict { "slope": array, "inter": array }
            slopeB_dict = self.transforms_[idxB]
            slopeB_arr = slopeB_dict["slope"]
            interB_arr = slopeB_dict["inter"]

            if reverse:
                new_slope = a_i * slopeB_arr
                new_inter = a_i * interB_arr + b_i
            else:
                new_slope = slopeB_arr * a_i
                new_inter = a_i * interB_arr + b_i

            self.transforms_[idxA] = {
                "slope": new_slope,
                "inter": new_inter
            }

    # ---------------------------
    # Internal utilities
    # ---------------------------
    def _overlap_columns(self, dfA, dfB):
        """
        Identifies overlapping column names from two dataframes, excluding columns
        specified to be skipped. The method checks for common column names between
        the two dataframes' sets of columns, ignoring those listed in the internal
        skip condition. If no overlap exists, an exception is raised.

        :param dfA: A Pandas DataFrame to compare for overlapping column names.
        :param dfB: Another Pandas DataFrame for comparison to identify overlapping
                    column names.
        :return: A sorted list of overlapping column names between `dfA` and `dfB`
                 while excluding skipped columns.
        :rtype: list[str]
        :raises ValueError: If there are no overlapping column names between the
                             two dataframes.
        """
        skip = self._skip_cols
        colsA = set(dfA.columns) - skip
        colsB = set(dfB.columns) - skip
        overlap = sorted(colsA.intersection(colsB))
        if not overlap:
            raise ValueError("No overlapping date columns between adjacency pair.")
        return overlap

    def _flatten_and_clean(self, dfA_sub, dfB_sub):
        """
        Flattens and cleans the input dataframes by converting them to 1D arrays and removing
        NaN values. The method ensures that only valid (non-NaN) corresponding elements from
        both arrays are retained, making the data suitable for further operations or analysis.

        :param dfA_sub: A pandas dataframe subset to be flattened and cleaned.
        :param dfB_sub: A pandas dataframe subset to be flattened and cleaned.
        :return: A tuple containing two 1D numpy arrays with NaN values removed. The first
         array corresponds to cleaned data from `dfA_sub`, and the second array corresponds
         to cleaned data from `dfB_sub`.
        """
        arrA = dfA_sub.values.flatten()
        arrB = dfB_sub.values.flatten()
        mask = (~np.isnan(arrA)) & (~np.isnan(arrB))
        return arrA[mask], arrB[mask]

    def _transform_df(self, df_index, df):
        """
        Apply a specific transformation to a dataframe based on the provided parameters.

        This method is responsible for transforming a dataframe using either a global or
        local approach. If the global approach is used, it applies a single set of slope
        and intercept values to the dataframe. For the local approach, it applies arrays
        of slope and intercept values that are specific to different parts of the dataframe.

        The appropriate transformation parameters are retrieved from the `transforms_`
        attribute for the given index. The transformation process is delegated to helper
        methods `_apply_global_transform` or `_apply_local_transform` depending on the
        selected approach.

        :param df_index: The index of the transformation to be applied, used to retrieve
                         the corresponding transformation parameters from `transforms_`.
        :type df_index: int
        :param df: The dataframe to which the transformation will be applied.
        :type df: pandas.DataFrame
        :return: The transformed dataframe after applying the specified transformation.
        :rtype: pandas.DataFrame
        """
        transform_i = self.transforms_[df_index]
        if self.approach=='global':
            slope, intercept = transform_i
            return self._apply_global_transform(df, slope, intercept)
        else:
            # local => transform_i is dict with slope/inter arrays
            slope_arr = transform_i["slope"]
            inter_arr = transform_i["inter"]
            return self._apply_local_transform(df, slope_arr, inter_arr)

    def _apply_global_transform(self, df, slope, intercept):
        """
        Applies a global linear transformation to the numeric columns of the given
        DataFrame while skipping specific columns. The transformation is applied using
        the formula: `new_value = (original_value * slope) + intercept`.

        :param df: The pandas DataFrame to be transformed.
        :param slope: The multiplier for the linear transformation.
        :param intercept: The added constant for the linear transformation.
        :return: A new pandas DataFrame with the linear transformation applied to all
            applicable numeric columns.
        :rtype: pandas.DataFrame
        """
        new_df = df.copy()
        numeric_cols = [c for c in df.columns if c not in self._skip_cols]
        for col in numeric_cols:
            new_df[col] = new_df[col]*slope + intercept
        return new_df

    def _apply_local_transform(self, df, slope_arr, inter_arr):
        """
        Applies a local transformation to the given DataFrame by scaling and shifting numeric
        column values based on per-row `slope_arr` and `inter_arr`.

        This method creates a new DataFrame with the transformed values while excluding any
        columns specified in the `_skip_cols` attribute. For numeric columns, it computes a
        per-row transformation using the formula:
            `new_val[r] = slope_arr[r]*old_val[r] + inter_arr[r]`.

        A ValueError is raised if the number of rows in the DataFrame and the length of
        `slope_arr` or `inter_arr` do not match.

        :param df: The input DataFrame with numeric and non-numeric columns.
        :type df: pandas.DataFrame
        :param slope_arr: A 1-dimensional array of scaling factors corresponding to the
            rows of the input DataFrame.
        :param inter_arr: A 1-dimensional array of shifting factors corresponding to the
            rows of the DataFrame.
        :return: A transformed DataFrame with numeric column values scaled and shifted based
            on the provided parameters.
        :rtype: pandas.DataFrame
        :raises ValueError: If the number of rows in the DataFrame and the length of
            `slope_arr` or `inter_arr` do not match.
        """
        new_df = df.copy()
        n_rows = len(df)
        if len(slope_arr)!=n_rows:
            raise ValueError("Mismatch in row count for local transform.")
        numeric_cols = [c for c in df.columns if c not in self._skip_cols]
        for col in numeric_cols:
            col_vals = new_df[col].values.astype(float)
            # per row: new_val[r] = slope_arr[r]*old_val[r] + inter_arr[r]
            col_vals = slope_arr * col_vals + inter_arr
            new_df[col] = col_vals
        return new_df
