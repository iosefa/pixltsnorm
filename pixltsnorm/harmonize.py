import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from .outlier_filter import filter_outliers
from .regression import fit_regression


class Harmonizer:
    """
    A class that harmonizes N sensors onto one target sensor's scale by
    chaining pairwise calibrations from the target sensor to the left
    and to the right. By default, the last sensor in the list is used
    as the target.

    This approach composes local (pairwise) linear transformations so that
    each sensor ultimately aligns with the target sensor. If 'linear' is
    selected, we simply fit slope/intercept for each adjacency and chain
    them. If 'seasonal_decompose' is used (only valid for two sensors),
    we de-seasonalize each sensor, fit a linear model to the residual, and
    then re-introduce the seasonal component on the target's scale..

    Attributes:
        method (str): 'linear' (for any number of sensors) or 'seasonal_decompose'
                      (supported only for 2 sensors).
        period (int or None): Period for seasonal decomposition.
        default_outlier_threshold (float): Outlier threshold for pairwise filtering.
        outlier_thresholds_ (list or None): Final array of thresholds used in chaining.
        target_index_ (int or None): Which sensor is used as the target reference.
        transforms_ (list of (slope, intercept)): For each sensor i,
            a tuple mapping sensor i onto the target sensor's scale.
        pairwise_left_ (list): Stores pairwise results from chaining left.
        pairwise_right_ (list): Stores pairwise results from chaining right.
        seasonal_map_a_ (dict or None): If 'seasonal_decompose' and 2 sensors,
            time -> seasonal component for sensor A (the non-target).
        seasonal_map_b_ (dict or None): If 'seasonal_decompose' and 2 sensors,
            time -> seasonal component for sensor B (the target).
    """

    def __init__(self, method='linear', period=None, outlier_threshold=0.2):
        """
        Args:
            method (str): 'linear' or 'seasonal_decompose'
            period (int, optional): Period for seasonal decomposition.
            outlier_threshold (float, optional): If using a single threshold for all adjacencies.
                                                 If you have multiple sensors, you can also pass
                                                 an array of thresholds to fit().
        """
        self.method = method
        self.period = period
        # If user wants multiple thresholds for multi-sensor, they can pass them at fit() time
        self.default_outlier_threshold = outlier_threshold

        self.outlier_thresholds_ = None
        self.transforms_ = None
        self.target_index_ = None
        self.pairwise_left_ = []
        self.pairwise_right_ = []

        # Seasonal maps for the special 2-sensor case
        self.seasonal_map_a_ = None
        self.seasonal_map_b_ = None

    def _harmonize_two_sensors(self, sensor_a_values, sensor_b_values, outlier_threshold, time_index=None):
        """
        Private helper to harmonize exactly 2 sensors, possibly with seasonal_decompose.
        Returns (coef, intercept) for A->B.
        May also store seasonal maps if using 'seasonal_decompose'.
        """
        # 1) Filter outliers
        a_filt, b_filt = filter_outliers(sensor_a_values, sensor_b_values, threshold=outlier_threshold)

        # 2) Linear
        if self.method == 'linear':
            model = fit_regression(a_filt, b_filt)
            return model['coef'], model['intercept']

        # 3) Seasonal Decomposition (2-sensor only)
        elif self.method == 'seasonal_decompose':
            if time_index is None or self.period is None:
                raise ValueError("time_index and period must be provided for method='seasonal_decompose'.")

            # Recreate mask to keep time alignment
            a_arr = np.array(sensor_a_values)
            b_arr = np.array(sensor_b_values)
            diffs = np.abs(b_arr - a_arr)
            mask = (diffs <= outlier_threshold)

            inlier_a = a_arr[mask]
            inlier_b = b_arr[mask]
            inlier_t = np.array(time_index)[mask]

            # Decompose each filtered series
            dec_a = seasonal_decompose(inlier_a, period=self.period, model='additive', extrapolate_trend='freq')
            dec_b = seasonal_decompose(inlier_b, period=self.period, model='additive', extrapolate_trend='freq')

            # Seasonal parts
            seas_a = dec_a.seasonal
            seas_b = dec_b.seasonal

            a_deseasoned = inlier_a - seas_a
            b_deseasoned = inlier_b - seas_b

            model = fit_regression(a_deseasoned, b_deseasoned)
            coef, intercept = model['coef'], model['intercept']

            # Store maps
            self.seasonal_map_a_ = dict(zip(inlier_t, seas_a))
            self.seasonal_map_b_ = dict(zip(inlier_t, seas_b))

            return coef, intercept

        else:
            raise ValueError(f"Unknown method='{self.method}'. Use 'linear' or 'seasonal_decompose'.")

    def fit(self, sensor_list, target_index=None, outlier_thresholds=None, time_indexes=None):
        """
        Fits the chain harmonization approach to map each sensor in sensor_list
        onto the scale of sensor_list[target_index], or the last sensor if
        target_index is None.

        Args:
            sensor_list (list of array-like): e.g. [arr0, arr1, arr2, ...]
            target_index (int or None): if None, defaults to last sensor.
            outlier_thresholds (list or None): If None, each adjacency uses
                                               self.default_outlier_threshold.
                                               If a list, must match len(sensor_list)-1.
            time_indexes (list of array-like or None): Only relevant if
                                                       method='seasonal_decompose' with 2 sensors,
                                                       or you want to pass them for each adjacency.
                                                       Typically you'd do something like:
                                                         [time_idx_0, time_idx_1, ...]
                                                       for each sensor.

        Returns:
            self
        """
        n = len(sensor_list)
        if n < 2:
            raise ValueError("Need at least two sensors to perform chain harmonization.")

        # Default target is last sensor
        if target_index is None:
            target_index = n - 1
        if not (0 <= target_index < n):
            raise ValueError(f"target_index {target_index} out of range [0..{n-1}]")

        self.target_index_ = target_index

        # Process outlier_thresholds
        if outlier_thresholds is None:
            outlier_thresholds = [self.default_outlier_threshold] * (n - 1)
        else:
            if len(outlier_thresholds) != n - 1:
                raise ValueError("Length of outlier_thresholds must be len(sensor_list)-1.")

        self.outlier_thresholds_ = outlier_thresholds

        # Initialize transforms: slope/intercept for each sensor -> target
        transforms = [None] * n
        transforms[target_index] = (1.0, 0.0)

        # Clear any previous pairwise data
        self.pairwise_left_ = []
        self.pairwise_right_ = []
        self.seasonal_map_a_ = None
        self.seasonal_map_b_ = None

        # Multi-sensor + seasonal_decompose?
        if self.method == 'seasonal_decompose' and n > 2:
            raise NotImplementedError(
                "Multi-sensor chaining with 'seasonal_decompose' is not supported in this example. "
                "The seasonal components for each adjacency would need special handling."
            )

        # ----- TWO-PASS BRIDGING -----

        # Left pass: from i=target_index down to i=1
        for i in range(target_index, 0, -1):
            # adjacency (i-1, i)
            out_thresh = outlier_thresholds[i-1]

            # For 2-sensor seasonal_decompose, we assume time_indexes is a list of length 2
            # or a single array if n=2. We'll pick the relevant time arrays if the user
            # provided them individually. If none provided, pass None.
            if time_indexes is None:
                t_index_a = None
                t_index_b = None
            else:
                if isinstance(time_indexes, list) and len(time_indexes) == n:
                    t_index_a = time_indexes[i-1]
                    t_index_b = time_indexes[i]
                elif isinstance(time_indexes, (list, np.ndarray)):
                    # Possibly a single array for 2 sensors total
                    t_index_a = time_indexes
                    t_index_b = time_indexes
                else:
                    t_index_a = None
                    t_index_b = None

            # We only need time_index for the pair (i-1, i). We'll just pass t_index_a if method=seasonal_decompose
            # In a real system, you'd confirm alignment if t_index_a != t_index_b. For simplicity, assume they're same.
            coef, intercept = self._harmonize_two_sensors(
                sensor_list[i-1],
                sensor_list[i],
                outlier_threshold=out_thresh,
                time_index=t_index_a
            )
            self.pairwise_left_.append(((i-1, i), (coef, intercept)))

            slope_i, inter_i = transforms[i]
            slope_new = coef * slope_i
            inter_new = coef * inter_i + intercept
            transforms[i-1] = (slope_new, inter_new)

        # Right pass: from i=target_index up to i=n-2
        for i in range(target_index, n - 1):
            out_thresh = outlier_thresholds[i]

            if time_indexes is None:
                t_index_a = None
                t_index_b = None
            else:
                if isinstance(time_indexes, list) and len(time_indexes) == n:
                    t_index_a = time_indexes[i]
                    t_index_b = time_indexes[i+1]
                elif isinstance(time_indexes, (list, np.ndarray)):
                    t_index_a = time_indexes
                    t_index_b = time_indexes
                else:
                    t_index_a = None
                    t_index_b = None

            coef, intercept = self._harmonize_two_sensors(
                sensor_list[i],
                sensor_list[i+1],
                outlier_threshold=out_thresh,
                time_index=t_index_a
            )
            self.pairwise_right_.append(((i, i+1), (coef, intercept)))

            slope_i, inter_i = transforms[i]
            slope_new = slope_i * coef
            inter_new = coef * inter_i + intercept
            transforms[i+1] = (slope_new, inter_new)

        self.transforms_ = transforms

        return self

    def transform(self, sensor_index, x, t=None):
        """
        Transform a new data point (or array) x from sensor 'sensor_index'
        onto the target sensor's scale.

        Args:
            sensor_index (int): index of the sensor in the original sensor_list.
            x (float or array-like): new reading(s) from sensor_index.
            t (optional): time index, if method='seasonal_decompose' and n=2 sensors.

        Returns:
            float or np.ndarray
        """
        if self.transforms_ is None:
            raise RuntimeError("Harmonizer must be fit before calling transform().")

        slope, intercept = self.transforms_[sensor_index]

        # Linear method => just apply slope/intercept
        if self.method == 'linear':
            if isinstance(x, (list, np.ndarray)):
                return slope * np.array(x) + intercept
            else:
                return slope * x + intercept

        # Seasonal method => only valid if n=2
        elif self.method == 'seasonal_decompose':
            # If there's more than 2 sensors, we already raise an error in fit().
            # So here we assume we have exactly 2 sensors in the list.
            if t is None:
                raise ValueError("Time index 't' must be provided for seasonal_decompose transform.")

            # If sensor_index is the target, slope=1, intercept=0 => no transform needed
            # but let's just do the generic approach anyway.

            # single value
            if not isinstance(x, (list, np.ndarray)):
                # Distinguish which map to use for the seasonal component?
                # If sensor_index = 0, we use self.seasonal_map_a_ for A
                # If sensor_index = 1, we use self.seasonal_map_b_ for B
                # But in your original code, you only stored the seasonal for "A->B" scenario
                # We'll define that sensor 0 is A, sensor 1 is B in the 2-sensor scenario
                # If sensor_index != 0, no transform needed. This is a design choice:
                # We'll interpret "A" as the non-target sensor, "B" as the target.
                # If sensor_index == target_index, the data is already in B-scale.

                if sensor_index == self.target_index_:
                    return x  # no transform needed
                else:
                    sea_a = self.seasonal_map_a_.get(t, 0.0)
                    sea_b = self.seasonal_map_b_.get(t, 0.0)
                    x_deseasoned = x - sea_a
                    return slope * x_deseasoned + intercept + sea_b

            # array-like
            else:
                x_arr = np.array(x)
                out = []
                for xi, ti in zip(x_arr, t):
                    if sensor_index == self.target_index_:
                        out.append(xi)
                    else:
                        sea_a = self.seasonal_map_a_.get(ti, 0.0)
                        sea_b = self.seasonal_map_b_.get(ti, 0.0)
                        xi_deseasoned = xi - sea_a
                        y_hat = slope * xi_deseasoned + intercept + sea_b
                        out.append(y_hat)
                return np.array(out)

        else:
            raise ValueError(f"Unknown method='{self.method}'.")
