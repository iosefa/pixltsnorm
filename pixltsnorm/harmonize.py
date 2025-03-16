"""
harmonize.py
============

Provides the core `Harmonizer` class for chaining and calibrating sensor data
using either simple linear transformations or seasonal decomposition. This class
supports multi-sensor time-series alignment by applying pairwise transformations
in a two-pass manner.

**Key Functionalities**:
- **Outlier Filtering**: Discard or ignore data points whose difference exceeds a
  specified threshold.
- **Linear Method**: Map each sensor’s scale to the target sensor with a slope-intercept fit.
- **Seasonal Decomposition** (2-sensor only): Remove a seasonal component from each
  sensor’s data, regress the residuals, then reapply the target sensor’s seasonal pattern.

**Usage**:
1. Instantiate a `Harmonizer` with a chosen method (e.g. 'linear').
2. Call `fit()` on a list of sensor data arrays, optionally specifying outlier thresholds
   and time indexes (for seasonal mode).
3. Transform new data by calling `transform(sensor_index, x, t=...)`.

**Example**::
    from pixltsnorm.harmonize import Harmonizer

    # Suppose we have two sensors (arrA, arrB), each is a 1D NumPy array
    harm = Harmonizer(method='linear', outlier_threshold=0.2)
    harm.fit([arrA, arrB], target_index=1)
    # Now transform new data from sensor A
    new_value_harmonized = harm.transform(sensor_index=0, x=some_value)

This module relies on separate utility and model functions:
- **`utils.filter_outliers`** for threshold-based outlier removal,
- **`models.fit_linear`** and **`models.fit_seasonal`** for different calibration approaches.
"""

import numpy as np
from .utils import filter_outliers
from .models import fit_linear, fit_seasonal


class Harmonizer:
    """
    Class for managing data transformations and outlier detections with configurable
    parameters and optional seasonal alignment. This class is designed to harmonize
    sensor data with flexible transformation methods (e.g., linear transformation or
    seasonal decomposition), and supports both two-sensor and multi-sensor setups.

    The Harmonizer can detect outliers, apply pairwise transformations between sensors,
    and handle seasonal adjustments using time indices and seasonal periods where
    applicable. It records transformation coefficients and seasonal components for
    future operations on new data.

    :ivar method: The method used for harmonization and interpolation.
    :type method: str
    :ivar period: Optional seasonal period definition for decomposition-based transformations.
    :type period: Optional
    :ivar default_outlier_threshold: Default value for outlier detection threshold.
    :type default_outlier_threshold: float
    :ivar outlier_thresholds_: Computed outlier thresholds associated with sensors after training.
    :type outlier_thresholds_: Optional
    :ivar transforms_: Transformation coefficients (slope, intercept) for each sensor-to-target mapping.
    :type transforms_: Optional
    :ivar target_index_: Index of the target sensor in the multi-sensor setup chain.
    :type target_index_: Optional
    :ivar pairwise_left_: Transformation information for the left-side pass (source-to-target).
    :type pairwise_left_: List
    :ivar pairwise_right_: Transformation information for the right-side pass (target-to-source).
    :type pairwise_right_: List
    :ivar seasonal_map_a_: Seasonal adjustments for sensor A in a two-sensor configuration.
    :type seasonal_map_a_: Optional
    :ivar seasonal_map_b_: Seasonal adjustments for sensor B in a two-sensor configuration.
    :type seasonal_map_b_: Optional
    """

    def __init__(self, method='linear', period=None, outlier_threshold=0.2):
        """
        Class for managing data transformations and outlier detections with configurable
        parameters and optional seasonal alignment. It supports initialization of core
        attributes including interpolation method, period, and an outlier threshold.

        :Attributes:
            - method (str): The method used for interpolation. Default is 'linear'.
            - period (Optional): The seasonal period when applicable.
            - default_outlier_threshold (float): The default threshold for outlier detection.
            - outlier_thresholds_ (Optional): Placeholder for outlier thresholds after fitting or transforming data.
            - transforms_ (Optional): Placeholder for transformation details.
            - target_index_ (Optional): Index of the target data.
            - pairwise_left_ (List): Coordinates for left pairwise computation.
            - pairwise_right_ (List): Coordinates for right pairwise computation.
            - seasonal_map_a_ (Optional): Seasonal mapping for sensor A in two-sensor approaches.
            - seasonal_map_b_ (Optional): Seasonal mapping for sensor B in two-sensor approaches.

        :param method: The interpolation method to use. Defaults to 'linear'.
        :type method: str
        :param period: An optional parameter defining the seasonal period. Defaults to None.
        :type period: Optional
        :param outlier_threshold: Threshold value for detecting outliers. Defaults to 0.2.
        :type outlier_threshold: float
        """
        self.method = method
        self.period = period
        self.default_outlier_threshold = outlier_threshold

        self.outlier_thresholds_ = None
        self.transforms_ = None
        self.target_index_ = None
        self.pairwise_left_ = []
        self.pairwise_right_ = []

        # For 2-sensor seasonal approach
        self.seasonal_map_a_ = None
        self.seasonal_map_b_ = None

    def _harmonize_two_sensors(self, sensor_a_vals, sensor_b_vals,
                               outlier_thresh, time_index=None):
        """
        Harmonizes two sets of sensor data by applying either a linear alignment or
        seasonal decomposition, based on the selected method. The function filters
        outliers in the input data and computes alignment parameters accordingly.

        This method supports different calibration strategies. For 'linear', a simple
        linear fit is applied to the filtered sensor data. If 'seasonal_decompose' is
        selected, the data is further aligned seasonally, using a given time index
        and period to compute decomposition results.

        :param sensor_a_vals: Values captured by the first sensor.
        :type sensor_a_vals: list or numpy.ndarray
        :param sensor_b_vals: Values captured by the second sensor.
        :type sensor_b_vals: list or numpy.ndarray
        :param outlier_thresh: Threshold value for filtering outliers. Data points
            with differences exceeding this threshold are considered outliers.
        :type outlier_thresh: float
        :param time_index: Optional time index associated with the input sensor data.
            Required if using the 'seasonal_decompose' method.
        :type time_index: list or numpy.ndarray or None
        :return: A tuple containing the fit coefficient and intercept of the selected
            calibration strategy.
        :rtype: tuple
        """
        # 1) filter outliers
        a_filt, b_filt = filter_outliers(sensor_a_vals, sensor_b_vals, threshold=outlier_thresh)

        if self.method == 'linear':
            # simple linear
            linres = fit_linear(a_filt, b_filt)
            return linres['coef'], linres['intercept']

        elif self.method == 'seasonal_decompose':
            # only valid if we have exactly 2 sensors & user gave time_index, period
            if time_index is None or self.period is None:
                raise ValueError("time_index and period required for seasonal_decompose.")

            # We must also filter out outliers in time sync. If you want to keep the same mask:
            # you can forcibly keep the same indices as a_filt,b_filt => you'd have to track them.
            # Or do your own approach. For brevity, we'll just do the same approach:
            # We'll pass a_filt,b_filt + time_index_inliers to fit_seasonal
            # by figuring out which original indices are in the 'a_filt,b_filt'.

            # We'll do a quick approach: we already have a_filt,b_filt. We must find which
            # indexes they correspond to in the original arrays. We'll assume the difference approach:
            arrA = np.array(sensor_a_vals)
            arrB = np.array(sensor_b_vals)
            mask = np.abs(arrB - arrA) <= outlier_thresh

            # subset time_index
            time_idx_inliers = np.array(time_index)[mask]

            # Now do seasonal fit
            seasres = fit_seasonal(a_filt, b_filt, time_idx_inliers, self.period)

            self.seasonal_map_a_ = dict(zip(time_idx_inliers, seasres['seasonal_x']))
            self.seasonal_map_b_ = dict(zip(time_idx_inliers, seasres['seasonal_y']))

            return seasres['coef'], seasres['intercept']

        else:
            raise ValueError(f"Unknown method='{self.method}'.")

    def fit(self, sensor_list, target_index=None, outlier_thresholds=None, time_indexes=None):
        """
        Fits the harmonization model for a chain of sensors. This function establishes
        a set of transformations to harmonize data from multiple sensors in a sequential
        manner, aligning them to the target sensor. Each sensor is corrected relative
        to its immediate neighbor, by calculating pairwise transformation coefficients.
        The fitting process applies both left and right passes to ensure alignment.

        The function also handles optional configurations such as outlier thresholds
        and specific time indexes for each pair of sensors. These configurations define
        how to identify outliers during harmonization and allow customization of the
        harmonization process for time-series data.

        :param sensor_list: List of sensors for harmonization. Each sensor represents
            an individual data source whose measurements will be aligned.
        :type sensor_list: list
        :param target_index: Index of the sensor that acts as the target (reference)
            for alignment. Defaults to the last sensor in the list.
        :type target_index: int, optional
        :param outlier_thresholds: Threshold values for detecting outliers in the data
            for each sensor pair. The length should be one less than the number of sensors
            if provided. Defaults to repeating a default threshold value.
        :type outlier_thresholds: list[float], optional
        :param time_indexes: Time index arrays corresponding to the data in each
            sensor, allowed as either a list of time arrays for each sensor or a single
            array that applies to all. Defaults to None.
        :type time_indexes: list[array-like] or array-like, optional
        :return: The current instance of the harmonization process after completing
            the fitting procedure with transformations initialized for each sensor.
        :rtype: self
        :raises ValueError: If the number of sensors is less than 2, if target_index
            is outside the valid range, or if the length of outlier_thresholds does not
            match expectations.
        :raises NotImplementedError: If using the 'seasonal_decompose' method for more
            than 2 sensors, as this is not yet supported.
        """
        n = len(sensor_list)
        if n<2:
            raise ValueError("Need >=2 sensors to chain harmonize.")
        if target_index is None:
            target_index = n-1
        if not (0 <= target_index < n):
            raise ValueError(f"target_index {target_index} out of range.")

        if self.method=='seasonal_decompose' and n>2:
            raise NotImplementedError("multi-sensor + seasonal_decompose not supported.")

        self.target_index_ = target_index

        # handle outlier_thresholds
        if outlier_thresholds is None:
            outlier_thresholds = [self.default_outlier_threshold]*(n-1)
        else:
            if len(outlier_thresholds)!= n-1:
                raise ValueError("wrong length for outlier_thresholds")

        self.outlier_thresholds_ = outlier_thresholds

        # init transforms => slope/intercept for each sensor->target
        transforms = [None]*n
        transforms[target_index] = (1.0, 0.0)

        self.pairwise_left_.clear()
        self.pairwise_right_.clear()
        self.seasonal_map_a_=None
        self.seasonal_map_b_=None

        # left pass
        for i in range(target_index,0,-1):
            out_thresh = outlier_thresholds[i-1]
            if time_indexes is None:
                t_idx_a=None
            else:
                # if time_indexes is a list of length n => pass time_indexes[i-1]
                # or if single array => pass it. We'll do minimal checks
                if isinstance(time_indexes,(list, np.ndarray)):
                    if len(time_indexes)==n:
                        t_idx_a=time_indexes[i-1]
                    else:
                        t_idx_a=time_indexes
                else:
                    t_idx_a=None

            coef, intercept = self._harmonize_two_sensors(
                sensor_list[i-1],
                sensor_list[i],
                outlier_thresh=out_thresh,
                time_index=t_idx_a
            )
            self.pairwise_left_.append(((i-1, i),(coef, intercept)))

            slope_i, inter_i = transforms[i]
            slope_new = coef*slope_i
            inter_new = coef*inter_i + intercept
            transforms[i-1]= (slope_new, inter_new)

        # right pass
        for i in range(target_index, n-1):
            out_thresh = outlier_thresholds[i]
            if time_indexes is None:
                t_idx_a=None
            else:
                if isinstance(time_indexes,(list, np.ndarray)):
                    if len(time_indexes)==n:
                        t_idx_a=time_indexes[i]
                    else:
                        t_idx_a=time_indexes
                else:
                    t_idx_a=None

            coef, intercept = self._harmonize_two_sensors(
                sensor_list[i],
                sensor_list[i+1],
                outlier_thresh=out_thresh,
                time_index=t_idx_a
            )
            self.pairwise_right_.append(((i, i+1),(coef, intercept)))

            slope_i, inter_i= transforms[i]
            slope_new = slope_i*coef
            inter_new = coef*inter_i + intercept
            transforms[i+1]= (slope_new, inter_new)

        self.transforms_= transforms
        return self

    def transform(self, sensor_index, x, t=None):
        """
        Transform input data using a pre-fitted transformation model.

        This method applies a transformation to the input data (x) based on a set of
        pre-fitted parameters and a selected transformation method. The supported
        methods include linear transformations and seasonal decomposition-based
        transformations. For seasonal decomposition, additional temporal information
        (t) is required.

        :param sensor_index: Index of the sensor for which the transformation should
            be applied.
        :type sensor_index: int
        :param x: Input data value(s) to be transformed. Can be a scalar or an array.
        :type x: Union[float, Sequence[float]]
        :param t: (Optional) Temporal index associated with the input data. Required
            when using the 'seasonal_decompose' method. Can be a scalar or an array.
        :type t: Optional[Union[float, Sequence[float]]]
        :return: Transformed data. The result has the same structure as the input
            data (scalar or array).
        :rtype: Union[float, np.ndarray]
        :raises RuntimeError: If the transformation model has not been fitted before
            calling this method.
        :raises ValueError: If the 'seasonal_decompose' method is chosen and the
            temporal index (t) is not provided, or if an unknown transformation method
            is specified.
        """
        if self.transforms_ is None:
            raise RuntimeError("must call fit first")

        slope, intercept= self.transforms_[sensor_index]

        if self.method=='linear':
            # normal
            return slope*np.array(x)+ intercept

        elif self.method=='seasonal_decompose':
            # only valid if n=2
            if t is None:
                raise ValueError("time index needed for seasonal_decompose transform")

            # if sensor_index== target => pass x back
            # else => x-> deseason => slope*(x-seasA)+ intercept + seasB
            out=[]
            x_arr=np.array(x,ndmin=1)  # handle scalar or array
            # handle t similarly
            t_arr=np.array(t,ndmin=1)

            for xi,ti in zip(x_arr,t_arr):
                if sensor_index== self.target_index_:
                    out.append(xi)
                else:
                    seasA= self.seasonal_map_a_.get(ti,0.0)
                    seasB= self.seasonal_map_b_.get(ti,0.0)
                    x_des= xi - seasA
                    y_hat= slope*x_des+ intercept+ seasB
                    out.append(y_hat)

            if len(x_arr)==1: # single value
                return out[0]
            return np.array(out)

        else:
            raise ValueError(f"unknown method {self.method}")
