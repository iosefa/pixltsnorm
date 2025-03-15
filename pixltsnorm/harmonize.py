"""
harmonize.py

Generic functions to:
1) Filter outliers + fit a linear model for bridging (sensor A -> sensor B),
2) Chain multiple sensors so sensor0 -> sensorN via intermediate overlaps.
"""

from .outlier_filter import filter_outliers
from .regression import fit_regression, apply_regression

def harmonize_series(sensor_a_values, sensor_b_values, outlier_threshold=0.2):
    """
    Harmonize 'sensor_a_values' onto 'sensor_b_values' scale by:
      1) Filtering outliers
      2) Fitting a linear model: b = slope*a + intercept
      3) Returning slope, intercept, and a transform function.

    Args:
        sensor_a_values (array-like): data from sensor A
        sensor_b_values (array-like): data from sensor B
        outlier_threshold (float): difference threshold for outlier detection

    Returns:
        dict with:
            'coef'        -> float (slope)
            'intercept'   -> float
            'filtered_a'  -> np.ndarray
            'filtered_b'  -> np.ndarray
            'transform'   -> function(x_array) => slope*x_array + intercept
    """
    # 1. Filter outliers
    a_filt, b_filt = filter_outliers(sensor_a_values, sensor_b_values, threshold=outlier_threshold)

    # 2. Fit linear regression
    model = fit_regression(a_filt, b_filt)

    # 3. Construct transform function
    def transform_func(x):
        return apply_regression(x, model)

    return {
        'coef': model['coef'],
        'intercept': model['intercept'],
        'filtered_a': a_filt,
        'filtered_b': b_filt,
        'transform': transform_func
    }


def chain_harmonization(sensor_list, outlier_thresholds=None):
    """
    Chain an arbitrary number of sensors so that sensor0 -> sensorN
    is derived by successive pairwise transformations.

    Example:
      sensor0 -> sensor1
      sensor1 -> sensor2
      sensor2 -> sensor3
      ...
      sensor(N-1) -> sensorN

    We'll collect pairwise (slope, intercept) for each adjacency, then
    combine them so that sensor0 -> sensorN is a single linear transform:
        final_slope, final_intercept

    Args:
        sensor_list (list of array-like): e.g. [sensor0, sensor1, sensor2, ... sensorN]
        outlier_thresholds (list or None):
          - If provided, should have length = len(sensor_list)-1, giving a threshold
            for each adjacent pair.
          - If None, we use a default (0.2) for all pairs.

    Returns:
        dict with:
            'pairwise' (list of dict): each entry is the result from harmonize_series
            'final_slope': float, overall sensor0->sensorN slope
            'final_intercept': float, overall sensor0->sensorN intercept
    """
    n = len(sensor_list)
    if n < 2:
        raise ValueError("Need at least two sensors to chain.")

    if outlier_thresholds is None:
        outlier_thresholds = [0.2] * (n - 1)
    else:
        if len(outlier_thresholds) != (n - 1):
            raise ValueError(
                "Length of outlier_thresholds must be exactly len(sensor_list)-1."
            )

    # We'll store each pairwise transform result in pairwise_results
    pairwise_results = []

    # We'll also keep track of the cumulative slope/intercept
    # that transforms sensor0 -> sensor i
    cumulative_slope = 1.0
    cumulative_intercept = 0.0

    current_reference = sensor_list[0]  # sensor0 data

    for i in range(n - 1):
        sensor_a = current_reference
        sensor_b = sensor_list[i + 1]

        # 1) Harm. sensor_a -> sensor_b
        pair_result = harmonize_series(
            sensor_a, sensor_b,
            outlier_threshold=outlier_thresholds[i]
        )
        a_i = pair_result['coef']
        b_i = pair_result['intercept']

        pairwise_results.append(pair_result)

        # 2) Update cumulative slope/intercept
        # If new_slope * old_slope, new_slope * old_intercept + new_intercept
        # since sensor_b = a_i*( sensor_a ) + b_i
        # and sensor_a is ( cumulative_slope * sensor0 + cumulative_intercept )
        # => sensor_b = a_i * (cum_slope*sensor0 + cum_int) + b_i
        # = (a_i*cum_slope)*sensor0 + (a_i*cum_int + b_i)
        new_slope = cumulative_slope * a_i
        new_intercept = a_i * cumulative_intercept + b_i

        cumulative_slope = new_slope
        cumulative_intercept = new_intercept

        # Now if we want to "walk" forward so that the new sensor is the reference,
        # we can do so, but typically for bridging we just need the final transform
        # from sensor0 -> sensorN. If the user wants intermediate transformations,
        # we already store them in pairwise_results.

        # Optionally, if you want each step to re-base the "current_reference",
        # you could apply pair_result['transform'](current_reference).
        # But it's not strictly needed to compute the final slope/intercept.

    return {
        'pairwise': pairwise_results,
        'final_slope': cumulative_slope,
        'final_intercept': cumulative_intercept
    }
