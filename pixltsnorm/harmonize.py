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
            'filtered_a'  -> np.ndarray (A after outlier filtering)
            'filtered_b'  -> np.ndarray (B after outlier filtering)
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


def chain_harmonization(sensor_list, target_index=None, outlier_thresholds=None):
    """
    A two-pass bridging approach mapping each array in `sensor_list` onto
    the scale of sensor_list[target_index] (or onto the last sensor if
    no target_index is provided).

    This parallels the logic of the 'global_harmonize.chain_global_bridging',
    but works purely on arrays (not DataFrames).

    Steps:
      1) If target_index is None, default to the last sensor in sensor_list.
      2) Initialize transforms[target_index] = (1.0, 0.0).
      3) Left pass: from i=target_index down to i=1
         - bridging (i-1 -> i) in forward direction
         - chain with i->target => (i-1)->target
      4) Right pass: from i=target_index up to i=n-2
         - bridging (i -> i+1) => (i+1)->target
      5) Return a dict with the final transforms and adjacency bridging results:
         {
           "transforms": [ (slope_i, intercept_i), ... ] for i in [0..n-1],
           "pairwise_left":  [ ( (i-1,i), bridging_res ), ... ],
           "pairwise_right": [ ( (j,j+1), bridging_res ), ... ]
         }

    Note: This function doesn't return "harmonized arrays" by default — you can
    apply them yourself if desired, or you can add that feature easily.

    Args:
        sensor_list (list of array-like):
            e.g. [arr0, arr1, arr2, ...].
        target_index (int or None):
            Index in [0..len(sensor_list)-1]. If None, default to the last sensor.
        outlier_thresholds (list or None):
            If None, each adjacency uses 0.2.
            If a list, must be length == len(sensor_list)-1.

    Returns:
        dict:
          {
            "transforms":    list of (slope, intercept), each for sensor[i] -> target,
            "pairwise_left":  list of ((i-1, i), bridging_result),
            "pairwise_right": list of ((i, i+1), bridging_result)
          }

    Raises:
        ValueError:
          - if n<2
          - if outlier_thresholds is the wrong length
          - if bridging has no valid overlap (i.e. all outliers, but that would typically
            be discovered earlier).
    """
    n = len(sensor_list)
    if n < 2:
        raise ValueError("Need at least two sensors in sensor_list for chain_harmonization.")

    # If target not specified, use the last sensor
    if target_index is None:
        target_index = n - 1

    if not (0 <= target_index < n):
        raise ValueError(f"target_index {target_index} out of range [0..{n-1}]")

    # Handle outlier thresholds
    if outlier_thresholds is None:
        outlier_thresholds = [0.2] * (n - 1)
    else:
        if len(outlier_thresholds) != (n - 1):
            raise ValueError("Length of outlier_thresholds must be len(sensor_list)-1")

    # We'll store (slope, intercept) for each sensor -> target
    transforms = [None]*n
    transforms[target_index] = (1.0, 0.0)

    pairwise_left = []
    pairwise_right = []

    # LEFT pass: from i=target_index down to i=1
    for i in range(target_index, 0, -1):
        # bridging (i-1) -> (i)
        bridging_res = harmonize_series(
            sensor_list[i-1],
            sensor_list[i],
            outlier_threshold=outlier_thresholds[i-1]  # adjacency i-1
        )
        pairwise_left.append(((i-1, i), bridging_res))

        # c, d from bridging => y = c*x + d
        a_i = bridging_res["coef"]
        b_i = bridging_res["intercept"]

        slope_i, inter_i = transforms[i]  # i->target
        slope_new = a_i * slope_i
        inter_new = a_i * inter_i + b_i
        transforms[i-1] = (slope_new, inter_new)

    # RIGHT pass: from i=target_index up to i=n-2
    for i in range(target_index, n-1):
        bridging_res = harmonize_series(
            sensor_list[i],
            sensor_list[i+1],
            outlier_threshold=outlier_thresholds[i]
        )
        pairwise_right.append(((i, i+1), bridging_res))

        a_i = bridging_res["coef"]
        b_i = bridging_res["intercept"]

        slope_i, inter_i = transforms[i]  # i->target
        slope_new = slope_i * a_i
        inter_new = a_i*inter_i + b_i
        transforms[i+1] = (slope_new, inter_new)

    return {
        "transforms":     transforms,
        "pairwise_left":  pairwise_left,
        "pairwise_right": pairwise_right
    }
