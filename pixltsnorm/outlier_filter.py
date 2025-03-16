import numpy as np

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