import pytest
import numpy as np
from pixltsnorm.harmonize import Harmonizer


def test_harmonizer_linear_basic():
    """
    Test a basic two-sensor linear fit with random data.
    """
    # Generate two simple correlated arrays
    np.random.seed(42)
    sensor_a = np.linspace(0, 1, 10) + np.random.normal(0, 0.01, 10)
    sensor_b = 2.0 * sensor_a + 0.05  # known slope=2.0, intercept=0.05

    harm = Harmonizer(method="linear", outlier_threshold=0.2)
    harm.fit([sensor_a, sensor_b], target_index=1)

    # We expect transforms_[0] to map sensor A -> B
    slope, intercept = harm.transforms_[0]
    # Tolerances can be adjusted
    assert pytest.approx(slope, 0.1) == 2.0
    assert pytest.approx(intercept, 0.1) == 0.05


def test_harmonizer_seasonal_exception():
    """
    If method='seasonal_decompose' but more than 2 sensors,
    we expect NotImplementedError from this function.
    """
    sensor_list = [np.arange(5), np.arange(5), np.arange(5)]
    harm = Harmonizer(method="seasonal_decompose", outlier_threshold=0.2)
    with pytest.raises(NotImplementedError):
        harm.fit(sensor_list, target_index=2)


def test_harmonizer_transform():
    """
    Check that 'transform()' is consistent with the fit we did.
    """
    # sensor A's data
    data_a = np.array([0.1, 0.2, 0.3])
    # sensor B is 2.0*A + 0.05
    data_b = 2.0 * data_a + 0.05

    harm = Harmonizer(method="linear", outlier_threshold=1.0)
    harm.fit([data_a, data_b], target_index=1)
    # transform sensor A's 0.2 -> ?
    mapped_val = harm.transform(sensor_index=0, x=0.2)
    # expected = 2.0*0.2 + 0.05 = 0.45
    assert pytest.approx(mapped_val, 0.01) == 0.45
