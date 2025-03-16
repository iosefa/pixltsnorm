import pytest
import numpy as np
from pixltsnorm.models import fit_linear, fit_seasonal


def test_fit_linear():
    # Create data with known slope=2.0, intercept=0.3
    x = np.linspace(0, 1, 20)
    y = 2.0 * x + 0.3
    res = fit_linear(x, y)
    assert pytest.approx(res["coef"], 0.05) == 2.0
    assert pytest.approx(res["intercept"], 0.05) == 0.3


def test_fit_seasonal_basic():
    """
    Basic check for fit_seasonal. We'll skip the real decomposition
    and just ensure the structure is correct.
    This function generally expects real timeseries + period.
    """
    xvals = np.array([0.2, 0.3, 0.5, 0.7])
    yvals = 1.5 * xvals + 0.1
    time_index = np.array([0, 1, 2, 3])  # or real dates
    period = 2

    out = fit_seasonal(xvals, yvals, time_index, period)
    # We expect certain keys:
    assert all(k in out for k in ["coef", "intercept", "seasonal_x", "seasonal_y"])
    # Because real decomposition + regression is domain-specific,
    # we won't check exact numeric output for a toy example.
