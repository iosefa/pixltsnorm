"""
models.py
=========

Provides core model-fitting functions for the ``pixltsnorm`` library, including
both **simple linear regression** and **seasonal decomposition** approaches for
time-series calibration:

1. **`fit_linear(...)`**
   - Applies a basic linear model: :math:`y = coef \cdot x + intercept`
   - Uses scikit-learn's :class:`LinearRegression` to fit slope and intercept.
   - Returns a dictionary with ``{'coef': float, 'intercept': float}``.

2. **`fit_seasonal(...)`**
   - Decomposes both ``x_values`` and ``y_values`` using statsmodels'
     :func:`seasonal_decompose` to extract seasonal components.
   - Regresses the residual (deseasonalized) y on the residual (deseasonalized) x.
   - Returns the fitted slope/intercept plus the extracted seasonal patterns.

**Typical Use Cases**:
- **Linear**: For straightforward sensor-to-sensor alignment or scale calibration.
- **Seasonal Decompose**: When time-series data has a known periodic cycle (like
  NDVI data with annual seasonality), you can remove this seasonal effect before
  fitting, then reapply the target sensor's seasonal pattern to the final transform.

These functions are invoked primarily by classes like
:class:`~pixltsnorm.harmonize.Harmonizer` or
:class:`~pixltsnorm.dataframe_harmonize.DataFrameHarmonizer`, but can be used directly
wherever you need either a simple linear or a seasonal decomposition–based calibration.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose


def fit_linear(x_values, y_values):
    """
    Fit a simple linear regression: y ~ coef*x + intercept
    Returns a dict { 'coef': float, 'intercept': float }
    """
    x = np.array(x_values).reshape(-1,1)
    y = np.array(y_values).reshape(-1,1)

    reg = LinearRegression().fit(x, y)
    return {
        'coef': reg.coef_[0][0],
        'intercept': reg.intercept_[0]
    }


def fit_seasonal(x_values, y_values, time_index, period):
    """
    Fit a 'seasonal_decompose' model for x->y:
      1) Decompose x, y => seasonal_x, seasonal_y
      2) Regress (y - seasonal_y) on (x - seasonal_x)
      3) Return { 'coef':..., 'intercept':..., 'seasonal_x':..., 'seasonal_y':... }

    You can store or return any additional info as needed.
    """
    # Decompose
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)

    dec_x = seasonal_decompose(x_arr, period=period, model='additive', extrapolate_trend='freq')
    dec_y = seasonal_decompose(y_arr, period=period, model='additive', extrapolate_trend='freq')

    seas_x = dec_x.seasonal
    seas_y = dec_y.seasonal

    x_deseason = x_arr - seas_x
    y_deseason = y_arr - seas_y

    # Fit linear on deseasoned
    linres = fit_linear(x_deseason, y_deseason)

    return {
        'coef': linres['coef'],
        'intercept': linres['intercept'],
        'seasonal_x': seas_x,
        'seasonal_y': seas_y
    }
