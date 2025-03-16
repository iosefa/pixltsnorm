import numpy as np
from sklearn.linear_model import LinearRegression


def fit_regression(x_values, y_values):
    """
    Fit a linear regression model: y = coef*x + intercept.

    Args:
        x_values (array-like): data values from sensor A (or any numeric array)
        y_values (array-like): data values from sensor B (or any numeric array)

    Returns:
        dict with {'coef': float, 'intercept': float, 'model': scikit-learn object}
    """
    x = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values).reshape(-1, 1)

    reg = LinearRegression().fit(x, y)

    return {
        'coef': reg.coef_[0][0],
        'intercept': reg.intercept_[0],
        'model': reg
    }


def apply_regression(x_values, model):
    """
    Apply a pre-fit linear model to an array of x-values.

    Args:
        x_values (array-like): numeric data (e.g. sensor A)
        model (dict): must have 'coef' and 'intercept' keys

    Returns:
        np.ndarray of predicted values
    """
    slope = model['coef']
    intercept = model['intercept']
    x_array = np.array(x_values)
    return slope * x_array + intercept