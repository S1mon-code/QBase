import numpy as np


def linear_regression(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Linear Regression line value (end-point of least-squares fit).

    For each bar, fits y = a + b*x to the most recent *period* data points
    and returns the fitted value at the rightmost point (i.e. the regression
    line end value).  First period-1 values are np.nan.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    # x = 0, 1, ..., period-1
    x = np.arange(period, dtype=np.float64)
    x_mean = (period - 1) / 2.0
    ss_xx = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        y = data[i - period + 1 : i + 1]
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        out[i] = intercept + slope * (period - 1)  # end-point value

    return out


def linear_regression_slope(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Slope of the linear regression line over a rolling window.

    Positive slope indicates an uptrend, negative indicates a downtrend.
    First period-1 values are np.nan.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    x = np.arange(period, dtype=np.float64)
    x_mean = (period - 1) / 2.0
    ss_xx = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        y = data[i - period + 1 : i + 1]
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        out[i] = ss_xy / ss_xx

    return out


def r_squared(data: np.ndarray, period: int = 14) -> np.ndarray:
    """R-squared (coefficient of determination) of a rolling linear regression.

    Measures goodness of fit: 1.0 = perfect linear trend, 0.0 = no linear
    relationship.  First period-1 values are np.nan.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    x = np.arange(period, dtype=np.float64)
    x_mean = (period - 1) / 2.0
    ss_xx = np.sum((x - x_mean) ** 2)

    for i in range(period - 1, n):
        y = data[i - period + 1 : i + 1]
        y_mean = np.mean(y)
        ss_yy = np.sum((y - y_mean) ** 2)
        if ss_yy == 0.0:
            out[i] = 1.0  # constant data is a perfect "fit"
            continue
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        out[i] = (ss_xy ** 2) / (ss_xx * ss_yy)

    return out
