import numpy as np


def rolling_std(
    data: np.ndarray,
    period: int = 20,
    ddof: int = 1,
) -> np.ndarray:
    """Rolling Standard Deviation.

    Computes the sample standard deviation over a rolling window.

    Formula:
        std = sqrt( sum((x_i - mean)^2) / (n - ddof) )

    Reference: Fundamental statistical measure used across all quant finance.
    """
    n = len(data)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        out[i] = np.std(window, ddof=ddof)

    return out


def z_score(
    data: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Rolling Z-Score.

    Measures how many standard deviations the current value is from the
    rolling mean.

    Formula:
        z = (x - mean(x, period)) / std(x, period)

    Values > 2 or < -2 suggest statistical extremes.
    """
    n = len(data)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        mean = np.mean(window)
        std = np.std(window, ddof=1)
        if std != 0:
            out[i] = (data[i] - mean) / std

    return out
