import numpy as np


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.

    First period-1 values are np.nan. Each subsequent value is the
    arithmetic mean of the preceding `period` elements.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    # Cumulative sum trick for O(n) computation
    cumsum = np.cumsum(data)
    out[period - 1] = cumsum[period - 1] / period
    out[period:] = (cumsum[period:] - cumsum[:-period]) / period

    return out
