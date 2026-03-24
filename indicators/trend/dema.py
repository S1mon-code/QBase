import numpy as np


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Internal EMA helper."""
    n = len(data)
    alpha = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def dema(data: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average.

    Formula: DEMA = 2 * EMA(data, period) - EMA(EMA(data, period), period)

    Reduces lag compared to a standard EMA by subtracting the smoothed EMA
    from twice the original EMA.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    ema1 = _ema(data, period)
    ema2 = _ema(ema1, period)
    return 2.0 * ema1 - ema2
