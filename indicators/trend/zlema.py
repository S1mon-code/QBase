import numpy as np


def zlema(data: np.ndarray, period: int) -> np.ndarray:
    """Zero-Lag Exponential Moving Average (ZLEMA).

    Removes the inherent lag of an EMA by pre-adjusting the input data:

      lag = (period - 1) / 2   (integer division)
      adjusted[i] = data[i] + (data[i] - data[i - lag])
      ZLEMA = EMA(adjusted, period)

    The idea is that a standard EMA applied to a straight line always lags
    by (period-1)/2 bars.  By adding in the recent momentum (the difference
    between current and lagged price), this lag is compensated.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    lag = (period - 1) // 2

    # Build lag-adjusted series
    adjusted = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i >= lag:
            adjusted[i] = data[i] + (data[i] - data[i - lag])
        else:
            adjusted[i] = data[i]  # not enough history, use raw

    # Apply EMA to adjusted series
    alpha = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = adjusted[0]
    for i in range(1, n):
        out[i] = alpha * adjusted[i] + (1.0 - alpha) * out[i - 1]

    return out
