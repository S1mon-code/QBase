import numpy as np


def vwma(closes: np.ndarray, volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume Weighted Moving Average.

    Formula: VWMA = sum(close * volume, period) / sum(volume, period)

    Gives more weight to bars with higher volume, making it more responsive
    to price moves backed by strong participation.
    First period-1 values are np.nan.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    cv = closes * volumes
    cum_cv = np.cumsum(cv)
    cum_v = np.cumsum(volumes)

    out[period - 1] = cum_cv[period - 1] / cum_v[period - 1]
    if n > period:
        out[period:] = (cum_cv[period:] - cum_cv[:-period]) / (
            cum_v[period:] - cum_v[:-period]
        )

    return out
