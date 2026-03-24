import numpy as np


def _wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average (linearly weighted).

    Weight for position k (0-based from oldest) = k + 1.
    First period-1 values are np.nan.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    weights = np.arange(1, period + 1, dtype=np.float64)
    w_sum = weights.sum()

    for i in range(period - 1, n):
        out[i] = np.dot(data[i - period + 1 : i + 1], weights) / w_sum
    return out


def hma(data: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average (Alan Hull).

    Formula:
      HMA(n) = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )

    Steps:
      1. Compute WMA with period n/2 (rounded).
      2. Compute WMA with period n.
      3. Difference series = 2 * step1 - step2.
      4. Apply WMA with period sqrt(n) (rounded) to the difference series.

    Produces a much faster-responding average with significantly less lag
    than a standard WMA or SMA of the same period.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    half_period = max(int(round(period / 2)), 1)
    sqrt_period = max(int(round(np.sqrt(period))), 1)

    wma_half = _wma(data, half_period)
    wma_full = _wma(data, period)

    # Difference series (NaN where either input is NaN)
    diff = 2.0 * wma_half - wma_full

    # Find first valid index in diff
    first_valid = -1
    for i in range(n):
        if not np.isnan(diff[i]):
            first_valid = i
            break

    if first_valid < 0:
        return np.full(n, np.nan, dtype=np.float64)

    # Apply WMA(sqrt_period) only to the valid portion of diff
    valid_data = diff[first_valid:]
    wma_result = _wma(valid_data, sqrt_period)

    out = np.full(n, np.nan, dtype=np.float64)
    out[first_valid:] = wma_result
    return out
