import numpy as np


def donchian(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian Channel.

    Returns (upper, lower, middle) where:
      upper  = highest high over the previous `period` bars (excl. current)
      lower  = lowest low over the previous `period` bars (excl. current)
      middle = (upper + lower) / 2

    First `period` values are np.nan (insufficient lookback).
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        upper[i] = np.max(highs[i - period:i])
        lower[i] = np.min(lows[i - period:i])
        middle[i] = (upper[i] + lower[i]) / 2.0

    return upper, lower, middle
