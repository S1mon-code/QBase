import numpy as np


def nr7(
    highs: np.ndarray,
    lows: np.ndarray,
) -> np.ndarray:
    """Narrow Range 7: True when current bar range is the smallest of the last 7 bars.

    Returns a boolean array. First 6 values are False (insufficient lookback).
    """
    return _narrow_range(highs, lows, period=7)


def nr4(
    highs: np.ndarray,
    lows: np.ndarray,
) -> np.ndarray:
    """Narrow Range 4: True when current bar range is the smallest of the last 4 bars.

    Returns a boolean array. First 3 values are False (insufficient lookback).
    """
    return _narrow_range(highs, lows, period=4)


def _narrow_range(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int,
) -> np.ndarray:
    """Core narrow range detection for any lookback period."""
    n = len(highs)
    if n == 0:
        return np.array([], dtype=bool)

    out = np.full(n, False, dtype=bool)
    ranges = highs - lows

    for i in range(period - 1, n):
        window = ranges[i - period + 1 : i + 1]
        current = ranges[i]
        # Current range must be strictly the smallest (less than all others)
        if current > 0 and current < np.min(window[:-1]):
            out[i] = True

    return out
