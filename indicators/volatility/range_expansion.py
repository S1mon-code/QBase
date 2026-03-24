import numpy as np


def range_expansion(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Range Expansion Index: ratio of current bar range to average range.

    Returns current_range / avg_range_of_previous_bars for each bar.
    Values > 1.0 indicate expansion; < 1.0 indicate contraction.
    First `period` values are np.nan (insufficient lookback).
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)
    ranges = highs - lows

    for i in range(period, n):
        avg_range = np.mean(ranges[i - period : i])
        if avg_range > 1e-12:
            out[i] = ranges[i] / avg_range
        else:
            out[i] = 0.0

    return out
