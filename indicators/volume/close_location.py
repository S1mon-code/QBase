import numpy as np


def close_location(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """Close Location Value (CLV).

    Measures where the close falls within the bar's high-low range:
      CLV = ((close - low) - (high - close)) / (high - low)

    Equivalent to: (2 * close - high - low) / (high - low).
    Range is [-1, 1]. +1 = close at high, -1 = close at low.
    Returns 0.0 for zero-range (flat) bars.
    """
    if highs.size == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    bar_range = highs - lows
    out = np.where(
        bar_range > 0,
        (2.0 * closes - highs - lows) / bar_range,
        0.0,
    )

    return out.astype(np.float64)
