import numpy as np


def williams_r(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               period: int = 14) -> np.ndarray:
    """Williams %R. Range: -100 (oversold) to 0 (overbought)."""
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < period:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        hh = np.max(highs[i - period + 1:i + 1])
        ll = np.min(lows[i - period + 1:i + 1])
        if hh == ll:
            result[i] = -50.0
        else:
            result[i] = -100.0 * (hh - closes[i]) / (hh - ll)

    return result
