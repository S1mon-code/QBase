import numpy as np


def ao(highs: np.ndarray, lows: np.ndarray, fast: int = 5, slow: int = 34) -> np.ndarray:
    """Awesome Oscillator (Bill Williams).

    AO = SMA(midpoint, fast) - SMA(midpoint, slow)
    where midpoint = (High + Low) / 2.

    Oscillates around zero; no fixed range.
    """
    if highs.size == 0:
        return np.array([], dtype=float)
    n = highs.size
    if n < slow:
        return np.full(n, np.nan)

    midpoint = (highs + lows) / 2.0

    # SMA via cumsum trick
    cumsum = np.cumsum(midpoint)

    sma_fast = np.full(n, np.nan)
    sma_fast[fast - 1:] = (cumsum[fast - 1:] - np.concatenate(([0.0], cumsum[:-fast]))) / fast

    sma_slow = np.full(n, np.nan)
    sma_slow[slow - 1:] = (cumsum[slow - 1:] - np.concatenate(([0.0], cumsum[:-slow]))) / slow

    result = sma_fast - sma_slow
    return result
