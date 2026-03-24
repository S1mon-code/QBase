import numpy as np


def volume_oscillator(
    volumes: np.ndarray,
    fast: int = 5,
    slow: int = 20,
) -> np.ndarray:
    """Volume Oscillator.

    Percentage difference between a fast and slow SMA of volume:
      VO = (SMA(Volume, fast) - SMA(Volume, slow)) / SMA(Volume, slow) * 100

    First ``slow - 1`` values are np.nan.

    Source: General technical analysis reference.
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=np.float64)

    volumes = volumes.astype(np.float64)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < slow:
        return result

    # Compute fast SMA
    fast_sma = np.full(n, np.nan, dtype=np.float64)
    running = np.sum(volumes[:fast])
    fast_sma[fast - 1] = running / fast
    for i in range(fast, n):
        running += volumes[i] - volumes[i - fast]
        fast_sma[i] = running / fast

    # Compute slow SMA
    slow_sma = np.full(n, np.nan, dtype=np.float64)
    running = np.sum(volumes[:slow])
    slow_sma[slow - 1] = running / slow
    for i in range(slow, n):
        running += volumes[i] - volumes[i - slow]
        slow_sma[i] = running / slow

    # VO starts where both SMAs are valid (slow - 1)
    for i in range(slow - 1, n):
        if slow_sma[i] != 0.0:
            result[i] = (fast_sma[i] - slow_sma[i]) / slow_sma[i] * 100.0
        else:
            result[i] = 0.0

    return result
