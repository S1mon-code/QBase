import numpy as np


def cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        period: int = 20) -> np.ndarray:
    """Commodity Channel Index: (TP - SMA(TP)) / (0.015 * mean_deviation)."""
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n < period:
        return np.full(n, np.nan)

    tp = (highs + lows + closes) / 3.0
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        seg = tp[i - period + 1:i + 1]
        mean_tp = np.mean(seg)
        mean_dev = np.mean(np.abs(seg - mean_tp))
        if mean_dev == 0:
            result[i] = 0.0
        else:
            result[i] = (tp[i] - mean_tp) / (0.015 * mean_dev)

    return result
