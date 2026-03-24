import numpy as np


def dpo(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Detrended Price Oscillator.

    DPO = Close[shift bars ago] - SMA(period)
    where shift = period // 2 + 1.

    Removes trend to isolate cycles. No fixed range.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    shift = period // 2 + 1
    if n < period + shift:
        return np.full(n, np.nan)

    # Compute rolling SMA of length `period`
    cumsum = np.cumsum(closes)
    sma = np.full(n, np.nan)
    sma[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))) / period

    # DPO compares price from `shift` bars ago to the current SMA
    # Equivalently: DPO[i] = closes[i - shift] - sma[i]
    result = np.full(n, np.nan)
    for i in range(period - 1 + shift, n):
        result[i] = closes[i - shift] - sma[i]

    return result
