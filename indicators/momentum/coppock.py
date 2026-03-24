import numpy as np


def coppock(
    closes: np.ndarray,
    wma_period: int = 10,
    roc_long: int = 14,
    roc_short: int = 11,
) -> np.ndarray:
    """Coppock Curve.

    Coppock = WMA(ROC(roc_long) + ROC(roc_short), wma_period)

    Originally designed for monthly data. A buy signal occurs when
    the curve turns up from below zero. No fixed range.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    max_roc = max(roc_long, roc_short)
    warmup = max_roc + wma_period - 1
    if n <= warmup:
        return np.full(n, np.nan)

    # ROC = (close / close_n_ago - 1) * 100
    roc_l = np.full(n, np.nan)
    roc_l[roc_long:] = (closes[roc_long:] / closes[:-roc_long] - 1.0) * 100.0

    roc_s = np.full(n, np.nan)
    roc_s[roc_short:] = (closes[roc_short:] / closes[:-roc_short] - 1.0) * 100.0

    roc_sum = roc_l + roc_s  # nan propagates before max_roc

    # Weighted Moving Average of roc_sum
    result = np.full(n, np.nan)
    for i in range(max_roc + wma_period - 1, n):
        window = roc_sum[i - wma_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        weights = np.arange(1, wma_period + 1, dtype=float)
        result[i] = np.dot(window, weights) / weights.sum()

    return result
