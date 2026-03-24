import numpy as np


def mfi(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Money Flow Index.

    RSI-style oscillator (0-100) that incorporates both price and volume.
    First `period` values are np.nan.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    tp = (highs + lows + closes) / 3.0
    raw_mf = tp * volumes

    # Determine direction: positive if TP > prev TP, negative if TP < prev TP
    # First bar has no direction — treated as zero flow
    tp_diff = np.zeros(n, dtype=np.float64)
    tp_diff[1:] = np.diff(tp)

    pos_flow = np.where(tp_diff > 0, raw_mf, 0.0)
    neg_flow = np.where(tp_diff < 0, raw_mf, 0.0)

    result = np.full(n, np.nan, dtype=np.float64)

    # Rolling sums over `period` bars (indices 1..period correspond to first window)
    pos_sum = np.sum(pos_flow[1 : period + 1])
    neg_sum = np.sum(neg_flow[1 : period + 1])

    if n > period:
        if neg_sum == 0.0:
            result[period] = 100.0
        else:
            result[period] = 100.0 - 100.0 / (1.0 + pos_sum / neg_sum)

    for i in range(period + 1, n):
        pos_sum += pos_flow[i] - pos_flow[i - period]
        neg_sum += neg_flow[i] - neg_flow[i - period]

        if neg_sum == 0.0:
            result[i] = 100.0
        else:
            result[i] = 100.0 - 100.0 / (1.0 + pos_sum / neg_sum)

    return result
