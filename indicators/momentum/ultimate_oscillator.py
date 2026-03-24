import numpy as np


def ultimate_oscillator(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    p1: int = 7,
    p2: int = 14,
    p3: int = 28,
) -> np.ndarray:
    """Ultimate Oscillator (Larry Williams).

    Uses buying pressure (BP) and true range (TR) across three timeframes.
    BP = Close - min(Low, Prior Close)
    TR = max(High, Prior Close) - min(Low, Prior Close)
    Avg_n = sum(BP, n) / sum(TR, n)
    UO = 100 * (4*Avg1 + 2*Avg2 + Avg3) / 7

    Range [0, 100].
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    if n <= p3:
        return np.full(n, np.nan)

    # Buying Pressure and True Range (start from index 1)
    prior_close = closes[:-1]
    cur_close = closes[1:]
    cur_high = highs[1:]
    cur_low = lows[1:]

    bp = cur_close - np.minimum(cur_low, prior_close)
    tr = np.maximum(cur_high, prior_close) - np.minimum(cur_low, prior_close)

    result = np.full(n, np.nan)

    for i in range(p3, len(bp) + 1):
        bp_sum1 = bp[i - p1:i].sum()
        tr_sum1 = tr[i - p1:i].sum()
        bp_sum2 = bp[i - p2:i].sum()
        tr_sum2 = tr[i - p2:i].sum()
        bp_sum3 = bp[i - p3:i].sum()
        tr_sum3 = tr[i - p3:i].sum()

        if tr_sum1 == 0 or tr_sum2 == 0 or tr_sum3 == 0:
            result[i] = np.nan
            continue

        avg1 = bp_sum1 / tr_sum1
        avg2 = bp_sum2 / tr_sum2
        avg3 = bp_sum3 / tr_sum3

        result[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0

    return result
