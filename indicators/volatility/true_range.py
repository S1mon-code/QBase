import numpy as np


def true_range(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """True Range (standalone).

    The True Range accounts for gaps between sessions by considering
    the previous close.

    Formula:
        TR = max(H - L, |H - prevClose|, |L - prevClose|)

    The first bar uses H - L (no previous close available).

    Reference: J. Welles Wilder Jr. (1978), "New Concepts in Technical
    Trading Systems." Building block for ATR, Chandelier Exit, etc.
    """
    n = len(closes)
    if n == 0:
        return np.full(n, np.nan)

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    return tr
