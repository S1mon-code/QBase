import numpy as np


def rate_of_change(closes: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change: (close / close_n_periods_ago - 1) * 100."""
    if closes.size == 0:
        return np.array([], dtype=float)
    if closes.size <= period:
        return np.full(closes.size, np.nan)

    roc = np.full(closes.size, np.nan)
    roc[period:] = (closes[period:] / closes[:-period] - 1.0) * 100.0
    return roc
