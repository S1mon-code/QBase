import numpy as np


def entropy(
    closes: np.ndarray,
    period: int = 20,
    bins: int = 10,
) -> np.ndarray:
    """Rolling Shannon entropy (base-2) of log return distribution.

    Returns are discretised into `bins` equal-width bins over each
    rolling window of `period` bars.  Higher entropy = more random/uncertain.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    # Need period+1 prices for period returns
    if n < period + 1:
        return out

    safe = np.maximum(closes, 1e-9)
    log_ret = np.diff(np.log(safe))  # length n-1

    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]
        out[i] = _shannon_entropy(window, bins)

    return out


def _shannon_entropy(returns: np.ndarray, num_bins: int) -> float:
    """Shannon entropy (base-2) of a discretised return distribution."""
    n = len(returns)
    if n < num_bins:
        return np.nan
    counts, _ = np.histogram(returns, bins=num_bins)
    probs = counts / float(n)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))
