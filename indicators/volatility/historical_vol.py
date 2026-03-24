import numpy as np


def historical_volatility(
    closes: np.ndarray,
    period: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Annualised historical volatility from rolling std of log returns.

    Formula: std(log_returns, ddof=1) * sqrt(annualize).
    First `period` values are np.nan.
    """
    n = len(closes)
    if n < 2 or n <= period:
        return np.full(n, np.nan)

    log_ret = np.log(closes[1:] / closes[:-1])  # length n-1

    out = np.full(n, np.nan)
    # We need `period` log returns, which maps to index `period` in the output
    for i in range(period, n):
        window = log_ret[i - period : i]
        out[i] = np.std(window, ddof=1) * np.sqrt(annualize)

    return out
