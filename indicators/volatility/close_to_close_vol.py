import numpy as np


def close_to_close_vol(
    closes: np.ndarray,
    period: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Close-to-Close Volatility using simple (arithmetic) returns.

    Unlike historical_vol.py which uses log returns, this uses simple
    percentage returns: r_i = (C_i - C_{i-1}) / C_{i-1}.

    This is the classic Garman (1980) close-to-close estimator baseline.

    Formula:
        sigma = std(simple_returns, ddof=1) * sqrt(annualize)

    Reference: Garman, M.B. & Klass, M.J. (1980). The close-to-close
    estimator is the baseline against which more efficient estimators
    (Parkinson, GK, RS, YZ) are compared.
    """
    n = len(closes)
    if n < 2 or n <= period:
        return np.full(n, np.nan)

    # Simple (arithmetic) returns
    simple_ret = (closes[1:] - closes[:-1]) / closes[:-1]  # length n-1

    out = np.full(n, np.nan)
    for i in range(period, n):
        window = simple_ret[i - period : i]
        out[i] = np.std(window, ddof=1) * np.sqrt(annualize)

    return out
