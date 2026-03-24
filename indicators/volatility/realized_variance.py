import numpy as np


def realized_variance(
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Realized Variance — sum of squared log returns over a rolling window.

    Formula:
        RV = sum(r_i^2) for i in window, where r_i = ln(C_i / C_{i-1})

    Reference: Andersen, T.G. & Bollerslev, T. (1998), "Answering the
    Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts."
    """
    n = len(closes)
    if n < 2 or n <= period:
        return np.full(n, np.nan)

    log_ret = np.log(closes[1:] / closes[:-1])  # length n-1
    ret_sq = log_ret ** 2

    out = np.full(n, np.nan)
    for i in range(period, n):
        # Window of period log returns ending at bar i
        out[i] = np.sum(ret_sq[i - period : i])

    return out


def realized_volatility(
    closes: np.ndarray,
    period: int = 20,
    annualize: int = 252,
) -> np.ndarray:
    """Realized Volatility — annualized square root of realized variance.

    Formula:
        RVol = sqrt(RV * annualize / period)

    This scales the rolling sum of squared returns to an annualized volatility.
    """
    rv = realized_variance(closes, period)
    n = len(closes)
    out = np.full(n, np.nan)
    valid = ~np.isnan(rv)
    out[valid] = np.sqrt(rv[valid] * annualize / period)
    return out
