import numpy as np


def garman_klass(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Garman-Klass Volatility Estimator (1980).

    More efficient than close-to-close volatility (7.4x). Uses OHLC data.
    Assumes Brownian motion with zero drift and no opening jumps.

    Formula (per bar):
        GK_i = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

    Rolling estimator:
        GK = sqrt( (1/n) * sum(GK_i) )

    Reference: Garman, M.B. & Klass, M.J. (1980), "On the Estimation of
    Security Price Volatilities from Historical Data," Journal of Business.
    """
    n = len(closes)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    # Per-bar Garman-Klass component
    log_hl = np.log(highs / lows)
    log_co = np.log(closes / opens)
    gk_daily = 0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2

    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = gk_daily[i - period + 1 : i + 1]
        variance = np.mean(window)
        out[i] = np.sqrt(max(variance, 0.0))

    return out
