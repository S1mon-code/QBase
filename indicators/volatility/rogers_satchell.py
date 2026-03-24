import numpy as np


def rogers_satchell(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Rogers-Satchell Volatility Estimator (1991).

    Drift-independent OHLC volatility estimator. Unlike Parkinson and
    Garman-Klass, this estimator is unbiased for non-zero drift.

    Formula (per bar):
        RS_i = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)

    Rolling estimator:
        RS = sqrt( (1/n) * sum(RS_i) )

    Reference: Rogers, L.C.G., Satchell, S.E. & Yoon, Y. (1994),
    "Estimating the Volatility of Stock Prices," Applied Financial Economics.
    """
    n = len(closes)
    if n == 0 or n < period:
        return np.full(n, np.nan)

    # Per-bar Rogers-Satchell component
    log_hc = np.log(highs / closes)
    log_ho = np.log(highs / opens)
    log_lc = np.log(lows / closes)
    log_lo = np.log(lows / opens)
    rs_daily = log_hc * log_ho + log_lc * log_lo

    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = rs_daily[i - period + 1 : i + 1]
        variance = np.mean(window)
        out[i] = np.sqrt(max(variance, 0.0))

    return out
