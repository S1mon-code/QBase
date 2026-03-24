import numpy as np


def yang_zhang(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Yang-Zhang Volatility Estimator (2000).

    Combines overnight (close-to-open), open-to-close, and Rogers-Satchell
    components. Most efficient OHLC estimator (14x more efficient than
    close-to-close). Handles opening jumps and drift.

    Formula:
        sigma^2 = sigma_o^2 + k * sigma_c^2 + (1 - k) * sigma_rs^2

    Where:
        sigma_o^2 = overnight variance (close-to-open log returns)
        sigma_c^2 = close-to-open (open-to-close) log return variance
        sigma_rs^2 = Rogers-Satchell variance
        k = 0.34 / (1.34 + (n+1)/(n-1))

    Reference: Yang, D. & Zhang, Q. (2000), "Drift-Independent Volatility
    Estimation Based on High, Low, Open, and Close Prices," Journal of Business.
    """
    n = len(closes)
    if n == 0 or n < period + 1:
        return np.full(n, np.nan)

    # Log returns
    # Overnight: log(Open_t / Close_{t-1})
    log_oc = np.log(opens[1:] / closes[:-1])  # length n-1
    # Open-to-close: log(Close_t / Open_t)
    log_co = np.log(closes[1:] / opens[1:])   # length n-1

    # Rogers-Satchell per bar (using bars 1..n-1)
    log_hc = np.log(highs[1:] / closes[1:])
    log_ho = np.log(highs[1:] / opens[1:])
    log_lc = np.log(lows[1:] / closes[1:])
    log_lo = np.log(lows[1:] / opens[1:])
    rs_daily = log_hc * log_ho + log_lc * log_lo

    # k parameter
    k = 0.34 / (1.34 + (period + 1.0) / (period - 1.0))

    out = np.full(n, np.nan)
    # We need period bars of log_oc, log_co, rs_daily (indices 0..n-2 in those arrays)
    # which corresponds to output index period onward
    for i in range(period, n):
        # Window in the log return arrays: [i-period .. i-1]
        oc_window = log_oc[i - period : i]
        co_window = log_co[i - period : i]
        rs_window = rs_daily[i - period : i]

        oc_mean = np.mean(oc_window)
        co_mean = np.mean(co_window)

        # Overnight variance (ddof=1)
        sigma_o_sq = np.sum((oc_window - oc_mean) ** 2) / (period - 1)
        # Close-to-open (intraday) variance (ddof=1)
        sigma_c_sq = np.sum((co_window - co_mean) ** 2) / (period - 1)
        # Rogers-Satchell variance
        sigma_rs_sq = np.mean(rs_window)

        yz_var = sigma_o_sq + k * sigma_c_sq + (1.0 - k) * sigma_rs_sq
        out[i] = np.sqrt(max(yz_var, 0.0))

    return out
