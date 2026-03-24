import numpy as np


def adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index using Wilder's smoothing method."""
    adx_vals, _, _ = adx_with_di(highs, lows, closes, period)
    return adx_vals


def adx_with_di(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX with directional indicators.

    Returns (adx, plus_di, minus_di). First ~2*period values are np.nan.
    """
    n = len(highs)
    if n == 0 or n < period + 1:
        nans = np.full(n, np.nan)
        return nans.copy(), nans.copy(), nans.copy()

    # --- raw +DM, -DM, TR ---
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    high_low = highs[1:] - lows[1:]
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    # --- Wilder smoothing helper ---
    def wilder_smooth(values: np.ndarray, p: int) -> np.ndarray:
        """First value = sum of first p elements, then Wilder's EMA."""
        out = np.full(len(values), np.nan)
        if len(values) < p:
            return out
        out[p - 1] = np.sum(values[:p])
        for i in range(p, len(values)):
            out[i] = out[i - 1] - out[i - 1] / p + values[i]
        return out

    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)
    smoothed_tr = wilder_smooth(tr, period)

    # --- +DI, -DI ---
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    valid = smoothed_tr > 0
    # smoothed arrays are length n-1 (aligned to index 1..n-1 of original)
    idx = np.where(valid)[0]
    plus_di[idx + 1] = 100.0 * smoothed_plus_dm[idx] / smoothed_tr[idx]
    minus_di[idx + 1] = 100.0 * smoothed_minus_dm[idx] / smoothed_tr[idx]

    # --- DX ---
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.full(n, np.nan)
    nonzero = di_sum > 0
    dx[nonzero] = 100.0 * di_diff[nonzero] / di_sum[nonzero]

    # --- ADX = Wilder smooth of DX ---
    # Collect valid DX values starting from index (period) in the original array
    # First valid DX is at original index = period (0-based)
    adx_out = np.full(n, np.nan)

    # Find the first index where DX is not nan
    first_valid = period  # DX first valid at original index = period
    dx_valid = dx[first_valid:]
    if len(dx_valid) < period:
        return adx_out, plus_di, minus_di

    # First ADX = mean of first `period` valid DX values
    adx_start = first_valid + period - 1  # original index
    adx_out[adx_start] = np.mean(dx_valid[:period])
    for i in range(adx_start + 1, n):
        adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period

    return adx_out, plus_di, minus_di
