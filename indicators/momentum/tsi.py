import numpy as np


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. First `period-1` values are np.nan."""
    result = np.full(arr.size, np.nan)
    if arr.size < period:
        return result
    result[period - 1] = arr[:period].mean()
    k = 2.0 / (period + 1.0)
    for i in range(period, arr.size):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _ema_skip_nan(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA that skips leading NaNs, then applies standard EMA."""
    result = np.full(arr.size, np.nan)
    valid_mask = ~np.isnan(arr)
    first_valid = np.argmax(valid_mask)
    if not valid_mask[first_valid]:
        return result

    sub = arr[first_valid:]
    sub_ema = _ema(sub, period)
    result[first_valid:] = sub_ema
    return result


def tsi(
    closes: np.ndarray,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """True Strength Index.

    TSI = 100 * EMA(EMA(momentum, long), short) / EMA(EMA(|momentum|, long), short)
    Signal = EMA(TSI, signal_period)

    Double-smoothed momentum oscillator. Typically ranges roughly -100 to 100.
    Returns (tsi_line, signal_line).
    """
    n = closes.size
    empty = np.array([], dtype=float)
    if n == 0:
        return empty, empty
    nans = np.full(n, np.nan)
    if n <= long_period + short_period:
        return nans.copy(), nans.copy()

    momentum = np.diff(closes)
    abs_momentum = np.abs(momentum)

    # Double smooth: first EMA(long), then EMA(short) of result
    ema1_mom = _ema(momentum, long_period)
    ema2_mom = _ema_skip_nan(ema1_mom, short_period)

    ema1_abs = _ema(abs_momentum, long_period)
    ema2_abs = _ema_skip_nan(ema1_abs, short_period)

    # TSI (momentum array is size n-1, so result indices are shifted by 1)
    tsi_raw = np.full(n - 1, np.nan)
    valid = (~np.isnan(ema2_mom)) & (~np.isnan(ema2_abs)) & (ema2_abs != 0)
    tsi_raw[valid] = 100.0 * ema2_mom[valid] / ema2_abs[valid]

    # Put back into full-size array (index 0 is NaN because diff loses one element)
    tsi_line = np.full(n, np.nan)
    tsi_line[1:] = tsi_raw

    # Signal line
    sig = _ema_skip_nan(tsi_raw, signal_period)
    signal_line = np.full(n, np.nan)
    signal_line[1:] = sig

    return tsi_line, signal_line
