import numpy as np


def supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Supertrend indicator based on ATR bands.

    Returns (supertrend_line, direction) where direction is 1 (bullish)
    or -1 (bearish). First `period` values of supertrend_line are np.nan.
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    st_line = np.full(n, np.nan)
    direction = np.ones(n, dtype=np.float64)

    if n < period:
        return st_line, direction

    # --- ATR via Wilder's smoothing ---
    high_low = highs[1:] - lows[1:]
    high_close = np.abs(highs[1:] - closes[:-1])
    low_close = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))

    atr = np.full(n, np.nan)
    # First ATR = mean of first `period` TR values  (TR starts at index 1)
    if len(tr) < period:
        return st_line, direction
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period

    # --- Basic upper / lower bands ---
    hl2 = (highs + lows) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # --- Final bands with flip logic ---
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)

    final_upper[period] = basic_upper[period]
    final_lower[period] = basic_lower[period]

    # Initial direction: bullish if close > upper band
    if closes[period] > final_upper[period]:
        direction[period] = 1
        st_line[period] = final_lower[period]
    else:
        direction[period] = -1
        st_line[period] = final_upper[period]

    for i in range(period + 1, n):
        # Tighten bands: upper can only move down, lower can only move up
        if basic_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if basic_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Direction flip
        if direction[i - 1] == -1 and closes[i] > final_upper[i]:
            direction[i] = 1
        elif direction[i - 1] == 1 and closes[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        # Supertrend line follows the appropriate band
        if direction[i] == 1:
            st_line[i] = final_lower[i]
        else:
            st_line[i] = final_upper[i]

    # Blank out warmup
    st_line[:period] = np.nan
    direction[:period] = np.nan

    return st_line, direction
