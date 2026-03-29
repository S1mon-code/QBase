import numpy as np


def range_expansion_signal(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    range_period: int = 25,
    range_mult: float = 1.9,
    atr_period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directional range expansion signal.

    Fires when the current bar's true range exceeds ``range_mult`` times
    the average range of the prior ``range_period`` bars. Direction follows
    the bar's open-to-close movement.

    This extends the basic range_expansion ratio indicator with:
    1. True Range calculation (accounts for gaps)
    2. Directional signal (+1 bullish bar, -1 bearish bar, 0 no signal)
    3. ATR value at signal time for stop/target computation

    Returns (signal, expansion_ratio, atr):
      signal          — +1 (bullish expansion), -1 (bearish), 0 (no signal)
      expansion_ratio — current_range / avg_range (always computed)
      atr             — ATR at each bar
    """
    n = len(closes)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()

    opens = opens.astype(np.float64)
    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    signal = np.zeros(n, dtype=np.float64)
    expansion_ratio = np.full(n, np.nan, dtype=np.float64)
    atr = np.full(n, np.nan, dtype=np.float64)

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR
    warmup = max(range_period, atr_period)
    if n > atr_period:
        running_atr = np.mean(tr[:atr_period])
        atr[atr_period - 1] = running_atr
        for i in range(atr_period, n):
            running_atr += (tr[i] - tr[i - atr_period]) / atr_period
            atr[i] = running_atr

    # Expansion ratio and directional signal
    for i in range(range_period, n):
        avg_range = np.mean(tr[i - range_period : i])
        if avg_range < 1e-12:
            expansion_ratio[i] = 0.0
            continue

        ratio = tr[i] / avg_range
        expansion_ratio[i] = ratio

        if ratio > range_mult:
            signal[i] = 1.0 if closes[i] > opens[i] else -1.0

    return signal, expansion_ratio, atr
