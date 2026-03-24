import numpy as np


def keltner_width(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 1.5,
) -> np.ndarray:
    """Keltner Channel Width.

    Measures the normalized bandwidth of the Keltner Channel.

    Keltner Channel:
        Middle = EMA(Close, ema_period)
        Upper  = Middle + multiplier * ATR(atr_period)
        Lower  = Middle - multiplier * ATR(atr_period)

    Width = (Upper - Lower) / Middle
         = 2 * multiplier * ATR / EMA

    Wider channels indicate higher volatility; narrower channels indicate
    compression (potential squeeze setups).

    Reference: Chester Keltner (1960), modernized by Linda Raschke
    using EMA and ATR.
    """
    n = len(closes)
    if n == 0:
        return np.full(n, np.nan)

    # EMA of closes
    ema = np.full(n, np.nan)
    if n < ema_period:
        return np.full(n, np.nan)
    ema[ema_period - 1] = np.mean(closes[:ema_period])
    k = 2.0 / (ema_period + 1)
    for i in range(ema_period, n):
        ema[i] = closes[i] * k + ema[i - 1] * (1.0 - k)

    # True Range
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR using Wilder's smoothing
    atr_vals = np.full(n, np.nan)
    if n < atr_period + 1:
        return np.full(n, np.nan)
    atr_vals[atr_period] = np.mean(tr[1 : atr_period + 1])
    alpha = 1.0 / atr_period
    for i in range(atr_period + 1, n):
        atr_vals[i] = atr_vals[i - 1] * (1.0 - alpha) + tr[i] * alpha

    # Width = (Upper - Lower) / Middle = 2 * multiplier * ATR / EMA
    out = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ema[i]) and not np.isnan(atr_vals[i]) and ema[i] != 0:
            out[i] = 2.0 * multiplier * atr_vals[i] / ema[i]

    return out
