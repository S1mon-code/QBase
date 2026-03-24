import numpy as np


def chaikin_volatility(
    highs: np.ndarray,
    lows: np.ndarray,
    ema_period: int = 10,
    roc_period: int = 10,
) -> np.ndarray:
    """Chaikin Volatility — Rate of Change of EMA of (High - Low).

    Steps:
        1. HL = High - Low
        2. EMA_HL = EMA(HL, ema_period)
        3. CV = (EMA_HL - EMA_HL[roc_period ago]) / EMA_HL[roc_period ago] * 100

    Reference: Marc Chaikin. Rising values signal increasing volatility;
    falling values signal decreasing volatility.
    """
    n = len(highs)
    if n == 0:
        return np.full(n, np.nan)

    hl = highs - lows

    # EMA of HL range
    ema_hl = np.full(n, np.nan)
    if n < ema_period:
        return np.full(n, np.nan)

    # Seed EMA with SMA
    ema_hl[ema_period - 1] = np.mean(hl[:ema_period])
    k = 2.0 / (ema_period + 1)
    for i in range(ema_period, n):
        ema_hl[i] = hl[i] * k + ema_hl[i - 1] * (1.0 - k)

    # Rate of Change of EMA
    out = np.full(n, np.nan)
    start = ema_period - 1 + roc_period
    for i in range(start, n):
        prev = ema_hl[i - roc_period]
        if prev != 0 and not np.isnan(prev):
            out[i] = (ema_hl[i] - prev) / prev * 100.0

    return out
