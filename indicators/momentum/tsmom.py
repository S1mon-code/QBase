import numpy as np


def tsmom(closes: np.ndarray, lookback: int = 252,
          vol_lookback: int = 60) -> np.ndarray:
    """Time Series Momentum — risk-adjusted momentum signal.

    signal = past_return / realized_vol
    Positive values indicate bullish momentum, negative bearish.
    Magnitude reflects risk-adjusted strength.
    """
    if closes.size == 0:
        return np.array([], dtype=float)
    n = closes.size
    warmup = max(lookback, vol_lookback + 1)
    if n <= warmup:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)

    for i in range(warmup, n):
        past_price = closes[i - lookback]
        if past_price <= 0:
            continue
        past_return = closes[i] / past_price - 1.0

        log_rets = np.diff(np.log(closes[i - vol_lookback:i + 1]))
        realized_vol = np.std(log_rets) * np.sqrt(252)
        realized_vol = max(realized_vol, 1e-8)

        result[i] = past_return / realized_vol

    return result
