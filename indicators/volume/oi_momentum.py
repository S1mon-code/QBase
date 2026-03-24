import numpy as np


def oi_momentum(oi: np.ndarray, period: int = 20) -> np.ndarray:
    """Open Interest Momentum (rate of change).

    Returns the fractional change of OI over ``period`` bars:
    (oi[i] - oi[i - period]) / oi[i - period].
    """
    if oi.size == 0:
        return np.array([], dtype=np.float64)

    n = len(oi)
    oi = oi.astype(np.float64)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        prev = oi[i - period]
        if prev > 0:
            out[i] = (oi[i] - prev) / prev
        else:
            out[i] = 0.0

    return out


def oi_sentiment(
    closes: np.ndarray,
    oi: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Combined OI + price + volume sentiment score.

    Classifies each bar into one of four quadrants based on OI rate-of-change
    and price direction over ``period`` bars, weighted by relative volume:

      Rising OI  + rising price  -> +1  (new longs, bullish continuation)
      Rising OI  + falling price -> -1  (new shorts, bearish continuation)
      Falling OI + rising price  -> -0.5 (short covering, weak rally)
      Falling OI + falling price -> +0.5 (long liquidation, weak selloff)

    The raw quadrant score is scaled by volume_ratio / 2 (capped at 1)
    so that volume surges amplify the signal.
    """
    if closes.size == 0 or oi.size == 0 or volumes.size == 0:
        return np.array([], dtype=np.float64)

    n = len(closes)
    closes = closes.astype(np.float64)
    oi = oi.astype(np.float64)
    volumes = volumes.astype(np.float64)

    out = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        # OI rate of change
        oi_prev = oi[i - period]
        if oi_prev > 0:
            oi_roc = (oi[i] - oi_prev) / oi_prev
        else:
            oi_roc = 0.0

        # Price direction
        price_rising = closes[i] > closes[i - period]

        # Quadrant score
        oi_rising = oi_roc > 0
        if oi_rising and price_rising:
            score = 1.0
        elif oi_rising and not price_rising:
            score = -1.0
        elif not oi_rising and price_rising:
            score = -0.5  # weak rally (short covering)
        else:
            score = 0.5   # weak selloff (long liquidation)

        # Volume weight: ratio of current volume to period average
        avg_vol = np.mean(volumes[max(0, i - period):i])
        if avg_vol > 0:
            vol_ratio = min(volumes[i] / avg_vol / 2.0, 1.0)
        else:
            vol_ratio = 0.5

        out[i] = score * vol_ratio

    return out
