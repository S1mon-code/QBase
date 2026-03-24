import numpy as np


def volume_rsi(
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Volume-Weighted RSI.

    RSI where gains and losses are weighted by volume, making it more
    responsive to high-volume price moves:

      gain[i] = max(close[i] - close[i-1], 0) * volume[i]
      loss[i] = max(close[i-1] - close[i], 0) * volume[i]

      avg_gain, avg_loss use Wilder smoothing (alpha = 1/period)
      VWRSI = 100 - 100 / (1 + avg_gain / avg_loss)

    First ``period`` values are np.nan.

    Source: Quong & Soudack, *Technical Analysis of Stocks & Commodities* V7:3.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    if n <= period:
        return result

    deltas = np.diff(closes)
    # Weight gains/losses by volume (volume at bar i corresponds to delta i-1..i)
    gains = np.where(deltas > 0, deltas * volumes[1:], 0.0)
    losses = np.where(deltas < 0, -deltas * volumes[1:], 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        result[period] = 100.0
    else:
        result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    alpha = 1.0 / period
    for i in range(period, len(deltas)):
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha
        if avg_loss == 0.0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return result
