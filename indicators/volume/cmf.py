import numpy as np


def cmf(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Chaikin Money Flow (Marc Chaikin).

    Oscillator in [-1, 1] measuring buying/selling pressure over a period:
      CLV = ((Close - Low) - (High - Close)) / (High - Low)
      MF_Volume = CLV * Volume
      CMF = Sum(MF_Volume, period) / Sum(Volume, period)

    First ``period - 1`` values are np.nan.

    Source: StockCharts ChartSchool.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    hl_range = highs - lows
    clv = np.where(hl_range != 0.0, ((closes - lows) - (highs - closes)) / hl_range, 0.0)
    mf_volume = clv * volumes

    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    mfv_sum = np.sum(mf_volume[:period])
    vol_sum = np.sum(volumes[:period])

    result[period - 1] = mfv_sum / vol_sum if vol_sum != 0.0 else 0.0

    for i in range(period, n):
        mfv_sum += mf_volume[i] - mf_volume[i - period]
        vol_sum += volumes[i] - volumes[i - period]
        result[i] = mfv_sum / vol_sum if vol_sum != 0.0 else 0.0

    return result
