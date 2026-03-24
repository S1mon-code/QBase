import numpy as np


def twiggs_money_flow(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 21,
) -> np.ndarray:
    """Twiggs Money Flow (Colin Twiggs).

    Modification of Chaikin Money Flow that uses True Range to capture gaps
    and Wilder (EMA) smoothing instead of simple sums:

      True High (TRH) = max(High, prev_Close)
      True Low  (TRL) = min(Low,  prev_Close)
      AD = ((Close - TRL) - (TRH - Close)) / (TRH - TRL) * Volume
           = (2*Close - TRH - TRL) / (TRH - TRL) * Volume

      TMF = Wilder_EMA(AD, period) / Wilder_EMA(Volume, period)

    Wilder EMA uses alpha = 1/period. First ``period`` values are np.nan.

    Source: incrediblecharts.com / Colin Twiggs.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return result

    # True High / True Low (start from index 1)
    trh = np.maximum(highs[1:], closes[:-1])
    trl = np.minimum(lows[1:], closes[:-1])
    tr_range = trh - trl

    # AD values (length n-1, corresponding to indices 1..n-1)
    ad = np.where(
        tr_range != 0.0,
        (2.0 * closes[1:] - trh - trl) / tr_range * volumes[1:],
        0.0,
    )
    vol = volumes[1:]

    m = len(ad)  # n - 1
    if m < period:
        return result

    # Wilder EMA: alpha = 1 / period
    alpha = 1.0 / period

    # Seed with SMA over first `period` values
    ema_ad = np.mean(ad[:period])
    ema_vol = np.mean(vol[:period])

    idx = period  # maps to result index period (offset by 1 from ad start)
    result[idx] = ema_ad / ema_vol if ema_vol != 0.0 else 0.0

    for i in range(period, m):
        ema_ad = alpha * ad[i] + (1.0 - alpha) * ema_ad
        ema_vol = alpha * vol[i] + (1.0 - alpha) * ema_vol
        result[i + 1] = ema_ad / ema_vol if ema_vol != 0.0 else 0.0

    return result
