import numpy as np


def wad(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Williams Accumulation/Distribution (Larry Williams, Achelis version).

    Cumulative indicator using True Range High/Low:
      TRH = max(High, prev_Close)
      TRL = min(Low,  prev_Close)

      if C > prev_C:  AD = C - TRL
      if C < prev_C:  AD = C - TRH
      if C == prev_C: AD = 0

      WAD = cumulative sum of AD

    First value is 0 (no prior bar to compare). This is the Steven Achelis
    modification that does not multiply by volume.

    Source: Tulip Indicators / Larry Williams.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)

    result = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        trh = max(highs[i], closes[i - 1])
        trl = min(lows[i], closes[i - 1])

        if closes[i] > closes[i - 1]:
            ad_val = closes[i] - trl
        elif closes[i] < closes[i - 1]:
            ad_val = closes[i] - trh
        else:
            ad_val = 0.0

        result[i] = result[i - 1] + ad_val

    return result
