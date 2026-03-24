import numpy as np


def emv(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Ease of Movement (Richard Arms).

    Relates price movement to volume via:
      Distance Moved = ((H + L) / 2) - ((prev_H + prev_L) / 2)
      Box Ratio      = (Volume / 1e8) / (H - L)
      EMV_1          = Distance Moved / Box Ratio
      EMV            = SMA(EMV_1, period)

    First ``period`` values are np.nan (1 bar lost to diff, then period-1 for SMA).

    Source: StockCharts ChartSchool.
    """
    n = len(highs)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    volumes = volumes.astype(np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return result

    # Distance moved: midpoint change
    mid = (highs + lows) / 2.0
    distance = np.diff(mid)  # length n-1

    # Box ratio
    hl_range = highs[1:] - lows[1:]
    box_ratio = np.where(hl_range != 0.0, (volumes[1:] / 1e8) / hl_range, 0.0)

    # 1-period EMV
    emv_1 = np.where(box_ratio != 0.0, distance / box_ratio, 0.0)

    # SMA of emv_1 over `period`
    warmup = period  # index in emv_1
    if len(emv_1) < period:
        return result

    running_sum = np.sum(emv_1[:period])
    result[period] = running_sum / period  # offset by 1 because emv_1 starts at bar 1

    for i in range(period, len(emv_1)):
        running_sum += emv_1[i] - emv_1[i - period]
        result[i + 1] = running_sum / period

    return result
