import numpy as np


def money_flow(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Raw Money Flow.

    Typical Price multiplied by volume for each bar:
      TP = (High + Low + Close) / 3
      Money Flow = TP * Volume

    No warmup; all values are valid.

    Source: General technical analysis reference.
    """
    if closes.size == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    tp = (highs + lows + closes) / 3.0
    return tp * volumes


def money_flow_ratio(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Money Flow Ratio.

    Ratio of positive money flow to negative money flow over a rolling period:
      TP = (H + L + C) / 3
      Raw MF = TP * Volume
      Positive MF: raw MF on bars where TP > prev_TP
      Negative MF: raw MF on bars where TP < prev_TP
      MFR = Sum(Positive MF, period) / Sum(Negative MF, period)

    First ``period`` values are np.nan.

    Source: General technical analysis (same building block as MFI).
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=np.float64)

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    tp = (highs + lows + closes) / 3.0
    raw_mf = tp * volumes

    result = np.full(n, np.nan, dtype=np.float64)

    # Direction based on TP change
    tp_diff = np.zeros(n, dtype=np.float64)
    tp_diff[1:] = np.diff(tp)

    pos_flow = np.where(tp_diff > 0, raw_mf, 0.0)
    neg_flow = np.where(tp_diff < 0, raw_mf, 0.0)

    if n <= period:
        return result

    pos_sum = np.sum(pos_flow[1 : period + 1])
    neg_sum = np.sum(neg_flow[1 : period + 1])

    result[period] = pos_sum / neg_sum if neg_sum != 0.0 else np.inf

    for i in range(period + 1, n):
        pos_sum += pos_flow[i] - pos_flow[i - period]
        neg_sum += neg_flow[i] - neg_flow[i - period]
        result[i] = pos_sum / neg_sum if neg_sum != 0.0 else np.inf

    return result
