import numpy as np


def klinger(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    fast: int = 34,
    slow: int = 55,
    signal: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Klinger Volume Oscillator (Stephen Klinger).

    Volume force indicator based on trend direction:
      Trend = +1 if (H + L + C) > (prev_H + prev_L + prev_C) else -1
      dm = H - L
      cm = cm_prev + dm  if trend unchanged, else dm_prev + dm  if trend flips
      VF = Volume * abs(2 * dm/cm - 1) * Trend * 100
      KVO = EMA(VF, fast) - EMA(VF, slow)
      Signal = EMA(KVO, signal)

    Returns (kvo, signal_line). Values before warmup are np.nan.

    Source: StockCharts ChartSchool.
    """
    n = len(closes)
    empty = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    if n == 0:
        return empty

    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    if n < 2:
        nan_arr = np.full(n, np.nan, dtype=np.float64)
        return (nan_arr.copy(), nan_arr.copy())

    # Trend direction
    hlc = highs + lows + closes
    trend = np.ones(n, dtype=np.float64)
    trend[0] = 1.0
    for i in range(1, n):
        if hlc[i] > hlc[i - 1]:
            trend[i] = 1.0
        elif hlc[i] < hlc[i - 1]:
            trend[i] = -1.0
        else:
            trend[i] = trend[i - 1]

    # dm and cm
    dm = highs - lows
    cm = np.zeros(n, dtype=np.float64)
    cm[0] = dm[0]
    for i in range(1, n):
        if trend[i] == trend[i - 1]:
            cm[i] = cm[i - 1] + dm[i]
        else:
            cm[i] = dm[i - 1] + dm[i]

    # Volume Force
    vf = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ratio = (2.0 * dm[i] / cm[i] - 1.0) if cm[i] != 0.0 else 0.0
        vf[i] = volumes[i] * abs(ratio) * trend[i] * 100.0

    # EMA helper
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(data), np.nan, dtype=np.float64)
        if len(data) < period:
            return out
        alpha = 2.0 / (period + 1)
        ema_val = np.mean(data[:period])
        out[period - 1] = ema_val
        for i in range(period, len(data)):
            ema_val = alpha * data[i] + (1.0 - alpha) * ema_val
            out[i] = ema_val
        return out

    ema_fast = _ema(vf, fast)
    ema_slow = _ema(vf, slow)

    kvo = np.full(n, np.nan, dtype=np.float64)
    start = slow - 1  # first index where both EMAs are valid
    for i in range(start, n):
        if not (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i])):
            kvo[i] = ema_fast[i] - ema_slow[i]

    # Signal line: EMA of valid KVO values
    # We need to apply EMA starting from the first non-nan KVO
    sig_line = np.full(n, np.nan, dtype=np.float64)
    valid_start = start
    kvo_valid = kvo[valid_start:]
    if len(kvo_valid) >= signal:
        alpha_s = 2.0 / (signal + 1)
        ema_s = np.nanmean(kvo_valid[:signal])
        sig_line[valid_start + signal - 1] = ema_s
        for i in range(signal, len(kvo_valid)):
            if not np.isnan(kvo_valid[i]):
                ema_s = alpha_s * kvo_valid[i] + (1.0 - alpha_s) * ema_s
                sig_line[valid_start + i] = ema_s

    return (kvo, sig_line)
