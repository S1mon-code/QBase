import numpy as np


def macro_regime_filter(
    closes: np.ndarray,
    volumes: np.ndarray,
    ma_period: int = 60,
    vol_period: int = 20,
    vol_lookback: int = 120,
    oi_period: int = 20,
    slope_threshold: float = 0.001,
    vol_percentile_threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-factor macro regime filter combining trend, volatility, and participation.

    Three independent regime signals aggregated into a composite score:

    1. Trend: MA slope direction over 5 bars
       +1 if slope > threshold, -1 if < -threshold, else 0
    2. Volatility: current realized vol percentile vs history
       +1 if percentile < vol_percentile_threshold (calm), -1 if above (chaotic)
    3. Participation: recent volume trend vs lagged volume trend
       +1 if vol_ma > vol_ma_prev (rising participation), else -1

    Regime score = trend + vol + participation, range [-3, +3].
    Regime signal: +1 if score >= 2 (bullish aligned), -1 if <= -2 (bearish), 0 otherwise.

    Returns (regime_score, regime_signal, trend_score, vol_score):
      regime_score  — raw composite score [-3, +3]
      regime_signal — filtered signal: +1, -1, or 0
      trend_score   — trend component [-1, 0, +1]
      vol_score     — volatility component [-1, +1]

    Source: BlackEdge S13, inspired by macro factor filtering.
    """
    n = len(closes)
    regime_score = np.full(n, np.nan, dtype=np.float64)
    regime_signal = np.zeros(n, dtype=np.float64)
    trend_score_arr = np.full(n, np.nan, dtype=np.float64)
    vol_score_arr = np.full(n, np.nan, dtype=np.float64)

    warmup = max(ma_period + 5, vol_lookback + vol_period, oi_period + 5)
    if n < warmup:
        return regime_score, regime_signal, trend_score_arr, vol_score_arr

    closes = closes.astype(np.float64)
    volumes = volumes.astype(np.float64)

    # Pre-compute log returns for vol calculation
    safe = np.maximum(closes, 1e-9)
    log_ret = np.diff(np.log(safe))

    for i in range(warmup, n):
        # --- Component 1: Trend (MA slope) ---
        ma_start = i - ma_period - 4
        if ma_start < 0:
            continue
        ma_window = closes[ma_start : i + 1]
        kernel = np.ones(ma_period, dtype=np.float64) / ma_period
        ma_vals = np.convolve(ma_window, kernel, mode='valid')
        if len(ma_vals) < 5 or ma_vals[-5] < 1e-9:
            continue
        ma_slope = (ma_vals[-1] - ma_vals[-5]) / ma_vals[-5]
        t_score = 1.0 if ma_slope > slope_threshold else (-1.0 if ma_slope < -slope_threshold else 0.0)

        # --- Component 2: Volatility percentile ---
        vol_start = max(0, i - vol_lookback)
        ret_window = log_ret[vol_start : i]
        if len(ret_window) < vol_period * 2:
            continue

        rolling_vols = []
        for j in range(vol_period, len(ret_window) + 1):
            rolling_vols.append(np.std(ret_window[j - vol_period : j]))

        if len(rolling_vols) < 2:
            continue

        current_vol = rolling_vols[-1]
        percentile = sum(1 for v in rolling_vols if v < current_vol) / len(rolling_vols)
        v_score = 1.0 if percentile < vol_percentile_threshold else -1.0

        # --- Component 3: Participation (volume trend) ---
        if i < oi_period + 5:
            continue
        vol_ma = np.mean(volumes[i - oi_period + 1 : i + 1])
        vol_ma_prev = np.mean(volumes[i - oi_period - 4 : i - 4])
        oi_score = 1.0 if vol_ma > vol_ma_prev else -1.0

        # --- Composite ---
        score = t_score + v_score + oi_score
        regime_score[i] = score
        trend_score_arr[i] = t_score
        vol_score_arr[i] = v_score

        if score >= 2.0:
            regime_signal[i] = 1.0
        elif score <= -2.0:
            regime_signal[i] = -1.0

    return regime_score, regime_signal, trend_score_arr, vol_score_arr
