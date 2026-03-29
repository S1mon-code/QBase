import numpy as np


def ou_params(
    closes: np.ndarray,
    period: int = 120,
    dt: float = 1.0 / 252.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full Ornstein-Uhlenbeck parameter estimation via AR(1) MLE.

    Fits dX = theta(mu - X)dt + sigma*dW on rolling windows of log prices.
    Uses AR(1) regression: y_t = a + b*y_{t-1} + eps, then derives OU params.

    Returns (theta, mu, sigma, ou_std, half_life):
      theta   — mean-reversion speed (higher = faster reversion)
      mu      — long-term equilibrium log-price
      sigma   — OU diffusion coefficient
      ou_std  — equilibrium standard deviation = sigma / sqrt(2*theta)
      half_life — ln(2)/theta in trading days (NaN when not mean-reverting)

    Source: Bertram (2010) optimal stopping for OU processes.
    """
    n = len(closes)
    theta = np.full(n, np.nan, dtype=np.float64)
    mu = np.full(n, np.nan, dtype=np.float64)
    sigma = np.full(n, np.nan, dtype=np.float64)
    ou_std = np.full(n, np.nan, dtype=np.float64)
    half_life = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return theta, mu, sigma, ou_std, half_life

    safe = np.maximum(closes.astype(np.float64), 1e-9)
    log_p = np.log(safe)

    for i in range(period, n):
        window = log_p[i - period : i + 1]
        x = window[:-1]
        y = window[1:]
        m = len(x)

        sx = np.sum(x)
        sy = np.sum(y)
        sxx = np.dot(x, x)
        sxy = np.dot(x, y)

        denom = m * sxx - sx * sx
        if abs(denom) < 1e-14:
            continue

        b = (m * sxy - sx * sy) / denom
        a = (sy - b * sx) / m

        # b must be in (0, 1) for mean-reverting OU
        if b <= 0.0 or b >= 1.0:
            continue

        residuals = y - a - b * x
        sigma_e = np.sqrt(np.sum(residuals ** 2) / m)

        th = -np.log(b) / dt
        if th < 1e-10:
            continue

        mu_val = a / (1.0 - b)
        b2 = b * b
        if 1.0 - b2 < 1e-14:
            continue
        sig = sigma_e * np.sqrt(2.0 * th / (1.0 - b2))
        std_val = sig / np.sqrt(2.0 * th)
        hl = np.log(2.0) / th / dt  # in trading days

        theta[i] = th
        mu[i] = mu_val
        sigma[i] = sig
        ou_std[i] = std_val
        half_life[i] = hl

    return theta, mu, sigma, ou_std, half_life


def ou_deviation(
    closes: np.ndarray,
    period: int = 120,
    dt: float = 1.0 / 252.0,
    min_half_life: float = 3.0,
    max_half_life: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """OU deviation signal: how far price is from equilibrium in OU-std units.

    Filters for valid mean-reverting regimes (half_life within bounds).

    Returns (deviation, valid_regime):
      deviation    — (log_price - mu) / ou_std, positive = overvalued
      valid_regime — 1.0 when half_life in [min_hl, max_hl], else 0.0
    """
    n = len(closes)
    deviation = np.full(n, np.nan, dtype=np.float64)
    valid_regime = np.zeros(n, dtype=np.float64)

    if n < period + 1:
        return deviation, valid_regime

    th, mu, sig, ou_std, hl = ou_params(closes, period, dt)

    safe = np.maximum(closes.astype(np.float64), 1e-9)
    log_p = np.log(safe)

    for i in range(period, n):
        if np.isnan(hl[i]) or np.isnan(ou_std[i]):
            continue
        if ou_std[i] < 1e-14:
            continue
        if min_half_life <= hl[i] <= max_half_life:
            valid_regime[i] = 1.0
            deviation[i] = (log_p[i] - mu[i]) / ou_std[i]

    return deviation, valid_regime
