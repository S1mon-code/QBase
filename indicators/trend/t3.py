import numpy as np


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Internal EMA helper."""
    n = len(data)
    alpha = 2.0 / (period + 1)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def t3(
    data: np.ndarray,
    period: int = 5,
    volume_factor: float = 0.7,
) -> np.ndarray:
    """Tillson T3 Moving Average.

    T3 applies a triple Generalised DEMA (GD).  A single GD is defined as:
      GD(data, n, v) = EMA(data, n) * (1 + v) - EMA(EMA(data, n), n) * v

    T3 = GD( GD( GD(data, n, v), n, v ), n, v )

    Equivalently, using six cascaded EMAs (e1..e6):
      T3 = -a^3*e6 + 3*a^2*(1+a)*e5 - 3*a*(1+a)^2*e4 + (1+a)^3*e3
    where a = volume_factor.

    A volume_factor of 0.7 (default) balances smoothness and responsiveness.
    When v=0 this reduces to an EMA; when v=1 it becomes a standard DEMA chain.
    """
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)

    e1 = _ema(data, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    e4 = _ema(e3, period)
    e5 = _ema(e4, period)
    e6 = _ema(e5, period)

    a = volume_factor
    c1 = -(a ** 3)
    c2 = 3.0 * a * a * (1.0 + a)
    c3 = -3.0 * a * (1.0 + a) ** 2
    c4 = (1.0 + a) ** 3

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
