import numpy as np


def ichimoku(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ichimoku Cloud (Ichimoku Kinko Hyo).

    Components:
      - Tenkan-sen  = (highest high + lowest low) / 2 over *tenkan* periods
      - Kijun-sen   = (highest high + lowest low) / 2 over *kijun* periods
      - Senkou Span A = (Tenkan + Kijun) / 2, shifted forward by *displacement*
      - Senkou Span B = (highest high + lowest low) / 2 over *senkou_b* periods,
                        shifted forward by *displacement*
      - Chikou Span = close shifted backward by *displacement*

    Returns (tenkan_sen, kijun_sen, senkou_a, senkou_b_line, chikou_span).
    Each array has length len(highs) + displacement to accommodate the forward
    shift of Senkou spans.  Chikou span is stored at index i - displacement
    (earlier entries are np.nan).
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty

    def _midpoint(period: int) -> np.ndarray:
        """(rolling max high + rolling min low) / 2."""
        out = np.full(n, np.nan, dtype=np.float64)
        for i in range(period - 1, n):
            hh = np.max(highs[i - period + 1 : i + 1])
            ll = np.min(lows[i - period + 1 : i + 1])
            out[i] = (hh + ll) / 2.0
        return out

    tenkan_sen = _midpoint(tenkan)
    kijun_sen = _midpoint(kijun)

    # Senkou Span A: (tenkan + kijun) / 2 displaced forward
    raw_a = (tenkan_sen + kijun_sen) / 2.0
    senkou_a = np.full(n + displacement, np.nan, dtype=np.float64)
    valid = ~np.isnan(raw_a)
    for i in range(n):
        if valid[i]:
            senkou_a[i + displacement] = raw_a[i]

    # Senkou Span B: midpoint over senkou_b periods, displaced forward
    raw_b = _midpoint(senkou_b)
    senkou_b_line = np.full(n + displacement, np.nan, dtype=np.float64)
    valid_b = ~np.isnan(raw_b)
    for i in range(n):
        if valid_b[i]:
            senkou_b_line[i + displacement] = raw_b[i]

    # Chikou Span: close shifted backward by displacement
    chikou_span = np.full(n, np.nan, dtype=np.float64)
    if n > displacement:
        chikou_span[: n - displacement] = closes[displacement:]

    # Pad tenkan/kijun to same length as senkou arrays
    tenkan_out = np.full(n + displacement, np.nan, dtype=np.float64)
    tenkan_out[:n] = tenkan_sen
    kijun_out = np.full(n + displacement, np.nan, dtype=np.float64)
    kijun_out[:n] = kijun_sen
    chikou_out = np.full(n + displacement, np.nan, dtype=np.float64)
    chikou_out[:n] = chikou_span

    return tenkan_out, kijun_out, senkou_a, senkou_b_line, chikou_out
