import numpy as np


def vortex(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Vortex Indicator (Etienne Botes & Douglas Siepman).

    Measures positive and negative trend movement using true range and
    vortex movement:

      VM+ = |High_t - Low_{t-1}|
      VM- = |Low_t  - High_{t-1}|
      TR  = max(High-Low, |High-Close_{t-1}|, |Low-Close_{t-1}|)

      VI+ = sum(VM+, period) / sum(TR, period)
      VI- = sum(VM-, period) / sum(TR, period)

    VI+ crossing above VI- signals a bullish trend; the reverse signals
    a bearish trend.

    Returns (vi_plus, vi_minus). First *period* values are np.nan.
    """
    n = len(highs)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty

    vi_plus = np.full(n, np.nan, dtype=np.float64)
    vi_minus = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return vi_plus, vi_minus

    # Compute VM+ , VM- , TR  (all start at index 1)
    vm_p = np.abs(highs[1:] - lows[:-1])
    vm_m = np.abs(lows[1:] - highs[:-1])

    hl = highs[1:] - lows[1:]
    hc = np.abs(highs[1:] - closes[:-1])
    lc = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(hl, np.maximum(hc, lc))

    # Rolling sums over *period* bars (vm/tr arrays are 0-indexed from bar 1)
    cum_vmp = np.cumsum(vm_p)
    cum_vmm = np.cumsum(vm_m)
    cum_tr = np.cumsum(tr)

    # First valid value at original index = period
    vi_plus[period] = cum_vmp[period - 1] / cum_tr[period - 1]
    vi_minus[period] = cum_vmm[period - 1] / cum_tr[period - 1]

    for i in range(period + 1, n):
        j = i - 1  # index into vm/tr arrays (shifted by 1)
        sum_vmp = cum_vmp[j] - cum_vmp[j - period]
        sum_vmm = cum_vmm[j] - cum_vmm[j - period]
        sum_tr = cum_tr[j] - cum_tr[j - period]
        if sum_tr != 0.0:
            vi_plus[i] = sum_vmp / sum_tr
            vi_minus[i] = sum_vmm / sum_tr

    return vi_plus, vi_minus
