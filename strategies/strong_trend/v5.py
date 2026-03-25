"""
Strong Trend Strategy v5 — Hull MA + CCI + Volume Climax
=========================================================
Captures powerful trending moves using:
  1. HMA (trend direction & dynamic support)
  2. CCI (momentum strength confirmation)
  3. Volume Climax (buying-pressure confirmation)

Long-only. Supports position scaling 0-3.

Usage:
    ./run.sh strategies/strong_trend/v5.py --symbols AG --freq daily --start 2022
"""
import sys
from pathlib import Path

# Ensure QBase root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest  # noqa: F401  — configures AlphaForge path

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy

# Indicators (3 core + ATR for risk management)
from indicators.trend.hma import hma
from indicators.momentum.cci import cci
from indicators.volume.volume_spike import volume_climax
from indicators.volatility.atr import atr


class StrongTrendV5(TimeSeriesStrategy):
    """HMA + CCI + Volume Climax — long-only strong-trend strategy."""

    name = "strong_trend_v5"
    warmup = 60
    freq = "daily"

    # ── Tunable parameters (5 total) ────────────────────────────────
    hma_period: int = 20
    cci_period: int = 20
    cci_threshold: int = 100
    climax_period: int = 20
    atr_trail_mult: float = 3.5

    def __init__(self):
        super().__init__()
        self._hma = None
        self._cci = None
        self._climax = None
        self._atr = None

    # ── Internal state ──────────────────────────────────────────────

    def on_init(self, context):
        """Reset tracking variables at strategy start."""
        self.highest_since_entry = 0.0  # for trailing stop
        self.position_scale = 0         # current add-on count (0-3)

    def on_init_arrays(self, context, bars):
        """Pre-compute all indicators on full data arrays."""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._hma = hma(closes, self.hma_period)
        self._cci = cci(highs, lows, closes, self.cci_period)
        self._climax = volume_climax(highs, lows, closes, volumes, self.climax_period)
        self._atr = atr(highs, lows, closes, period=14)

    # ----------------------------------------------------------------
    #  Main bar handler
    # ----------------------------------------------------------------
    def on_bar(self, context):
        i = context.bar_index

        if i < 1:
            return

        price = context.close_raw
        side, lots = context.position

        # --- Look up pre-computed indicators ---
        hma_now = self._hma[i]
        hma_prev = self._hma[i - 1]
        cci_now = self._cci[i]
        atr_now = self._atr[i]

        # Guard: skip bar if any indicator is not ready
        if np.isnan(hma_now) or np.isnan(hma_prev) or np.isnan(cci_now) or np.isnan(atr_now):
            return

        # HMA direction
        hma_rising = hma_now > hma_prev

        # Check for positive volume climax in the last 3 bars
        recent_climax = self._climax[max(0, i - 2):i + 1]
        has_buying_climax = any(
            not np.isnan(v) and v > 0 for v in recent_climax
        )

        # Contract multiplier for position sizing
        contract_mult = getattr(context, "contract_multiplier", 100)

        # ── FLAT — look for entry ──────────────────────────────────
        if lots == 0:
            if (
                hma_rising
                and price > hma_now
                and cci_now > self.cci_threshold
                and has_buying_climax
            ):
                lot_size = self._calc_lots(context, atr_now, contract_mult)
                if lot_size > 0:
                    context.buy(lot_size)
                    self.highest_since_entry = price
                    self.position_scale = 1
            return

        # ── LONG — manage position ─────────────────────────────────
        if side == 1:
            # Update trailing-stop tracker
            if price > self.highest_since_entry:
                self.highest_since_entry = price

            # -- Trailing stop check --------------------------------
            trail_stop = self.highest_since_entry - self.atr_trail_mult * atr_now
            if price < trail_stop:
                context.close_long()
                self._reset_state()
                return

            # -- Hard exit: price below HMA or CCI deeply negative --
            if price < hma_now or cci_now < -100:
                context.close_long()
                self._reset_state()
                return

            # -- Reduce: CCI fading but trend intact ----------------
            #    CCI dropped below 0 while HMA still rising → trim half
            if cci_now < 0 and hma_rising and self.position_scale > 0:
                half = max(lots // 2, 1)
                context.close_long(lots=half)
                self.position_scale = max(self.position_scale - 1, 0)
                return

            # -- Add: extremely strong momentum, room to scale ------
            if (
                cci_now > 200
                and price > hma_now
                and self.position_scale < 3
            ):
                add_lots = self._calc_lots(context, atr_now, contract_mult)
                if add_lots > 0:
                    context.buy(add_lots)
                    self.position_scale += 1
                    # Refresh trailing high after adding
                    if price > self.highest_since_entry:
                        self.highest_since_entry = price

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------
    def _calc_lots(self, context, atr_now: float, contract_mult: float) -> int:
        """Position sizing: risk 2% of equity per unit of risk.

        Risk per lot = atr_trail_mult * ATR * contract_multiplier.
        """
        risk_budget = context.equity * 0.02
        risk_per_lot = self.atr_trail_mult * atr_now * contract_mult
        if risk_per_lot <= 0:
            return 1
        raw = risk_budget / risk_per_lot
        return max(1, int(raw))

    def _reset_state(self):
        """Clear tracking variables after a full exit."""
        self.highest_since_entry = 0.0
        self.position_scale = 0
