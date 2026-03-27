import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ichimoku import ichimoku
from indicators.volume.mfi import mfi
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV102(TimeSeriesStrategy):
    """
    策略简介：Ichimoku Cloud突破 + MFI资金流确认的日线多头策略。

    使用指标：
    - Ichimoku(9,26,52,26): 价格突破云层上边界确认趋势
    - MFI(14): Money Flow Index > 50确认资金流入
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 价格 > Senkou Span A 且 > Senkou Span B（突破云层上方）
    - Tenkan > Kijun（短期趋势确认）
    - MFI > 50

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - 价格跌入云层内（Tenkan < Kijun）

    优点：Ichimoku多维度确认趋势，云层提供天然支撑
    缺点：Ichimoku参数多，在不同市场适应性不同
    """
    name = "medium_trend_v102"
    warmup = 120
    freq = "daily"

    tenkan: int = 9
    kijun: int = 26
    senkou_b: int = 52
    displacement: int = 26
    mfi_period: int = 14       # Optuna: 10-20
    atr_stop_mult: float = 3.0 # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._tenkan_sen = None
        self._kijun_sen = None
        self._senkou_a = None
        self._senkou_b_line = None
        self._mfi = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        n = len(closes)

        t, k, sa, sb, _ = ichimoku(
            highs, lows, closes,
            tenkan=self.tenkan, kijun=self.kijun,
            senkou_b=self.senkou_b, displacement=self.displacement
        )
        # Trim to original length (ichimoku returns extended arrays)
        self._tenkan_sen = t[:n]
        self._kijun_sen = k[:n]
        self._senkou_a = sa[:n]
        self._senkou_b_line = sb[:n]

        self._mfi = mfi(highs, lows, closes, volumes, period=self.mfi_period)
        self._atr = atr(highs, lows, closes, period=14)
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        tk = self._tenkan_sen[i]
        kj = self._kijun_sen[i]
        sa = self._senkou_a[i]
        sb = self._senkou_b_line[i]
        mfi_val = self._mfi[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        if np.isnan(tk) or np.isnan(kj) or np.isnan(sa) or np.isnan(sb) or np.isnan(mfi_val):
            return

        closes = context.get_full_close_array()
        close_price = closes[i]
        cloud_top = max(sa, sb)

        self.bars_since_last_scale += 1

        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_long(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        if side == 1 and tk < kj:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and close_price > cloud_top and tk > kj and mfi_val > 50:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        elif side == 1 and self._should_add(price, atr_val, tk, kj):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, tk, kj):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if price < self.entry_price + atr_val:
            return False
        if tk <= kj:
            return False
        return True

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
        stop_dist = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_dist <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_dist)
        margin = context.close_raw * spec.multiplier * spec.margin_rate
        if margin <= 0:
            return 0
        return max(1, min(risk_lots, int(context.equity * 0.30 / margin)))

    def _reset_state(self):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
