import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.regime.market_state import market_state
from indicators.regime.oi_regime import oi_regime
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MarketPhaseOIRegime(TimeSeriesStrategy):
    """
    策略简介：市场阶段判断 + OI 状态分析的 Wyckoff 式策略。

    使用指标：
    - market_state(20): 多维市场状态分类 (0=quiet,1=up,2=down,3=volatile,4=breakout)
    - oi_regime(60): OI 阶段 (0=accumulation,1=markup,2=distribution,3=markdown)
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - OI regime = accumulation(0) 且 OI 正在上升
    - market_state = breakout(4) 或 trending_up(1)

    进场条件（做空）：
    - OI regime = distribution(2) 且 OI 正在上升
    - market_state = trending_down(2) 或 breakout(4) 方向下

    出场条件：
    - ATR 追踪止损 / 分层止盈
    - OI regime 转入 markdown/markup 反向
    - market_state 变为 quiet

    优点：结合持仓量结构分析，捕捉主力资金行为
    缺点：OI 数据质量影响大，日线频率信号较少
    """
    name = "v65_market_phase_oi_regime"
    warmup = 200
    freq = "daily"

    ms_period: int = 20
    oi_period: int = 60
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ms_state = None
        self._ms_conf = None
        self._oi_regime = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._ms_state, self._ms_conf = market_state(closes, volumes, oi, period=self.ms_period)
        self._oi_regime, _ = oi_regime(closes, oi, volumes, period=self.oi_period)
        self._atr = atr(highs, lows, closes, period=self.atr_period)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        ms = self._ms_state[i]
        oi_r = self._oi_regime[i]
        atr_val = self._atr[i]
        if np.isnan(ms) or np.isnan(oi_r) or np.isnan(atr_val):
            return

        ms = int(ms)
        oi_r = int(oi_r)
        self.bars_since_last_scale += 1

        # ── 1. 止损检查 ──
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing)
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return
        elif side == -1:
            self.lowest_since_entry = min(self.lowest_since_entry, price)
            trailing = self.lowest_since_entry + self.atr_stop_mult * atr_val
            self.stop_price = min(self.stop_price, trailing)
            if price >= self.stop_price:
                context.close_short()
                self._reset_state()
                return

        # ── 2. 分层止盈 ──
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
        elif side == -1 and self.entry_price > 0:
            profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                context.close_short(lots=max(1, lots // 3))
                self._took_profit_3atr = True
                return

        # ── 3. 信号弱化退出 ──
        if side == 1 and (oi_r == 3 or ms == 0):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (oi_r == 1 or ms == 0):
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场逻辑 ──
        if side == 0:
            if oi_r == 0 and ms in (1, 4):
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self._set_entry(price, price - self.atr_stop_mult * atr_val)
            elif oi_r == 2 and ms in (2, 4):
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self._set_entry(price, price + self.atr_stop_mult * atr_val)

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and oi_r in (0, 1) and ms == 1):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and oi_r in (2, 3) and ms == 2):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _set_entry(self, price, stop):
        self.entry_price = price
        self.stop_price = stop
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.position_scale = 1
        self.bars_since_last_scale = 0

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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
