import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.gaussian_mixture_regime import gmm_regime
from indicators.volatility.nr7 import nr7
from indicators.spread.cross_momentum import cross_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class GaussianNR7CrossMomentumStrategy(TimeSeriesStrategy):
    """
    策略简介：Gaussian Process代理（GMM regime）+ NR7窄幅突破 + 跨品种动量的日频多空策略。

    使用指标：
    - GMM Regime(120,3): 高斯混合模型状态识别，用于判断当前市场环境
    - NR7: 最近7天最窄幅度，突破=波动率扩张信号
    - Cross Momentum(AG vs AU, 20): 银vs金相对动量
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - NR7触发（波动率压缩到极致）
    - 价格突破NR7日高点
    - AG vs AU cross momentum > 0（银跑赢金）

    进场条件（做空）：
    - NR7触发
    - 价格跌破NR7日低点
    - AG vs AU cross momentum < 0（银跑输金）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Cross momentum方向反转

    优点：NR7精确捕捉波动率压缩后的爆发+跨品种动量增加信息维度
    缺点：NR7信号稀疏，假突破风险，GMM在非稳态市场不稳定
    """
    name = "v166_gp_nr7_cross_momentum"
    warmup = 400
    freq = "daily"

    cm_period: int = 20
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._nr7 = None
        self._cross_mom = None
        self._atr = None
        self._avg_volume = None
        self._highs = None
        self._lows = None

    def on_init(self, context):
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
        self.nr7_high = 0.0
        self.nr7_low = 0.0
        self.nr7_triggered = False

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._highs = highs
        self._lows = lows
        self._atr = atr(highs, lows, closes, period=14)
        self._nr7 = nr7(highs, lows)

        # Cross momentum: AG vs AU
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            self._cross_mom = cross_momentum(closes, au_closes, self.cm_period)
        else:
            self._cross_mom = np.zeros(len(closes))

        window = 20
        self._avg_volume = np.full_like(volumes, np.nan)
        for idx in range(window, len(volumes)):
            self._avg_volume[idx] = np.mean(volumes[idx-window:idx])

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        if i < 2:
            return
        nr7_prev = self._nr7[i - 1]
        cm_val = self._cross_mom[i]
        atr_val = self._atr[i]
        if np.isnan(atr_val):
            return
        if np.isnan(cm_val):
            cm_val = 0.0

        self.bars_since_last_scale += 1

        # Track NR7 breakout levels
        if not np.isnan(nr7_prev) and nr7_prev > 0.5:
            self.nr7_high = self._highs[i - 1]
            self.nr7_low = self._lows[i - 1]
            self.nr7_triggered = True

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

        # ── 3. Cross momentum反转退出 ──
        if side == 1 and cm_val < 0:
            context.close_long()
            self._reset_state()
        elif side == -1 and cm_val > 0:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑（NR7突破） ──
        if side == 0 and self.nr7_triggered:
            if price > self.nr7_high and cm_val > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = 1
                    self.nr7_triggered = False
            elif price < self.nr7_low and cm_val < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
                    self.direction = -1
                    self.nr7_triggered = False

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and cm_val > 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and cm_val < 0):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.sell(add)
                self.position_scale += 1
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
        self.direction = 0
        self.nr7_triggered = False
