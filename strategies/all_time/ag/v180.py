import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.regime_transition_matrix import transition_features
from indicators.regime.market_state import market_state
from indicators.trend.ichimoku import ichimoku
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class RegimeTransitionIchimokuGoldSilverStrategy(TimeSeriesStrategy):
    """
    策略简介：Regime Transition检测 + Ichimoku方向 + 金银比确认的日频多空策略。

    使用指标：
    - Market State(20) + Transition Features: regime变化检测，transition时入场
    - Ichimoku(9,26,52,26): 趋势方向确认（云上/云下）
    - Gold/Silver Ratio(60): Z-score确认银的相对强度
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Regime刚发生transition（前后state不同）
    - Ichimoku价格在云上方且Tenkan > Kijun
    - 金银比Z-score < 1.0（银未过高估）

    进场条件（做空）：
    - Regime刚发生transition
    - Ichimoku价格在云下方且Tenkan < Kijun
    - 金银比Z-score > -1.0（银未过低估）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Ichimoku云方向反转

    优点：在regime转换时入场捕捉新趋势起点+Ichimoku多维确认+金银比基本面
    缺点：regime transition可能为假转换，Ichimoku参数固定可能不适配
    """
    name = "v180_regime_transition_ichimoku_gold_silver"
    warmup = 400
    freq = "daily"

    ms_period: int = 20
    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    ichi_senkou_b: int = 52
    gs_period: int = 60
    gs_zscore_thresh: float = 1.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ms = None
        self._tenkan = None
        self._kijun = None
        self._senkou_a = None
        self._senkou_b = None
        self._gs_zscore = None
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
        self.direction = 0

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()
        oi = context.get_full_oi_array()

        self._atr = atr(highs, lows, closes, period=14)
        self._ms, _ = market_state(closes, volumes, oi, period=self.ms_period)

        # Ichimoku
        self._tenkan, self._kijun, self._senkou_a, self._senkou_b, _ = ichimoku(
            highs, lows, closes, self.ichi_tenkan, self.ichi_kijun,
            self.ichi_senkou_b, self.ichi_kijun)

        # Gold/Silver ratio
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            _, self._gs_zscore = gold_silver_ratio(au_closes, closes, self.gs_period)
        else:
            self._gs_zscore = np.zeros(len(closes))

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

        if i < 2:
            return
        ms_now = self._ms[i]
        ms_prev = self._ms[i - 1]
        tenkan = self._tenkan[i]
        kijun = self._kijun[i]
        spa = self._senkou_a[i]
        spb = self._senkou_b[i]
        gs_z = self._gs_zscore[i]
        atr_val = self._atr[i]
        if np.isnan(ms_now) or np.isnan(ms_prev) or np.isnan(tenkan) or np.isnan(kijun):
            return
        if np.isnan(spa) or np.isnan(spb) or np.isnan(atr_val):
            return
        if np.isnan(gs_z):
            gs_z = 0.0

        self.bars_since_last_scale += 1
        regime_changed = ms_now != ms_prev
        cloud_top = max(spa, spb)
        cloud_bot = min(spa, spb)

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

        # ── 3. Ichimoku云反转退出 ──
        if side == 1 and price < cloud_bot:
            context.close_long()
            self._reset_state()
        elif side == -1 and price > cloud_top:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑（regime transition时） ──
        if side == 0 and regime_changed:
            if price > cloud_top and tenkan > kijun and gs_z < self.gs_zscore_thresh:
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
            elif price < cloud_bot and tenkan < kijun and gs_z > -self.gs_zscore_thresh:
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

        # ── 5. 加仓逻辑 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and price > cloud_top and tenkan > kijun):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and price < cloud_bot and tenkan < kijun):
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
