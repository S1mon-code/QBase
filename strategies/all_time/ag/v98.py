import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.ichimoku import ichimoku
from indicators.seasonality.seasonal_momentum import seasonal_momentum
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class IchimokuSeasonalStrategy(TimeSeriesStrategy):
    """
    策略简介：Ichimoku Cloud 趋势突破 + 季节性动量对齐的日频多空策略。

    使用指标：
    - Ichimoku(9,26,52,26): 云带突破判断趋势转换
    - Seasonal Momentum(3): 历史同期收益率预期，过滤逆季节性交易
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - 价格从云带下方突破至云上方
    - 季节性20日预期收益 > 0（顺季节性做多）

    进场条件（做空）：
    - 价格从云带上方跌至云下方
    - 季节性20日预期收益 < 0（顺季节性做空）

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 价格重回云带内部

    优点：Ichimoku提供多维度趋势信息+季节性过滤逆势交易
    缺点：季节性历史数据需足够长才有统计意义，日频信号少
    """
    name = "v98_ichimoku_seasonal"
    warmup = 300
    freq = "daily"

    tenkan: int = 9
    kijun: int = 26
    senkou_b_period: int = 52
    displacement: int = 26
    seasonal_years: int = 3
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._senkou_a = None
        self._senkou_b = None
        self._seasonal_5d = None
        self._seasonal_20d = None
        self._atr = None
        self._avg_volume = None
        self._closes = None

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
        datetimes = context.get_full_datetime_array()

        self._closes = closes
        _, _, senkou_a, senkou_b_line, _ = ichimoku(
            highs, lows, closes, self.tenkan, self.kijun,
            self.senkou_b_period, self.displacement)
        self._senkou_a = senkou_a
        self._senkou_b = senkou_b_line

        self._seasonal_5d, self._seasonal_20d = seasonal_momentum(
            closes, datetimes, self.seasonal_years)
        self._atr = atr(highs, lows, closes, period=14)

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

        atr_val = self._atr[i]
        if np.isnan(atr_val):
            return

        if i >= len(self._senkou_a) or i >= len(self._senkou_b):
            return
        sa = self._senkou_a[i]
        sb = self._senkou_b[i]
        if np.isnan(sa) or np.isnan(sb):
            return

        seasonal_20 = self._seasonal_20d[i]
        if np.isnan(seasonal_20):
            seasonal_20 = 0.0

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom

        # Check previous bar cloud position for breakout detection
        if i < 1:
            return
        prev_price = self._closes[i - 1]
        prev_sa = self._senkou_a[i - 1] if i - 1 < len(self._senkou_a) else np.nan
        prev_sb = self._senkou_b[i - 1] if i - 1 < len(self._senkou_b) else np.nan
        if np.isnan(prev_sa) or np.isnan(prev_sb):
            return
        prev_top = max(prev_sa, prev_sb)
        prev_bottom = min(prev_sa, prev_sb)
        was_above = prev_price > prev_top
        was_below = prev_price < prev_bottom

        breakout_up = above_cloud and not was_above
        breakout_down = below_cloud and not was_below

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

        # ── 3. 信号弱化退出（回到云内） ──
        if side == 1 and below_cloud:
            context.close_long()
            self._reset_state()
        elif side == -1 and above_cloud:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if breakout_up and seasonal_20 > 0:
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
            elif breakout_down and seasonal_20 < 0:
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
                    and above_cloud):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and below_cloud):
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
