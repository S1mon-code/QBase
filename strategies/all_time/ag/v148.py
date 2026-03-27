import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.volatility.atr import atr
from indicators.ml.wavelet_decompose import wavelet_features
from indicators.regime.efficiency_ratio import efficiency_ratio
from indicators.trend.ichimoku import ichimoku

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV148(TimeSeriesStrategy):
    """
    策略简介：小波去噪信号 + 噪声regime过滤 + Ichimoku云确认的多空策略。

    使用指标：
    - Wavelet Features(level=4): 去噪趋势方向 + 能量比判断噪声水平
    - Efficiency Ratio(20): 低噪声regime过滤（ER高=趋势清晰）
    - Ichimoku(9,26,52): 云上方做多、云下方做空
    - ATR(14): 止损距离计算

    进场条件（做多）：去噪趋势向上 + ER>0.3(低噪声) + 价格在云上方
    进场条件（做空）：去噪趋势向下 + ER>0.3(低噪声) + 价格在云下方

    出场条件：
    - ATR 追踪止损
    - 分层止盈（3ATR/5ATR）
    - 去噪趋势方向反转

    优点：去噪后信号质量高，Ichimoku多维确认
    缺点：小波分解计算慢，Ichimoku在震荡市信号模糊
    """
    name = "ag_alltime_v148"
    warmup = 250
    freq = "daily"

    er_threshold: float = 0.3      # Optuna: 0.2-0.5
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._atr = None
        self._avg_volume = None
        self._wv_trend = None
        self._wv_energy = None
        self._er = None
        self._senkou_a = None
        self._senkou_b = None

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
        n = len(closes)

        self._atr = atr(highs, lows, closes, period=14)
        self._wv_trend, _, self._wv_energy = wavelet_features(closes, level=4)
        self._er, _ = efficiency_ratio(closes, period=20)

        tenkan_sen, kijun_sen, senkou_a, senkou_b_line, chikou = ichimoku(
            highs, lows, closes, tenkan=9, kijun=26, senkou_b=52, displacement=26
        )
        # Senkou spans are shifted forward by displacement, trim to N
        self._senkou_a = senkou_a[:n] if len(senkou_a) >= n else np.full(n, np.nan)
        self._senkou_b = senkou_b_line[:n] if len(senkou_b_line) >= n else np.full(n, np.nan)

        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return

        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        wv_trend = self._wv_trend[i]
        er_val = self._er[i]
        sa = self._senkou_a[i] if i < len(self._senkou_a) else np.nan
        sb = self._senkou_b[i] if i < len(self._senkou_b) else np.nan
        if np.isnan(wv_trend) or np.isnan(er_val) or np.isnan(sa) or np.isnan(sb):
            return

        prev_trend = self._wv_trend[i - 1] if i > 0 else np.nan
        if np.isnan(prev_trend):
            return

        trend_dir = 1 if wv_trend > prev_trend else -1
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom

        self.bars_since_last_scale += 1

        # 1. Stop loss
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

        # 2. Tiered profit-taking
        if side != 0 and self.entry_price > 0:
            if side == 1:
                profit_atr = (price - self.entry_price) / atr_val
            else:
                profit_atr = (self.entry_price - price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_5atr = True
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                close_lots = max(1, lots // 3)
                if side == 1:
                    context.close_long(lots=close_lots)
                else:
                    context.close_short(lots=close_lots)
                self._took_profit_3atr = True
                return

        # 3. Signal exit: denoised trend reversal
        if side == 1 and trend_dir == -1:
            context.close_long()
            self._reset_state()
        elif side == -1 and trend_dir == 1:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # 4. Entry
        if side == 0 and er_val > self.er_threshold:
            if trend_dir == 1 and above_cloud:
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
            elif trend_dir == -1 and below_cloud:
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

        # 5. Scale-in
        elif side != 0 and self._should_add(price, atr_val, trend_dir):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                if self.direction == 1:
                    context.buy(add_lots)
                else:
                    context.sell(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, trend_dir):
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:
            return False
        if self.direction == 1:
            if price < self.entry_price + atr_val:
                return False
            if trend_dir != 1:
                return False
        elif self.direction == -1:
            if price > self.entry_price - atr_val:
                return False
            if trend_dir != -1:
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
        self.lowest_since_entry = 999999.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
        self.direction = 0
