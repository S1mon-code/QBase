import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.ridge_forecast import rolling_ridge
from indicators.trend.ichimoku import ichimoku
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class LSTMIchimokuGoldSilverStrategy(TimeSeriesStrategy):
    """
    策略简介：LSTM代理（Ridge滚动预测）+ Ichimoku云方向 + 金银比Z-score的日频多空策略。

    使用指标：
    - Rolling Ridge(120,5): 替代LSTM的线性ML预测，forecast>0=看多
    - Ichimoku(9,26,52,26): 价格在云上=多头，价格在云下=空头
    - Gold/Silver Ratio(60): Z-score偏低=银相对金被低估=做多银

    进场条件（做多）：
    - Ridge forecast > 0（ML预测上涨）
    - 价格 > Senkou Span A 和 Senkou Span B（在云上方）
    - 金银比Z-score < 1.0（银未被高估）

    进场条件（做空）：
    - Ridge forecast < 0（ML预测下跌）
    - 价格 < Senkou Span A 和 Senkou Span B（在云下方）
    - 金银比Z-score > -1.0（银未被低估）

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Ichimoku云方向反转

    优点：ML预测+传统技术+跨品种三维信号确认，低相关度
    缺点：Ridge对非线性模式捕捉有限，金银比数据依赖
    """
    name = "v161_lstm_ichimoku_gold_silver"
    warmup = 360
    freq = "daily"

    ridge_period: int = 120
    ridge_horizon: int = 5
    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    ichi_senkou_b: int = 52
    gs_period: int = 60
    gs_zscore_thresh: float = 1.0
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._ridge_pred = None
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

        self._atr = atr(highs, lows, closes, period=14)

        # ML features: returns + volatility
        rets = np.zeros_like(closes)
        rets[1:] = closes[1:] / closes[:-1] - 1
        vol20 = np.full_like(closes, np.nan)
        for idx in range(20, len(closes)):
            vol20[idx] = np.std(rets[idx-20:idx])
        features = np.column_stack([rets, vol20])
        features = np.nan_to_num(features, nan=0.0)
        self._ridge_pred, _ = rolling_ridge(closes, features, period=self.ridge_period,
                                             forecast_horizon=self.ridge_horizon)

        # Ichimoku
        tenkan, kijun, self._senkou_a, self._senkou_b, chikou = ichimoku(
            highs, lows, closes, self.ichi_tenkan, self.ichi_kijun,
            self.ichi_senkou_b, self.ichi_kijun)

        # Gold/Silver ratio
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            self._gs_ratio, self._gs_zscore = gold_silver_ratio(au_closes, closes, self.gs_period)
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

        ridge_val = self._ridge_pred[i]
        spa = self._senkou_a[i]
        spb = self._senkou_b[i]
        gs_z = self._gs_zscore[i]
        atr_val = self._atr[i]
        if np.isnan(ridge_val) or np.isnan(spa) or np.isnan(spb) or np.isnan(atr_val):
            return
        if np.isnan(gs_z):
            gs_z = 0.0

        self.bars_since_last_scale += 1
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

        # ── 3. 信号反转退出 ──
        if side == 1 and price < cloud_bot:
            context.close_long()
            self._reset_state()
        elif side == -1 and price > cloud_top:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if ridge_val > 0 and price > cloud_top and gs_z < self.gs_zscore_thresh:
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
            elif ridge_val < 0 and price < cloud_bot and gs_z > -self.gs_zscore_thresh:
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
                    and ridge_val > 0 and price > cloud_top):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and ridge_val < 0 and price < cloud_bot):
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
