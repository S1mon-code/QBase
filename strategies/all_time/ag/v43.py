import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.boosting_signal import gradient_boost_signal
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr
from indicators.momentum.rsi import rsi

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class BoostingGoldSilverStrategy(TimeSeriesStrategy):
    """
    策略简介：Gradient Boosting方向预测 + 金银比确认的多空策略。

    使用指标：
    - Gradient Boosting Signal: 滚动训练GBT分类器预测方向概率
    - Gold/Silver Ratio: Au/Ag比值的z-score，高z=银被低估
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Boosting signal > 0.6（预测上涨概率高）
    - 金银比z-score > 0（银相对金被低估，有回归潜力）

    进场条件（做空）：
    - Boosting signal < 0.4（预测下跌概率高）
    - 金银比z-score < 0（银相对金被高估）

    出场条件：
    - ATR追踪止损 / 分层止盈 / 信号反转

    优点：ML模型自适应市场变化，金银比提供基本面支撑
    缺点：需要AU辅助数据，Boosting可能过拟合
    """
    name = "v43_boosting_gold_silver"
    warmup = 400
    freq = "daily"

    boost_period: int = 120
    gsr_period: int = 60
    boost_bull_thresh: float = 0.6
    boost_bear_thresh: float = 0.4
    atr_period: int = 14
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._boost_signal = None
        self._gsr_zscore = None
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
        n = len(closes)

        # Build features for boosting
        rsi_arr = rsi(closes, 14)
        atr_arr = atr(highs, lows, closes, 14)
        returns = np.full(n, np.nan)
        returns[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-9)
        vol_ratio = volumes / np.maximum(np.convolve(volumes, np.ones(20)/20, mode='same'), 1e-9)
        features = np.column_stack([rsi_arr, atr_arr / np.maximum(closes, 1e-9), returns, vol_ratio])

        self._boost_signal, _ = gradient_boost_signal(
            closes, features, period=self.boost_period)

        # Gold/Silver ratio - load AU auxiliary data
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == n:
            _, self._gsr_zscore = gold_silver_ratio(au_closes, closes, period=self.gsr_period)
        else:
            self._gsr_zscore = np.full(n, 0.0)

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

        boost_val = self._boost_signal[i]
        gsr_z = self._gsr_zscore[i]
        atr_val = self._atr[i]
        if np.isnan(boost_val) or np.isnan(gsr_z) or np.isnan(atr_val) or atr_val <= 0:
            return

        self.bars_since_last_scale += 1

        # ── 1. 止损 ──
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
        if side == 1 and (boost_val < 0.5 or gsr_z < -0.5):
            context.close_long()
            self._reset_state()
            return
        elif side == -1 and (boost_val > 0.5 or gsr_z > 0.5):
            context.close_short()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0:
            if boost_val > self.boost_bull_thresh and gsr_z > 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.buy(base_lots)
                    self.entry_price = price
                    self.stop_price = price - self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0
            elif boost_val < self.boost_bear_thresh and gsr_z < 0:
                base_lots = self._calc_lots(context, atr_val)
                if base_lots > 0:
                    context.sell(base_lots)
                    self.entry_price = price
                    self.stop_price = price + self.atr_stop_mult * atr_val
                    self.highest_since_entry = price
                    self.lowest_since_entry = price
                    self.position_scale = 1
                    self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and boost_val > self.boost_bull_thresh):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and boost_val < self.boost_bear_thresh):
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
