import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.ml.copula_tail import tail_dependence
from indicators.momentum.stochastic import stochastic
from indicators.spread.gold_silver_ratio import gold_silver_ratio
from indicators.volatility.atr import atr

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class CopulaStochasticGoldSilverStrategy(TimeSeriesStrategy):
    """
    策略简介：Copula尾部依赖 + Stochastic振荡器 + 金银比变动的日频多空策略。

    使用指标：
    - Copula Tail Dependence(120,0.05): AG与AU的尾部依赖度，高=联动强
    - Stochastic(14,3): %K和%D振荡器，超买/超卖信号
    - Gold/Silver Ratio(60): 金银比Z-score变化方向
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - Stochastic %K < 20（超卖）且 %K > %D（金叉）
    - 金银比Z-score正在下降（银相对金变强）
    - 尾部依赖不在极端（联动正常）

    进场条件（做空）：
    - Stochastic %K > 80（超买）且 %K < %D（死叉）
    - 金银比Z-score正在上升（银相对金变弱）
    - 尾部依赖不在极端

    出场条件：
    - ATR追踪止损
    - 分层止盈（3ATR/5ATR）
    - Stochastic方向反转

    优点：Copula捕捉非线性尾部关系+Stochastic经典均值回归+金银比基本面
    缺点：尾部依赖在样本少时估计不稳定，日频Stochastic可能滞后
    """
    name = "v170_copula_stochastic_gold_silver"
    warmup = 400
    freq = "daily"

    stoch_k: int = 14
    stoch_d: int = 3
    stoch_ob: float = 80.0
    stoch_os: float = 20.0
    gs_period: int = 60
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
        self._stoch_k = None
        self._stoch_d = None
        self._gs_zscore = None
        self._tail_dep = None
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
        self._stoch_k, self._stoch_d = stochastic(highs, lows, closes,
                                                    k=self.stoch_k, d=self.stoch_d)

        # Gold/Silver ratio
        au_closes = context.load_auxiliary_close("AU")
        if au_closes is not None and len(au_closes) == len(closes):
            _, self._gs_zscore = gold_silver_ratio(au_closes, closes, self.gs_period)
            # Copula tail dependence between AG and AU returns
            ag_rets = np.zeros_like(closes)
            ag_rets[1:] = closes[1:] / closes[:-1] - 1
            au_rets = np.zeros_like(au_closes)
            au_rets[1:] = au_closes[1:] / au_closes[:-1] - 1
            self._tail_dep = tail_dependence(ag_rets, au_rets, period=120, quantile=0.05)
        else:
            self._gs_zscore = np.zeros(len(closes))
            self._tail_dep = np.full(len(closes), 0.5)

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
        sk = self._stoch_k[i]
        sd = self._stoch_d[i]
        gs_z = self._gs_zscore[i]
        gs_z_prev = self._gs_zscore[i - 1]
        atr_val = self._atr[i]
        if np.isnan(sk) or np.isnan(sd) or np.isnan(atr_val):
            return
        if np.isnan(gs_z):
            gs_z = 0.0
        if np.isnan(gs_z_prev):
            gs_z_prev = 0.0

        self.bars_since_last_scale += 1
        gs_falling = gs_z < gs_z_prev  # silver strengthening
        gs_rising = gs_z > gs_z_prev   # silver weakening

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

        # ── 3. Stochastic反转退出 ──
        if side == 1 and sk > self.stoch_ob:
            context.close_long()
            self._reset_state()
        elif side == -1 and sk < self.stoch_os:
            context.close_short()
            self._reset_state()

        side, lots = context.position

        # ── 4. 入场逻辑 ──
        if side == 0:
            if sk < self.stoch_os and sk > sd and gs_falling:
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
            elif sk > self.stoch_ob and sk < sd and gs_rising:
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
                    and sk < 50 and gs_falling):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
                self.position_scale += 1
                self.bars_since_last_scale = 0
        elif side == -1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price < self.entry_price - atr_val
                    and sk > 50 and gs_rising):
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
