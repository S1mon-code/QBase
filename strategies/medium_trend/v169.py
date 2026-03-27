import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import conftest
from strategies.all_time.ag.strategy_utils import fast_avg_volume

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from alphaforge.data.contract_specs import ContractSpecManager

_SPEC_MANAGER = ContractSpecManager()
from indicators.volatility.atr import atr

from indicators.ml.hmm_regime import hmm_regime
from indicators.ml.kmeans_regime import kmeans_regime
from indicators.trend.adx import adx as calc_adx

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class StrategyV169(TimeSeriesStrategy):
    """
    策略简介：HMM+K-Means双regime+ADX趋势做多（4h）。
    进场条件（做多）：核心信号满足阈值
    出场条件：ATR追踪止损 / 分层止盈 / 策略特定退出
    优点：针对性信号组合，低冗余
    缺点：需在足够历史数据验证
    """
    name = "mt_v169"
    warmup = 500
    freq = "4h"
    atr_stop_mult: float = 3.0

    def __init__(self):
        super().__init__()
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
        self._atr = atr(highs, lows, closes, period=14)
        self._adx = calc_adx(highs,lows,closes,period=14)
        self._hmm_lab,_,_ = hmm_regime(closes, n_states=3, period=252)
        from indicators.momentum.rsi import rsi
        features = np.column_stack([rsi(closes,14), self._adx, self._atr])
        self._km_lab, _ = kmeans_regime(features, period=120, n_clusters=3)
        n=len(closes); safe=np.maximum(closes,1e-9); rets=np.diff(np.log(safe),prepend=np.nan)
        self._hb=np.full(n,np.nan); self._kb=np.full(n,np.nan)
        for idx in range(60,n):
            sr=rets[idx-60:idx]
            for arr,ba in [(self._hmm_lab,self._hb),(self._km_lab,self._kb)]:
                sl=arr[idx-60:idx]; v=~(np.isnan(sl)|np.isnan(sr))
                if v.sum()>10:
                    bm,bl=-np.inf,-1
                    for lb in range(3):
                        m=(sl==lb)&v
                        if m.sum()>0:
                            mn=np.mean(sr[m])
                            if mn>bm: bm,bl=mn,lb
                    c=arr[idx]
                    if not np.isnan(c) and bm>0: ba[idx]=1.0 if int(c)==bl else 0.0
                    else: ba[idx]=0.0
        self._avg_volume = fast_avg_volume(volumes, 20)

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position
        if context.is_rollover:
            return
        if not np.isnan(self._avg_volume[i]) and context.volume < self._avg_volume[i] * 0.1:
            return
        atr_val = self._atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            return
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

        if side == 1 and not np.isnan(self._hb[i]) and self._hb[i]==0:
            context.close_long()
            self._reset_state()
            return

        if side == 0 and not np.isnan(self._hb[i]) and self._hb[i]==1 and not np.isnan(self._kb[i]) and self._kb[i]==1 and not np.isnan(self._adx[i]) and self._adx[i]>20:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0
        elif side == 1 and self.position_scale < MAX_SCALE and self.bars_since_last_scale >= 10 and price > self.entry_price + atr_val:
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _calc_add_lots(self, base_lots):
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
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
