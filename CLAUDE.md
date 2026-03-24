# QBase — Agent 开发指南

QBase 是量化策略研究工作区。指标库 + 按品种组织的策略开发，回测统一用 AlphaForge。

## 项目结构

```
QBase/
├── indicators/              # 100个量价指标（纯numpy函数）
│   ├── momentum/            # 25个动量/振荡类
│   ├── trend/               # 25个趋势类
│   ├── volatility/          # 25个波动率类
│   └── volume/              # 25个成交量/持仓类
├── strategies/              # 按市场状态 → 品种组织
│   ├── strong_trend/        # 强趋势（涨幅>100%，6个月+）
│   ├── medium_trend/        # 中趋势（涨幅20-80%，2-4个月）
│   ├── weak_trend/          # 弱趋势（小幅趋势）
│   └── mean_reversion/      # 震荡（区间震荡行情）
│       └── <symbol>/        # 如 ag/, au/, I/
│           ├── v1.py        # 策略文件: v{版本}.py
│           └── portfolio/   # 该品种在该状态下的组合方案
├── screener/                # 品种筛选器
├── fundamental/             # 基本面量化（预留）
├── research_log/            # 实验记录
├── tests/                   # 指标单元测试
├── config.yaml              # AlphaForge路径等配置
├── conftest.py              # sys.path 配置（pytest自动加载）
└── run.sh                   # 策略运行入口
```

## 核心规则

### 1. 策略指标限制：最多 3 个

每个策略最多使用 **3 个指标**的组合。不要堆砌指标。

选择指标时的思路：
- **趋势策略**: 1个趋势判断 + 1个动量确认 + 1个波动率过滤（如 ADX + ROC + ATR）
- **震荡策略**: 1个振荡器 + 1个波动率 + 1个量价确认（如 RSI + Bollinger + MFI）
- **突破策略**: 1个通道/区间 + 1个波动率 + 1个量能确认（如 Donchian + NR7 + Volume Spike）

指标可以来自任意分类，自由组合。关键是 3 个指标之间要有互补性，不要选同类指标（如同时用 RSI + Stochastic + Williams %R 是错误的）。

### 2. 策略目录结构：按市场状态分类

策略按**市场状态**分四大类，策略本身不分品种 — 同一个趋势策略可以跑在任何品种上：

```
strategies/
├── strong_trend/            # 强趋势策略（捕捉涨幅>100%的大行情）
│   ├── v1.py                # 策略 v1（不同指标组合）
│   ├── v2.py                # 策略 v2
│   └── portfolio/           # 强趋势策略组合方案
│       ├── weights.json
│       └── README.md
├── medium_trend/            # 中趋势策略（涨幅20-80%，2-4个月）
│   ├── v1.py
│   └── portfolio/
├── weak_trend/              # 弱趋势策略（小幅趋势、波段）
│   ├── v1.py
│   └── portfolio/
└── mean_reversion/          # 震荡策略（区间震荡、均值回归）
    ├── v1.py
    └── portfolio/
```

**市场状态定义：**

| 状态 | 涨幅范围 | 持续时间 | 参考数据 |
|------|---------|---------|---------|
| 强趋势 | >100% | 6个月+ | trend/RALLIES.md |
| 中趋势 | 20-80% | 2-4个月 | trend/MEDIUM_TRENDS.md |
| 弱趋势 | 5-20% | 1-3个月 | 待整理 |
| 震荡 | ±5%内 | 不定 | 待整理 |

**策略是品种无关的。** 运行时通过 `--symbols` 指定品种：
```bash
# 同一个强趋势策略，跑不同品种
./run.sh strategies/strong_trend/v1.py --symbols AG --freq daily --start 2022
./run.sh strategies/strong_trend/v1.py --symbols J --freq daily --start 2016
./run.sh strategies/strong_trend/v1.py --symbols ZC --freq daily --start 2021
```

### 3. portfolio 文件夹

每个市场状态下的 `portfolio/` 是该状态的**终极运行方案**：
- 包含该状态下多个策略版本的加权组合
- 权重通过多品种回测研究确定
- `weights.json` 格式：

```json
{
  "name": "strong_trend_portfolio_v1",
  "regime": "strong_trend",
  "strategies": {
    "v1": {"weight": 0.5, "sharpe": 1.2, "max_dd": -0.15},
    "v2": {"weight": 0.3, "sharpe": 0.9, "max_dd": -0.12},
    "v3": {"weight": 0.2, "sharpe": 0.7, "max_dd": -0.18}
  },
  "method": "risk_parity",
  "backtest_period": "2020-01-01 to 2025-12-31",
  "portfolio_sharpe": 1.8
}
```

## 回测：统一用 AlphaForge

所有回测通过 AlphaForge 执行。AlphaForge 位于 `~/Desktop/AlphaForge/`。

### 策略基类

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # QBase root
import conftest  # 配置 AlphaForge 路径

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.trend.adx import adx
from indicators.momentum.roc import rate_of_change
from indicators.volatility.atr import atr

class MyStrategy(TimeSeriesStrategy):
    name = "my_strategy"
    warmup = 60

    def on_init(self, context):
        self.entry_price = 0.0

    def on_bar(self, context):
        closes = context.get_close_array(60)
        highs = context.get_high_array(60)
        lows = context.get_low_array(60)
        price = context.current_bar.close_raw
        side, lots = context.position

        # 计算指标（最多3个）
        adx_val = adx(highs, lows, closes, period=14)[-1]
        roc_val = rate_of_change(closes, 20)[-1]
        atr_val = atr(highs, lows, closes, period=14)[-1]

        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        # 交易逻辑...
```

### AlphaForge Context API

```python
# 市场数据
context.current_bar.close_raw     # 原始收盘价
context.current_bar.high / low / open / volume / oi
context.datetime                   # 当前bar时间
context.symbol                     # 品种代码

# 历史数据
context.get_close_array(n)         # 最近n根close (np.ndarray)
context.get_high_array(n)
context.get_low_array(n)
context.get_open_array(n)
context.get_volume_array(n)

# 持仓
side, lots = context.position      # side: 1=多, -1=空, 0=无
context.equity                     # 当前权益
context.available_cash             # 可用资金

# 下单（下一bar的open执行）
context.buy(lots)                  # 开多
context.sell(lots)                 # 开空
context.close_long()               # 平多
context.close_short()              # 平空
```

### 运行回测

```bash
# 单品种
./run.sh strategies/ag/trend_v1.py --symbols AG --freq daily --start 2022

# 多品种测试
./run.sh strategies/ag/trend_v1.py --symbols AG,AU,CU --freq daily --start 2020

# 全市场扫描
./run.sh --scan strategies/ag/trend_v1.py --top 20 --start 2020
```

### 训练/测试分割

```python
# 训练集开发策略参数
af run strategy.py --symbols AG --start 2013 --end 2021

# 测试集验证（不改参数）
af run strategy.py --symbols AG --start 2022 --end 2026
```

## 指标库（100个）

### Momentum（25个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| Rate of Change | `roc.py` | `rate_of_change(closes, period=12)` |
| RSI | `rsi.py` | `rsi(closes, period=14)` |
| MACD | `macd.py` | `macd(closes, fast=12, slow=26, signal=9)` → (line, signal, hist) |
| CCI | `cci.py` | `cci(highs, lows, closes, period=20)` |
| Stochastic | `stochastic.py` | `stochastic(highs, lows, closes, k=14, d=3)` → (%K, %D) |
| KDJ | `stochastic.py` | `kdj(highs, lows, closes, period=9, k=3, d=3)` → (K, D, J) |
| Williams %R | `williams_r.py` | `williams_r(highs, lows, closes, period=14)` |
| KAMA | `kama.py` | `kama(closes, period=10, fast_sc=2, slow_sc=30)` |
| TSMOM | `tsmom.py` | `tsmom(closes, lookback=252, vol_lookback=60)` |
| Momentum Accel | `momentum_accel.py` | `momentum_acceleration(closes, fast=10, slow=20)` |
| Awesome Oscillator | `awesome_oscillator.py` | `ao(highs, lows, fast=5, slow=34)` |
| Chande MO | `cmo.py` | `cmo(closes, period=14)` |
| Detrended Price | `dpo.py` | `dpo(closes, period=20)` |
| PPO | `ppo.py` | `ppo(closes, fast=12, slow=26, signal=9)` → (ppo, signal, hist) |
| Ultimate Osc | `ultimate_oscillator.py` | `ultimate_oscillator(highs, lows, closes, 7, 14, 28)` |
| Coppock Curve | `coppock.py` | `coppock(closes, wma=10, roc_long=14, roc_short=11)` |
| TSI | `tsi.py` | `tsi(closes, long=25, short=13, signal=7)` → (tsi, signal) |
| KST | `kst.py` | `kst(closes, ...)` → (kst, signal) |
| RVI | `rvi.py` | `rvi(opens, highs, lows, closes, period=10)` → (rvi, signal) |
| Elder Force | `elder_force.py` | `elder_force_index(closes, volumes, period=13)` |
| Schaff Trend | `schaff_trend.py` | `schaff_trend_cycle(closes, period=10, fast=23, slow=50)` |
| Connors RSI | `connors_rsi.py` | `connors_rsi(closes, rsi=3, streak=2, pct_rank=100)` |
| Fisher Transform | `fisher_transform.py` | `fisher_transform(highs, lows, period=10)` → (fisher, trigger) |
| Stochastic RSI | `stoch_rsi.py` | `stoch_rsi(closes, rsi=14, stoch=14, k=3, d=3)` → (%K, %D) |
| Choppiness Index | `chop_zone.py` | `choppiness_index(highs, lows, closes, period=14)` |
| Ergodic | `ergodic.py` | `ergodic(closes, short=5, long=20, signal=5)` → (erg, signal) |

### Trend（25个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| ADX | `adx.py` | `adx(highs, lows, closes, period=14)` |
| Supertrend | `supertrend.py` | `supertrend(highs, lows, closes, period=10, mult=3.0)` → (line, dir) |
| Aroon | `aroon.py` | `aroon(highs, lows, period=25)` → (up, down, osc) |
| EMA | `ema.py` | `ema(data, period)` / `ema_cross(data, fast, slow)` |
| SMA | `sma.py` | `sma(data, period)` |
| Donchian | `donchian.py` | `donchian(highs, lows, period=20)` → (upper, lower, mid) |
| Keltner | `keltner.py` | `keltner(highs, lows, closes, ema=20, atr=10, mult=1.5)` → (u, m, l) |
| EMA Ribbon | `ema_ribbon.py` | `ema_ribbon(data, periods)` / `ema_ribbon_signal(data, periods)` |
| Fractal | `fractal.py` | `fractal_high(highs)` / `fractal_low(lows)` / `fractal_levels(h, l)` |
| Higher Low | `higher_low.py` | `higher_lows(lows, lookback=20)` / `lower_highs(highs, lookback=20)` |
| Ichimoku | `ichimoku.py` | `ichimoku(highs, lows, closes, 9, 26, 52, 26)` → 5 components |
| Parabolic SAR | `psar.py` | `psar(highs, lows, af_start=0.02, af_step=0.02, af_max=0.2)` → (sar, dir) |
| DEMA | `dema.py` | `dema(data, period)` |
| TEMA | `tema.py` | `tema(data, period)` |
| HMA | `hma.py` | `hma(data, period)` |
| VWMA | `vwma.py` | `vwma(closes, volumes, period=20)` |
| Linear Regression | `linear_regression.py` | `linear_regression(data, period)` / `linear_regression_slope()` / `r_squared()` |
| ALMA | `alma.py` | `alma(data, period=9, offset=0.85, sigma=6)` |
| T3 | `t3.py` | `t3(data, period=5, volume_factor=0.7)` |
| McGinley | `mcginley.py` | `mcginley_dynamic(data, period=14)` |
| Vortex | `vortex.py` | `vortex(highs, lows, closes, period=14)` → (VI+, VI-) |
| Mass Index | `mass_index.py` | `mass_index(highs, lows, ema=9, sum=25)` |
| RWI | `rwi.py` | `rwi(highs, lows, closes, period=14)` → (rwi_high, rwi_low) |
| ZLEMA | `zlema.py` | `zlema(data, period)` |
| Trend Intensity | `trend_intensity.py` | `trend_intensity(closes, period=14)` |

### Volatility（25个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| ATR | `atr.py` | `atr(highs, lows, closes, period=14)` |
| Bollinger Bands | `bollinger.py` | `bollinger_bands(closes, period=20, std=2.0)` → (u, m, l) |
| Historical Vol | `historical_vol.py` | `historical_volatility(closes, period=20, ann=252)` |
| TTM Squeeze | `ttm_squeeze.py` | `ttm_squeeze(h, l, c, bb=20, bb_m=2.0, kc=20, kc_m=1.5)` → (squeeze, mom) |
| Hurst | `hurst.py` | `hurst_exponent(data, max_lag=20)` |
| Entropy | `entropy.py` | `entropy(closes, period=20, bins=10)` |
| VoV | `vov.py` | `vov(closes, vol_period=20, vov_period=20)` |
| NR7 | `nr7.py` | `nr7(highs, lows)` / `nr4(highs, lows)` |
| Range Expansion | `range_expansion.py` | `range_expansion(highs, lows, closes, period=14)` |
| NATR | `natr.py` | `natr(highs, lows, closes, period=14)` |
| Chaikin Volatility | `chaikin_vol.py` | `chaikin_volatility(highs, lows, ema=10, roc=10)` |
| Ulcer Index | `ulcer_index.py` | `ulcer_index(closes, period=14)` |
| Garman-Klass | `garman_klass.py` | `garman_klass(opens, highs, lows, closes, period=20)` |
| Parkinson | `parkinson.py` | `parkinson(highs, lows, period=20)` |
| Rogers-Satchell | `rogers_satchell.py` | `rogers_satchell(opens, highs, lows, closes, period=20)` |
| Yang-Zhang | `yang_zhang.py` | `yang_zhang(opens, highs, lows, closes, period=20)` |
| Realized Variance | `realized_variance.py` | `realized_variance(closes, period=20)` / `realized_volatility(...)` |
| ADR | `adr.py` | `average_day_range(highs, lows, period=14)` / `adr_percent(...)` |
| Rolling Std | `std_dev.py` | `rolling_std(data, period=20)` / `z_score(data, period=20)` |
| True Range | `true_range.py` | `true_range(highs, lows, closes)` |
| Chandelier Exit | `chandelier_exit.py` | `chandelier_exit(highs, lows, closes, period=22, mult=3.0)` → (long, short) |
| Keltner Width | `keltner_width.py` | `keltner_width(highs, lows, closes, ema=20, atr=10, mult=1.5)` |
| Volatility Ratio | `vol_ratio.py` | `volatility_ratio(highs, lows, closes, period=14)` |
| ATR Ratio | `chop.py` | `atr_ratio(highs, lows, closes, short=5, long=20)` |
| Close-to-Close Vol | `close_to_close_vol.py` | `close_to_close_vol(closes, period=20, ann=252)` |

### Volume（25个）

| 指标 | 文件 | 函数签名 |
|------|------|---------|
| OBV | `obv.py` | `obv(closes, volumes)` |
| MFI | `mfi.py` | `mfi(highs, lows, closes, volumes, period=14)` |
| VWAP | `vwap.py` | `vwap(highs, lows, closes, volumes)` |
| OI Divergence | `oi_divergence.py` | `oi_divergence(closes, oi, period=20)` |
| OI Momentum | `oi_momentum.py` | `oi_momentum(oi, period=20)` / `oi_sentiment(c, oi, v, period=20)` |
| Volume Spike | `volume_spike.py` | `volume_spike(v, period=20, threshold=2.0)` / `volume_climax(...)` / `volume_dry_up(...)` |
| Close Location | `close_location.py` | `close_location(highs, lows, closes)` |
| Price Density | `price_density.py` | `price_density(highs, lows, closes, period=20)` |
| VPT | `volume_price_trend.py` | `vpt(closes, volumes)` |
| A/D Line | `ad_line.py` | `ad_line(highs, lows, closes, volumes)` |
| CMF | `cmf.py` | `cmf(highs, lows, closes, volumes, period=20)` |
| EMV | `emv.py` | `emv(highs, lows, volumes, period=14)` |
| Force Index | `force_index.py` | `force_index(closes, volumes, period=13)` |
| Klinger | `klinger.py` | `klinger(highs, lows, closes, volumes, fast=34, slow=55, signal=13)` → (kvo, signal) |
| NVI/PVI | `nvi.py` | `nvi(closes, volumes)` / `pvi(closes, volumes)` |
| Volume Oscillator | `volume_oscillator.py` | `volume_oscillator(volumes, fast=5, slow=20)` |
| VROC | `vroc.py` | `vroc(volumes, period=14)` |
| Twiggs Money Flow | `twiggs.py` | `twiggs_money_flow(highs, lows, closes, volumes, period=21)` |
| WAD | `wad.py` | `wad(highs, lows, closes)` |
| Volume Profile | `volume_profile.py` | `volume_profile(closes, volumes, bins=20)` / `poc(closes, volumes, period=20)` |
| Volume RSI | `volume_weighted_rsi.py` | `volume_rsi(closes, volumes, period=14)` |
| TVI | `trade_volume_index.py` | `tvi(closes, volumes, min_tick=0.5)` |
| Volume Momentum | `volume_momentum.py` | `volume_momentum(volumes, period=14)` / `relative_volume(volumes, period=20)` |
| Money Flow | `money_flow.py` | `money_flow(h, l, c, v)` / `money_flow_ratio(h, l, c, v, period=14)` |
| Demand Index | `demand_index.py` | `demand_index(highs, lows, closes, volumes, period=14)` |

## 品种筛选器

```python
from screener.scanner import scan

# 找趋势最强的品种（训练用）
trend_top = scan(mode="trend", start="2024-01-01", end="2025-01-01", top_n=10)

# 找震荡特征明显的品种
mr_top = scan(mode="mean_reversion", start="2024-01-01", top_n=10)

# 找波动率收缩的品种（潜在突破）
breakout_top = scan(mode="breakout", start="2024-01-01", top_n=10)
```

## 开发工作流

1. **选品种** — 用 screener 或直接指定
2. **选策略类型** — trend / mean_reversion / breakout
3. **选 3 个指标** — 从 100 个中选，确保互补（不同分类）
4. **开发策略** — 继承 `TimeSeriesStrategy`，用 QBase 指标
5. **回测验证** — AlphaForge 训练集 + 测试集
6. **记录结果** — 写入 `research_log/`
7. **组合优化** — 多策略加权 → `portfolio/weights.json`

## 策略质量门槛

策略必须通过以下标准才能进入 portfolio：

| 指标 | 最低要求 |
|------|---------|
| 样本外 Sharpe | > 0.3 |
| 最大回撤 | < -35% |
| 样本外测试期 | ≥ 2 年 |
| 交易次数 | ≥ 30 笔（统计显著性） |
| 训练集/测试集 Sharpe 比 | 测试集 ≥ 训练集的 50% |

不满足的策略可以保留在品种目录供参考，但不进 portfolio。

## 防过拟合规范

1. **训练/测试分割** — 默认: 训练 2013-2021, 测试 2022-2026。不可在测试集上调参
2. **参数数量限制** — 策略可调参数 ≤ 5 个（不含通用的 warmup/capital）
3. **指标参数** — 优先用指标的默认参数。如需调参，在训练集上做，测试集验证
4. **Walk-forward** — 对关键策略做滚动验证（5年训练→1年测试），确认稳健性
5. **多品种验证** — 趋势策略至少在 3 个品种上测试，避免单品种过拟合

## 版本管理规范

### 策略版本
- **v1 → v2**: 核心逻辑变更（换指标、改信号方向、改仓位模型）
- **同版本调参**: 参数微调不升版本，在 research_log 中记录
- **新策略**: 不同逻辑思路 = 新文件（如 `trend_v1.py` 和 `breakout_v1.py`）

### 研究记录（必须）

每次实验**必须**在 `research_log/` 中记录。格式：

```markdown
## YYYY-MM-DD <品种> <策略名> <版本>
- **指标**: 指标1(参数) + 指标2(参数) + 指标3(参数)
- **训练**: 品种列表, 时间范围
- **测试**: 品种, 时间范围
- **结果**: Sharpe=X.XX, MaxDD=X.X%, WinRate=X.X%, Trades=N
- **结论**: 一句话总结什么有效/无效
- **下一步**: 后续计划
```

### Git Commit 规范

```
[品种] 策略类型: 简短描述

示例:
[AG] trend_v1: initial strategy with ADX+ROC+ATR
[AG] trend_v1: tune ADX threshold from 25 to 20
[AG] portfolio: add risk parity weights v1
[indicator] add parabolic SAR to trend category
```

## 注意事项

- 指标是纯函数（numpy in → numpy out），每次 on_bar 传入数组调用
- 日线频率性能无问题；分钟级大回看窗口可能慢
- 所有价格计算用复权价（context.get_close_array），下单用原始价（context.current_bar.close_raw）
- 信号在下一个 bar 的 open 执行，不是当前 bar
- 同品种不可同时持有多空仓位
- 保证金不足会被拒绝开仓，权益低于维持保证金会被强平
