# QBase — Agent 开发指南

QBase 是量化策略研究工作区。指标库 + 按品种组织的策略开发，回测统一用 AlphaForge。

## 项目结构

```
QBase/
├── indicators/                  # 100个量价指标（纯numpy函数）
│   ├── momentum/                # 25个动量/振荡类
│   ├── trend/                   # 25个趋势类
│   ├── volatility/              # 25个波动率类
│   └── volume/                  # 25个成交量/持仓类
├── strategies/                  # 策略库
│   ├── strong_trend/            # 强趋势策略 (v1-v20 + optimizer.py)
│   │   ├── v1.py ~ v20.py      # 20个策略（不同指标组合）
│   │   ├── optimizer.py         # Optuna 参数优化器
│   │   ├── validate_and_iterate.py  # 测试集验证+迭代优化
│   │   ├── optimization_results.json
│   │   └── portfolio/
│   ├── medium_trend/            # 中趋势策略（品种无关）
│   ├── weak_trend/              # 弱趋势策略（品种无关）
│   ├── mean_reversion/          # 震荡策略（品种无关）
│   └── all_time/                # 全时间策略（按品种，覆盖所有行情状态）
│       ├── ag/                  # 白银专用策略
│       ├── i/                   # 铁矿石专用策略
│       └── .../                 # 其他品种按需添加
├── trend/                       # 行情时段参考数据
│   ├── RALLIES.md               # 强趋势时段表（19品种，涨幅99%-907%）
│   └── MEDIUM_TRENDS.md         # 中趋势时段表（40段，涨幅20%-80%）
├── screener/                    # 品种筛选器
├── fundamental/                 # 基本面量化（预留）
├── research_log/                # 实验记录
│   └── strong_trend.md          # 强趋势策略完整结果
├── tests/                       # 指标单元测试
├── config.yaml                  # AlphaForge路径等配置
├── conftest.py                  # sys.path 配置（pytest自动加载）
└── run.sh                       # 策略运行入口
```

## 核心规则

### 1. 策略指标规则

每个策略使用 **1-3 个指标**，只要有效就行，不强制用满 3 个。

**指标来源不限于 100 个库存指标。** 可以随时开发新指标加入库中，例如：
- 品种特有指标（如铁煤比 I/J ratio、金银比 AU/AG ratio）
- 跨品种价差/比值指标
- 基本面衍生指标（库存、持仓结构等）
- 自定义统计量

开发新指标时**必须**放入 `indicators/` 对应分类，保持纯函数风格（numpy in → numpy out），供未来所有策略复用。分类规则：

| 分类 | 放入 | 示例 |
|------|------|------|
| 品种比值/价差 | `indicators/momentum/` 或新建 `indicators/spread/` | 铁煤比、金银比 |
| 跨品种相关性 | `indicators/momentum/` | 板块动量扩散 |
| 波动率衍生 | `indicators/volatility/` | 自定义波动率模型 |
| 量价/持仓衍生 | `indicators/volume/` | OI 结构分析 |
| 趋势衍生 | `indicators/trend/` | 自定义趋势评分 |

如果现有 4 个分类都不合适，可以新建分类文件夹（如 `indicators/spread/`、`indicators/fundamental/`）。

**选择思路（参考，不强制）：**
- 趋势策略常见组合: 趋势判断 + 动量确认 + 波动率过滤
- 震荡策略常见组合: 振荡器 + 波动率 + 量价确认
- 单指标策略也完全可行，如果该指标本身信号足够强

**核心原则：力求高效，不堆砌。** 1 个强指标 > 3 个弱指标。

### 2. 加仓减仓机制

#### 加仓：金字塔递减

每次加仓手数递减，控制均价风险：

```python
# 标准配置
SCALE_FACTORS = [1.0, 0.5, 0.25]   # 首仓100% → 加仓50% → 加仓25%
MAX_SCALE = 3                        # 最多3层（首仓 + 2次加仓）
```

**加仓三大前提（缺一不可）：**

1. **盈利前提** — 现有持仓浮盈 ≥ 1×ATR 才允许加仓。绝不在亏损时加仓
2. **信号确认** — 策略指标仍然给出加仓信号（动量加速、突破新高等）
3. **最小间隔** — 两次加仓之间至少间隔 10 根 bar，避免在同一位置连续加仓

```python
# 加仓逻辑模板
def _should_add(self, context, price, atr_val):
    """检查是否满足加仓条件"""
    if self.position_scale >= MAX_SCALE:
        return False
    if self.bars_since_last_scale < 10:       # 最小间隔
        return False
    if price < self.entry_price + atr_val:    # 盈利前提：浮盈 ≥ 1ATR
        return False
    # ... 策略特定的信号确认条件
    return True

def _calc_add_lots(self, base_lots):
    """金字塔递减手数"""
    factor = SCALE_FACTORS[self.position_scale]  # 0.5, 0.25, ...
    return max(1, int(base_lots * factor))
```

#### 减仓：分层止盈 + 信号弱化

**分层止盈（锁定利润）：**

| 浮盈达到 | 动作 | 剩余仓位 |
|---------|------|---------|
| 3×ATR | 减仓 1/3 | 67% |
| 5×ATR | 再减 1/3 | 33% |
| 追踪止损触发 | 平掉剩余 | 0% |

**信号弱化减仓：**
- 指标走弱但未反转（如 ROC 回落但仍为正、ADX 下降但仍 >20）→ 减半仓
- 主退出信号触发 → 全部平仓

```python
# 减仓逻辑模板
def _check_partial_exit(self, context, price, atr_val):
    """分层止盈检查"""
    profit_atr = (price - self.entry_price) / atr_val

    if profit_atr >= 5.0 and not self._took_profit_5atr:
        # 盈利达 5ATR → 再减 1/3
        lots_to_close = max(1, context.position[1] // 3)
        context.close_long(lots=lots_to_close)
        self._took_profit_5atr = True
        self.position_scale -= 1

    elif profit_atr >= 3.0 and not self._took_profit_3atr:
        # 盈利达 3ATR → 减 1/3
        lots_to_close = max(1, context.position[1] // 3)
        context.close_long(lots=lots_to_close)
        self._took_profit_3atr = True
        self.position_scale -= 1
```

**关键原则：**
- 不要一次全部减完 — 趋势可能继续，减仓是锁利润不是放弃
- 至少保留 25% 底仓直到主退出信号触发
- 减仓后的止损只能收紧，不能放宽

### 3. 策略目录结构

策略库有两种组织方式：

**A. 按市场状态（品种无关）** — 同一策略可跑任何品种：

```
strategies/
├── strong_trend/            # 强趋势策略（捕捉涨幅>100%的大行情）
│   ├── v1.py ~ v20.py
│   └── portfolio/
├── medium_trend/            # 中趋势策略（涨幅20-80%，2-4个月）
├── weak_trend/              # 弱趋势策略（小幅趋势、波段）
└── mean_reversion/          # 震荡策略（区间震荡、均值回归）
```

**B. 全时间策略 `all_time/`（按品种）** — 针对单一品种的全天候策略：

```
strategies/all_time/
├── ag/                      # 白银全时间策略
│   ├── v1.py                # 覆盖所有行情状态，专为AG优化
│   └── portfolio/
├── i/                       # 铁矿石全时间策略
│   ├── v1.py
│   └── portfolio/
└── .../                     # 其他品种按需添加
```

**两者的区别：**

| 维度 | 市场状态策略 (strong_trend 等) | 全时间策略 (all_time) |
|------|------------------------------|---------------------|
| 品种 | 品种无关，跑任何品种 | 绑定单一品种，专门优化 |
| 行情 | 只在特定行情状态下交易 | 覆盖所有行情，全天候运行 |
| 方向 | **strong/medium_trend: 只做多** | **多空都可以** |
| 训练 | 多品种训练，跨品种泛化 | 单品种全历史训练 |
| 优势 | 泛化能力强 | 针对性强，捕捉品种特性 |
| 适用 | 不确定品种会出什么行情时 | 确定要长期交易某品种时 |

**方向约束：**
- `strong_trend/`、`medium_trend/` → **只做多**（最低仓位 0，可加仓减仓，不可开空）
- `weak_trend/`、`mean_reversion/` → 视策略而定
- `all_time/` → **多空均可**（做多 + 做空，灵活应对所有行情）

**全时间策略开发要点：**
- 策略需要内置行情状态判断（趋势/震荡/突破），在不同状态下切换逻辑
- 可以参考市场状态策略的有效指标组合，整合到一个策略中
- 训练和测试都在该品种自身的全历史数据上进行
- 需要覆盖该品种经历过的各种行情（牛市、熊市、震荡、极端事件）

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

### 4. portfolio（待定）

Portfolio 的加权方案和组合逻辑尚需进一步讨论确定。当前每个策略目录下预留了 `portfolio/` 文件夹。

## 频率体系与多周期协作

### 支持频率

AlphaForge 原生支持：`5min`, `10min`, `15min`, `30min`, `60min`, `daily`
通过合成支持：`20min`（从10min×2）, `4h`（从60min×4）

**最低频率: 5min。** 不使用 1min（噪音太大，回测慢）。

### 开发原则：从小周期到大周期

策略开发时**优先从小周期开始**，逐步放大到大周期验证：

```
5min → 10min → 30min → 1h → 4h → daily
```

理由：小周期产生更多交易信号，alpha 更细腻。如果一个逻辑在 5min 上有效，再在 30min/daily 上验证稳健性。反之，如果只在 daily 有效，可能只是偶然。

### 多周期协作（Multi-Timeframe）

允许且鼓励在策略中组合不同周期的信号。典型模式：

**大周期定方向 + 小周期定进场：**
```python
# 示例：4h 判断趋势方向，5min 找精确入场点
class MultiTFStrategy(TimeSeriesStrategy):
    freq = "5min"  # 主运行频率
    warmup = 500   # 5min bars，约2个交易日

    def on_bar(self, context):
        # 小周期数据（5min）
        closes_5m = context.get_close_array(200)

        # 合成大周期数据（从 5min 聚合到 4h）
        # 方法1: 用最近 N 根 5min bar 的 close 取每 48 根的最后一个
        # 方法2: 内部维护一个 4h bar 缓冲区
        closes_4h = closes_5m[::48][-20:]  # 简化示例

        # 大周期：趋势方向
        trend_dir = 1 if supertrend(h_4h, l_4h, closes_4h, 10, 3)[1][-1] == 1 else -1

        # 小周期：入场时机（仅顺大周期方向交易）
        if trend_dir == 1:
            rsi_val = rsi(closes_5m, 14)[-1]
            if rsi_val < 30:  # 5min RSI 超卖 → 在上升趋势中买入回调
                context.buy(lots)
```

**常见多周期组合：**

| 大周期（方向） | 小周期（进场） | 适用场景 |
|---------------|---------------|---------|
| daily | 4h | 中长线趋势，日内择时 |
| 4h | 30min/1h | 波段交易 |
| 4h | 5min/10min | 精确入场，紧止损 |
| 1h | 5min | 日内趋势跟踪 |

**注意：** 多周期不是强制的。单周期策略（如 v1-v20 强趋势策略）完全有效。多周期是进阶技术，用于提升 alpha 或降低回撤。

### 频率选择指南

| 策略类型 | 推荐频率 | 理由 |
|---------|---------|------|
| 强趋势 | daily / 4h | 持仓周期长，高频噪音无意义 |
| 中趋势 | 4h / 1h | 需要更快的入场/出场响应 |
| 弱趋势/波段 | 1h / 30min | 捕捉短期波动 |
| 震荡/均值回归 | 30min / 5min | 快进快出，需要精确时机 |
| 多周期协作 | 5min（主频）+ 4h（方向） | 精确入场 + 趋势过滤 |

## 回测：统一用 AlphaForge

所有回测通过 AlphaForge 执行。AlphaForge 位于 `~/Desktop/AlphaForge/`，含 95 品种 1min 数据 (2005-2026)。

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
    freq = "daily"  # 或 "5min", "10min", "30min", "60min", "4h"

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
# 单品种单频率
./run.sh strategies/strong_trend/v1.py --symbols AG --freq daily --start 2022

# 多品种测试
./run.sh strategies/strong_trend/v1.py --symbols AG,J,ZC --freq daily --start 2020

# 不同频率
./run.sh strategies/strong_trend/v9.py --symbols AG --freq 60min --start 2024
```

### 训练/测试分割

**方法 1: 时段级分割（推荐用于趋势策略）**

用 `trend/RALLIES.md` 和 `trend/MEDIUM_TRENDS.md` 中的行情时段。将品种分为训练集和测试集：

```python
# 训练品种（强趋势）：J, ZC, JM, I, NI, SA 等
# 测试品种（留出）：AG, EC
# 在训练品种的强趋势时段上优化参数
# 在测试品种的对应时段上验证
```

**方法 2: 时间级分割（通用）**

```bash
# 训练集：2013-2021
af run strategy.py --symbols AG --start 2013 --end 2021

# 测试集：2022-2026（不改参数！）
af run strategy.py --symbols AG --start 2022 --end 2026
```

**方法 3: Walk-forward（最严格）**

```python
# 5年滚动训练→1年测试
for year in range(2018, 2026):
    train: year-5 ~ year-1
    test: year
```

选择哪种方法取决于策略类型。趋势策略推荐方法 1（因为趋势不是每年都有），震荡策略推荐方法 2。

## Optuna 参数优化流程

### 优化器位置

每个市场状态目录下有独立的 `optimizer.py`：

```
strategies/strong_trend/optimizer.py     # 强趋势优化器
strategies/medium_trend/optimizer.py     # 中趋势优化器（待开发）
```

### 优化流程

```bash
# 1. 优化单个策略（150 trials，6 个训练品种）
python strategies/strong_trend/optimizer.py --strategy v1 --trials 150

# 2. 优化全部策略
python strategies/strong_trend/optimizer.py --strategy all --trials 150

# 3. 测试集验证 + 迭代优化（自动3轮）
python strategies/strong_trend/validate_and_iterate.py
```

### 优化规范

- **目标函数**: `mean(Sharpe across N training symbols)` — 多品种平均防过拟合
- **训练品种**: 至少 6 个，覆盖不同板块（黑色、能源、化工、有色等）
- **Trials**: 每策略 150-200 次
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **参数范围**: 每个参数 2-3x 范围，不要太宽
- **结果保存**: `optimization_results.json`，包含每个策略的最优参数和 Sharpe

### 非标准频率支持

优化器自动处理 4h/1h 等非原生频率：
- `4h`: 加载 60min 数据，每 4 根合成 1 根 4h bar
- `1h`: 直接用 60min
- `20min`: 加载 10min，每 2 根合成

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

### 单策略开发流程

1. **确定市场状态** — strong_trend / medium_trend / weak_trend / mean_reversion
2. **选 3 个指标** — 从 100 个中选，确保互补（不同分类），参考 `research_log/` 中已验证有效的组合
3. **选频率** — 从小周期开始 (5min → 10min → 30min → 1h → 4h → daily)，可考虑多周期协作
4. **写策略** — 继承 `TimeSeriesStrategy`，放入对应市场状态目录
5. **训练集验证** — 在训练品种 + 训练时段上跑，确认逻辑正确
6. **Optuna 优化** — `optimizer.py --strategy vN --trials 150`，多品种平均 Sharpe
7. **测试集验证** — `validate_and_iterate.py`，在留出品种上验证
8. **记录结果** — 写入 `research_log/`，包含指标、参数、结果、结论

### 批量开发流程（推荐）

一次性开发 10-20 个不同指标组合的策略版本，然后批量优化筛选：

```bash
# 1. 开发 v1-v20（可用 subagent 并行）
# 2. 批量优化
python strategies/strong_trend/optimizer.py --strategy all --trials 150
# 3. 测试集验证 + 迭代
python strategies/strong_trend/validate_and_iterate.py
# 4. 淘汰 Sharpe < 0 的，保留最优
```

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

1. **品种级分割** — 训练品种和测试品种完全分开。优化时只碰训练品种，测试品种从不参与参数选择
2. **多品种平均** — Optuna 目标函数 = mean(Sharpe across 6+ 训练品种)，不是单品种最优
3. **参数数量限制** — 每策略可调参数 ≤ 5 个。参数范围窄（2-3x），不给优化器太大空间
4. **指标参数** — 优先用指标默认参数。Optuna 调的是策略参数（阈值、周期），不是指标内部参数
5. **迭代验证** — `validate_and_iterate.py` 自动 3 轮迭代：验证 → 重优化表现差的 → 再验证
6. **Walk-forward** — 对关键策略做滚动验证（5年训练→1年测试），确认参数稳定性
7. **一致性奖励** — 重优化时目标函数加入 `min_sharpe * 0.3` bonus，奖励所有品种都盈利的参数

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

## 风险管理框架

### 单笔风险

每笔交易最大风险 = 账户权益的 **2%**。这是仓位计算的基础：

```python
# 标准仓位公式
lots = (equity * 0.02) / (stop_distance * contract_multiplier)

# 示例：AG 白银，权益 100万，ATR=50，止损 3.5×ATR
# lots = 1,000,000 * 0.02 / (50 * 3.5 * 15) = 7.6 → 7 手
```

### 总敞口限制

| 限制 | 数值 | 说明 |
|------|------|------|
| 单策略最大持仓 | 权益的 **30%** 保证金占用 | 含所有加仓层 |
| 单品种最大持仓 | 权益的 **40%** 保证金占用 | 同品种多策略合计 |
| 总账户最大持仓 | 权益的 **80%** 保证金占用 | 留 20% 现金缓冲 |
| 单笔最大亏损 | 权益的 **2%** | 通过仓位公式控制 |
| 单日最大亏损 | 权益的 **5%** | 触发后当日停止交易 |

### 合约参数：动态获取，不要硬编码

从 AlphaForge 的 `ContractSpecManager` 获取品种参数：

```python
from alphaforge.data.contract_specs import ContractSpecManager

specs = ContractSpecManager()
spec = specs.get("AG")
spec.multiplier      # 15
spec.tick_size        # 1.0
spec.margin_rate      # 0.08
spec.commission_open  # 0.00005 (ratio)

# 仓位计算中使用
margin_per_lot = price * spec.multiplier * spec.margin_rate
max_lots_by_margin = int(equity * 0.30 / margin_per_lot)  # 30% 上限
lots = min(risk_lots, max_lots_by_margin)
```

## 中国期货特殊规则

### 夜盘归属

中国期货夜盘（21:00-次日02:30）归属**下一个交易日**。AlphaForge 的 `context.trading_day` 已正确处理此逻辑。

注意事项：
- 周五夜盘 → 属于下周一交易日
- 节假日前最后一个夜盘 → 属于节后第一个交易日
- 判断日内/隔夜时用 `trading_day` 而非 `datetime`

### 涨跌停

- 触及涨跌停时 AlphaForge 拒绝成交
- 不同品种涨跌停幅度不同（一般 4%-8%，通过 `spec.price_limit` 获取）
- 策略不应在涨跌停附近下单（成交概率低）

### 数据可用范围

AlphaForge 含 95 品种的 1min 连续合约数据，但**各品种上市时间不同**：

| 上市时间段 | 品种示例 |
|-----------|---------|
| 2005-2010 | CU, AL, ZN, RU, A, M, Y, C, CF, SR, TA, L |
| 2010-2015 | AG, AU, I, J, JM, RB, HC, FG, RM, OI, CS |
| 2015-2020 | NI, SN, SC, SP, EG, EB, SA, SS, PG, LU, BC |
| 2020-2026 | LC, SI, PX, EC, BR, AO, PS |

使用品种前先确认数据起始日期：
```python
loader = MarketDataLoader("data/")
symbols = loader.available_symbols()  # 所有可用品种
bars = loader.load("AG", freq="daily", start="2012-05-10")  # AG 从2012年开始
```

## 实战经验参考

详见 `research_log/lessons_learned.md`（独立文档，记录已验证的实战经验和教训）。

## 注意事项

- 指标是纯函数（numpy in → numpy out），每次 on_bar 传入数组调用
- 日线频率性能无问题；分钟级大回看窗口可能慢
- 所有价格计算用复权价（context.get_close_array），下单用原始价（context.current_bar.close_raw）
- 信号在下一个 bar 的 open 执行，不是当前 bar
- 同品种不可同时持有多空仓位
- 保证金不足会被拒绝开仓，权益低于维持保证金会被强平
- FIFO 平仓：先平昨仓（便宜），再平今仓（贵）
- 单笔成交量不超过该 bar 成交量的 10%
