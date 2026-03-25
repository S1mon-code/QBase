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

### 4. Portfolio 构建

#### 核心原则

**组合的唯一目标：最大化组合 Sharpe，同时控制回撤。**

不是把好策略堆在一起，而是找到一组**互补**的策略。一个负 Sharpe 但与其他策略负相关的策略，可能比一个正 Sharpe 但高度相关的策略更有价值。

#### 构建流程

```
开发 N 个策略 → 全部回测（不按 Sharpe 过滤）→ 贪心组合优化
→ Sharpe 加权 HRP 赋权 → 权重上限裁剪 → Leave-one-out 检验 → 最终 portfolio
```

#### Step 1: 贪心组合优化（策略选择）

**不使用 Sharpe 门槛过滤。** 改为以组合 Sharpe 最大化为目标的贪心选择：

```python
# 贪心构建算法
pool = sort_by_sharpe_descending(all_strategies)  # 含负 Sharpe 策略
selected = [pool[0]]  # 从最高 Sharpe 开始

for strategy in pool[1:]:
    candidate = selected + [strategy]
    candidate_sharpe = calc_portfolio_sharpe(candidate)

    if candidate_sharpe > current_portfolio_sharpe:
        selected.append(strategy)  # 加入后组合更好 → 保留
        current_portfolio_sharpe = candidate_sharpe
    # 否则跳过（包括正 Sharpe 但高相关的，和负 Sharpe 但没帮助的）
```

**关键：负 Sharpe + 负相关性的策略会被自然纳入。** 比如一个 Sharpe=-0.3 但与组合相关性 -0.5 的策略，加入后可能让组合 MaxDD 从 -10% 降到 -6%，整体 Sharpe 反而提升。

**不再使用硬性相关性阈值（0.7）过滤。** 相关性过滤太粗暴 — 两个相关性 0.75 的策略如果回撤不重叠，组合仍有价值。贪心算法会自然处理：加入后组合 Sharpe 提升就保留，否则跳过。

#### 多频率策略混合

Portfolio 中的策略可能混合 5min、1h、4h、daily 等不同频率：

**收益率对齐：** 所有策略的权益曲线统一 resample 到**日线级别**后再做相关性计算和赋权。

```python
daily_equity = resample_to_daily(strategy_equity_curve)
daily_returns = daily_equity.pct_change()
```

**仓位管理注意：**
- 高频策略（5min/10min）日内可能多次开平仓，日线策略可能持仓数天
- 两者同时运行时，总保证金占用可能在盘中瞬间叠加
- 风控的总敞口限制（80%）按**实时合计**，不是按日线合计

**信号冲突处理：**
- 不同频率策略可能对同一品种产生相反信号（如 daily 做多，5min 做空）
- 在 `all_time/` 中这是允许的（多空均可），但需要注意净头寸
- 在 `strong_trend/` 中不会冲突（都是只做多）

#### Step 2: Sharpe 加权 HRP 赋权

纯 HRP 只按相关性分配权重，会给低相关性但低质量的策略过多权重。改为 **Sharpe 加权 HRP**：

```python
# Step 2a: 标准 HRP 权重（基于相关性聚类）
w_hrp = hrp_weights(returns_matrix)

# Step 2b: Sharpe 调整因子
sharpe_factor = {v: max(0.1, sharpe) for v, sharpe in strategy_sharpes.items()}
# 对负 Sharpe 策略（因负相关被纳入）给予最低权重 0.1

# Step 2c: 混合权重 = HRP × Sharpe 调整，然后归一化
w_final = {v: w_hrp[v] * sharpe_factor[v] for v in strategies}
w_final = normalize(w_final)  # 归一化到 sum=1
```

**效果：** 高 Sharpe + 低相关的策略获得最多权重。低 Sharpe 的"对冲策略"获得适当但不过多的权重。

#### Step 3: 权重上限裁剪

**单策略最大权重 15%。** 超出部分按比例重新分配给其他策略：

```python
MAX_WEIGHT = 0.15

while any(w > MAX_WEIGHT for w in weights.values()):
    excess = {v: w - MAX_WEIGHT for v, w in weights.items() if w > MAX_WEIGHT}
    for v in excess:
        weights[v] = MAX_WEIGHT
    # 将多余权重按比例分配给未达上限的策略
    total_excess = sum(excess.values())
    under_cap = {v: w for v, w in weights.items() if w < MAX_WEIGHT}
    for v in under_cap:
        weights[v] += total_excess * (weights[v] / sum(under_cap.values()))
```

**为什么 15%：** 实测中 v41 在 AG 上拿了 24%、v31 在 LC 上拿了 22%，单策略集中度过高。15% 上限保证至少 7 个策略共同承担风险。

#### Step 4: Leave-one-out 边际检验

构建完成后，逐个移除策略验证其边际贡献：

```python
for strategy in portfolio:
    without = portfolio - {strategy}
    sharpe_without = calc_portfolio_sharpe(without)

    if sharpe_without > current_sharpe:
        # 去掉这个策略反而更好 → 移除它
        portfolio.remove(strategy)
        current_sharpe = sharpe_without
        print(f"Removed {strategy}: portfolio Sharpe improved to {sharpe_without}")
```

**这是最后的清理步骤。** 贪心构建时加入的策略，在其他策略也加入后可能变成冗余。Leave-one-out 检验能发现并清理这些冗余。

#### Step 5: Portfolio 质量要求

**Portfolio 评分必须达到 B+ (75分) 以上才可使用：**

| 指标 | 最低要求 | 理想目标 |
|------|---------|---------|
| 综合评分 | ≥ 75 (B+) | ≥ 85 (A) |
| Portfolio Sharpe / 最佳单策略 | ≥ 0.8 | ≥ 1.0 |
| Max Drawdown | < -15% | < -5% |
| 策略间平均相关性 | < 0.5 | < 0.3 |
| 最大单策略权重 | ≤ 15% | ≤ 10% |

如果评分低于 B+，必须在 `research_log/` 中分析原因并迭代：
- 组合 Sharpe 低于单策略？→ 检查 Sharpe 加权是否生效
- 回撤过大？→ 检查是否缺少对冲策略（负相关层）
- 相关性过高？→ 扩大策略池（加入不同频率/指标类型）
- 某策略权重过高？→ 降低权重上限

#### Step 4: Portfolio 止损标准

| 级别 | 触发条件 | 动作 |
|------|---------|------|
| **预警** | 组合回撤达 **-10%** | 记录日志，检查各策略状态，不自动操作 |
| **减仓** | 组合回撤达 **-15%** | 所有策略仓位减半，暂停加仓 |
| **熔断** | 组合回撤达 **-20%** | 全部平仓，停止交易，人工审查后重启 |
| **单日熔断** | 单日亏损达权益 **-5%** | 当日全部平仓，次日恢复 |

熔断后的恢复规则：
- 减仓后：回撤收窄至 -10% 以内，恢复正常仓位
- 全面熔断后：必须人工审查并确认市场环境正常后才能重启
- 连续 2 次触发 -20% 熔断：停止该 portfolio，重新优化权重

#### Step 5: 重平衡（待定）

重平衡机制后续讨论确定。

## 频率体系与多周期协作

### 支持频率

AlphaForge 支持任意频率：`<N>min`（如 `1min`, `5min`, `20min`）、`<N>h`（如 `1h`, `4h`）、`daily`

**最低频率: 5min。** 不使用 1min（噪音太大，回测慢）。

### 开发原则：从小周期到大周期

策略开发时**优先从小周期开始**，逐步放大到大周期验证：

```
5min → 10min → 30min → 1h → 4h → daily
```

理由：小周期产生更多交易信号，alpha 更细腻。如果一个逻辑在 5min 上有效，再在 30min/daily 上验证稳健性。反之，如果只在 daily 有效，可能只是偶然。

### 多周期协作（Multi-Timeframe）

允许且鼓励在策略中组合不同周期的信号。典型模式：

**大周期定方向 + 小周期定进场（预计算模式）：**
```python
# 示例：4h 判断趋势方向，5min 找精确入场点
class MultiTFStrategy(TimeSeriesStrategy):
    freq = "5min"  # 主运行频率
    warmup = 500   # 5min bars，约2个交易日

    def __init__(self):
        super().__init__()
        self._rsi = None           # 5min RSI
        self._st_dir_4h = None     # 4h Supertrend direction
        self._4h_map = None        # 5min → 4h 索引映射

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        n = len(closes)
        step = 48  # 5min × 48 = 4h

        # 5min 指标
        self._rsi = rsi(closes, 14)

        # 聚合 → 4h
        n_4h = n // step
        trim = n_4h * step
        c4h = closes[:trim].reshape(n_4h, step)[:, -1]
        h4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        l4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        _, self._st_dir_4h = supertrend(h4h, l4h, c4h, 10, 3)
        self._4h_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]

        trend_dir = self._st_dir_4h[j]
        rsi_val = self._rsi[i]

        if trend_dir == 1 and rsi_val < 30:
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

### 策略架构：预计算模式（on_init_arrays）

**所有策略必须使用预计算模式。** 传统模式（每 bar 重算指标）在高频数据上慢 20-50x。

引擎调用顺序：
```
strategy.on_init(context)              # 1. 初始化状态变量
strategy.on_init_arrays(context, bars) # 2. 预计算所有指标（一次性）
for each bar:
    strategy.on_bar(context)           # 3. 查表 + 交易逻辑
strategy.on_stop(context)              # 4. 清理
```

**标准策略模板（单周期）：**

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
    freq = "daily"  # 或 "5min", "10min", "30min", "60min", "1h", "4h"

    # 可调参数
    adx_period: int = 14
    roc_period: int = 20

    def __init__(self):
        super().__init__()
        # 声明预计算数组
        self._adx = None
        self._roc = None
        self._atr = None

    def on_init(self, context):
        self.entry_price = 0.0

    def on_init_arrays(self, context, bars):
        """一次性预计算所有指标。只调一次。"""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

    def on_bar(self, context):
        i = context.bar_index  # 当前bar在完整数组中的索引

        # 查表 — O(1)
        adx_val = self._adx[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]

        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        price = context.close_raw  # 直接属性，比 context.current_bar.close_raw 更快
        side, lots = context.position

        # 交易逻辑...
```

**多周期策略模板（如 30min 主频 + 4h 方向）：**

```python
class MultiTFStrategy(TimeSeriesStrategy):
    freq = "30min"
    warmup = 500

    def __init__(self):
        super().__init__()
        self._rsi = None        # 30min RSI
        self._atr = None        # 30min ATR
        self._st_dir_4h = None  # 4h Supertrend direction
        self._st_line_4h = None # 4h Supertrend line
        self._4h_map = None     # 30min index → 4h index 映射

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        n = len(closes)
        step = 8  # 30min × 8 = 4h

        # 30min 指标 — 直接全数组
        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

        # 聚合 → 4h bars
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        # 4h 指标
        self._st_line_4h, self._st_dir_4h = supertrend(
            highs_4h, lows_4h, closes_4h, self.st_period, self.st_mult)

        # 映射: 30min bar i → 最近完成的 4h bar index（避免前视偏差）
        self._4h_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]  # 对应的 4h bar index

        cur_st_dir = self._st_dir_4h[j]
        cur_rsi = self._rsi[i]
        cur_atr = self._atr[i]
        # ... 交易逻辑
```

### 性能对比

| 周期 | ~Bars (5年) | 传统模式 | 预计算模式 | 加速比 |
|------|------------|---------|-----------|--------|
| 1min | ~1,500K | >10min | ~8-15s | ~50x |
| 5min | ~350K | ~98s | ~3-5s | ~25x |
| 30min | ~25K | ~7s | <1s | ~10x |
| 1h | ~12K | ~3s | <0.5s | ~6x |
| daily | ~1.2K | <0.3s | <0.1s | ~3x |

### AlphaForge Context API

```python
# === on_init_arrays 中可用 ===
context.get_full_close_array()      # 完整复权close (np.ndarray, shape=(N,))
context.get_full_high_array()       # 完整high
context.get_full_low_array()        # 完整low
context.get_full_open_array()       # 完整open
context.get_full_volume_array()     # 完整volume
context.get_full_oi_array()         # 完整持仓量
context.get_full_close_raw_array()  # 完整原始close（未复权）

# === on_bar 中可用 ===
i = context.bar_index               # 当前bar在完整数组中的索引（关键！）
val = self._precomputed[i]          # O(1) 查表

# 快速属性访问（跳过 Bar namedtuple 创建）
price = context.close_raw           # 比 context.current_bar.close_raw 更快
context.open_raw / context.high_raw / context.low_raw / context.volume

# 传统方式（仍可用，但慢）
context.current_bar.close_raw
context.get_close_array(n)          # 最近n根close

# 持仓与权益
side, lots = context.position       # side: 1=多, -1=空, 0=无
context.equity                      # 当前权益
context.available_cash              # 可用资金

# 下单（下一bar的open执行）
context.buy(lots)                   # 开多
context.sell(lots)                  # 开空
context.close_long()                # 平多（全部）
context.close_long(lots=5)          # 平多（指定手数）
context.close_short()               # 平空
```

### 组合回测 (PortfolioBacktester)

```python
from alphaforge.engine.portfolio import PortfolioConfig, StrategyAllocation, PortfolioBacktester
from alphaforge.data.contract_specs import ContractSpecManager

config = PortfolioConfig(
    total_capital=3_000_000,
    allocations=[
        StrategyAllocation(
            name="trend_v9",
            strategy_file="strategies/strong_trend/v9.py",
            symbols="AG",
            freq="1h",
            weight=0.4,
            params={},  # 可选：覆盖策略属性
        ),
        StrategyAllocation("trend_v12", "strategies/strong_trend/v12.py", "AG", "daily", weight=0.3),
        StrategyAllocation("trend_v1",  "strategies/strong_trend/v1.py",  "AG", "daily", weight=0.3),
    ],
    rebalance="none",  # "none", "monthly", "quarterly"
)

runner = PortfolioBacktester(spec_manager=ContractSpecManager(), data_dir="data/")
report = runner.run(config, start="2023-01-01", end="2025-12-31")

# 组合指标
report.combined_result.sharpe / .total_return / .max_drawdown / .equity_curve
# 逐策略
for slot in report.slot_results:
    slot.name, slot.weight, slot.result.sharpe
# 相关性矩阵
report.correlation_matrix  # pd.DataFrame
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

## Portfolio 评分体系

每个 Portfolio 用 **0-100 综合评分**，覆盖 4 个维度。适用于单品种 all_time 和多品种组合。

### 评分维度与权重

| 维度 | 权重 | 包含指标 | 关注点 |
|------|:----:|---------|--------|
| **收益风险比** | 40% | Sharpe, Calmar, MaxDD, 回撤持续时间 | 赚多少、亏多少、亏多久 |
| **组合质量** | 25% | 平均相关性, Portfolio/最佳单策略比, 正Sharpe占比 | 分散有效吗、组合有增值吗 |
| **实操性** | 20% | 策略数量, 最大单策略权重, 频率多样性 | 能不能实际跑起来 |
| **稳定性** | 15% | 收益一致性, 权益曲线稳定度 | 是不是靠运气 |

### 各指标评分标准 (0-10)

**Sharpe Ratio:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| ≥3.0 | 10 | 卓越 |
| 2.0 | 6.7 | 优秀 |
| 1.0 | 3.3 | 及格 |
| ≤0 | 0 | 不合格 |

**Calmar Ratio (年化收益/最大回撤):**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| ≥10 | 10 | 极低风险高收益 |
| 5 | 5 | 中等 |
| 1 | 1 | 勉强 |

**Max Drawdown:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| <3% | 10 | 极低风险 |
| <5% | 8.5 | 低风险 |
| <10% | 6 | 中等 |
| <20% | 3 | 高风险 |
| >30% | 0 | 不可接受 |

**平均相关性:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| <0.2 | 10 | 完美分散 |
| <0.3 | 8.5 | 优秀 |
| <0.5 | 5.5 | 及格 |
| >0.7 | 1 | 无分散价值 |

**Portfolio / 最佳单策略 Sharpe 比:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| ≥1.15 | 9 | 组合显著增值 |
| ≥1.0 | 8 | 组合优于单策略 |
| ≥0.8 | 6 | 可接受 |
| <0.6 | 1 | 组合没有价值 |

**最大单策略权重:**

| 值 | 得分 | 评价 |
|:--:|:----:|------|
| <10% | 10 | 充分分散 |
| <15% | 8 | 良好 |
| <25% | 5 | 集中风险 |
| >30% | 2 | 过度集中 |

### 等级映射

| 分数 | 等级 | 含义 |
|:----:|:----:|------|
| 90+ | A+ | 卓越 — 可直接进入模拟盘 |
| 85-89 | A | 优秀 — 小幅优化后可上线 |
| 80-84 | A- | 良好 — 需要补充 walk-forward 验证 |
| 75-79 | B+ | 中上 — 核心指标过关，部分维度待提升 |
| 70-74 | B | 中等 — 可用但有明显弱点 |
| 60-69 | C | 勉强 — 需要重大改进 |
| <60 | D/F | 不合格 — 重新构建 |

### 当前 Portfolio 评分

| Portfolio | 分数 | 等级 | 主要弱点 |
|-----------|:----:|:----:|---------|
| LC HRP (碳酸锂) | **87.6** | **A** | 最大权重 21.6% |
| AG HRP (白银) | **71.5** | **B** | 组合/单策略比 0.79, 相关性 0.45 |

### 运行评分

```bash
python strategies/strong_trend/portfolio_scorer.py
```

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

### 主连数据与换仓移仓

**重要：回测和训练使用的是主力连续合约（主连）数据，但主连本身不可交易。**

实际交易中需要处理合约切换（换仓/移仓）问题：

**回测阶段（当前）：**
- AlphaForge 的数据已做复权处理，`close` 是复权价（用于指标计算），`close_raw` 是原始价（用于成交计算）
- `context.current_bar.is_rollover` 标记展期日
- `context.current_bar.origin_symbol` 显示实际合约代码（如 "AG2506"）
- 回测引擎自动处理展期日的持仓转移，成本已包含在滑点模型中

**实盘需额外考虑（未来）：**
- 换仓时机：主力合约切换日，通常在交割月前 1-2 个月
- 换仓成本：平旧仓 + 开新仓的双向手续费 + 价差滑点
- 换仓风险：新旧合约价差可能不连续（特别是交割月附近）
- 需要开发独立的换仓模块来处理合约到期和切换逻辑

**策略开发时的注意：**
- 展期日信号可能失真（复权跳变），可通过 `is_rollover` 跳过该 bar 的信号
- 跨月价差策略需要直接使用 `origin_symbol` 区分合约月份
- 回测结果中的收益已包含换仓成本估算，但实盘可能更高

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

## 前视偏差（Look-ahead Bias）检查

前视偏差是量化开发中最隐蔽的错误 — 无意间在计算中使用了未来数据，导致回测虚高。

**必须遵守的规则：**

1. **信号用当前 bar 数据，执行在下一 bar** — AlphaForge 已强制：`on_bar` 中下单，下一 bar 的 open 成交。不可绕过
2. **指标只能用历史数据** — `context.get_close_array(n)` 返回的是截至当前 bar 的历史，这是安全的。不要用 pandas 的 `shift(-1)` 等前移操作
3. **不要用未来的 Sharpe/收益来选策略** — 训练集上选策略，测试集上验证。不能在测试集上挑结果最好的参数

**常见前视偏差陷阱：**

| 陷阱 | 说明 | 正确做法 |
|------|------|---------|
| 用收盘价决定当 bar 交易 | 收盘价在 bar 结束时才知道 | 信号在当 bar 产生，下 bar open 执行 |
| 指标用未来数据填充 NaN | 如用后续值回填 warmup 期 | warmup 期保持 NaN，不交易 |
| 全样本标准化 | 用全部数据的 mean/std 标准化 | 只用历史数据做滚动标准化 |
| 用测试集调参 | 看了测试结果后改参数 | 参数锁定后才看测试集，不可回头改 |
| 展期日复权跳变 | 复权价在展期日跳变，非真实价格变动 | 用 `is_rollover` 跳过展期日信号 |

**开发时自查：** 对每个指标和信号问自己 — "在这个 bar 的时间点，我真的已经知道这个值了吗？"

## 策略生命周期

策略从开发到实盘的完整路径，每一步有明确的晋升标准：

```
回测开发 → 参数优化 → 样本外验证 → 模拟盘 → 小资金实盘 → 正式实盘
```

| 阶段 | 环境 | 晋升标准 | 淘汰条件 |
|------|------|---------|---------|
| **回测开发** | AlphaForge 历史数据 | 训练集 Sharpe > 0.5 | 逻辑不通或无法盈利 |
| **参数优化** | Optuna + 多品种训练集 | 多品种平均 Sharpe > 0.3 | 只在单品种有效 |
| **样本外验证** | 留出的测试品种/时段 | 测试集 Sharpe > 0 | 过拟合，测试集亏损 |
| **模拟盘** | 实时行情，虚拟资金 | 运行 1 个月+，表现与回测一致 | 实时表现严重偏离回测 |
| **小资金实盘** | 真实资金（10-20% 仓位） | 运行 3 个月+，Sharpe > 0 | 持续亏损或风控触发 |
| **正式实盘** | 全额资金 | 持续监控 | 触发 portfolio 止损标准 |

**关键原则：**
- 每一步都不可跳过
- 只有通过当前阶段才能进入下一阶段
- 任何阶段发现问题都退回"回测开发"重新来
- 模拟盘与回测的收益偏差 > 30% 需要调查原因（滑点？延迟？数据差异？）

## 流动性与策略容量

不是所有品种都能承载大资金。策略开发时必须关注流动性：

**流动性分级：**

| 级别 | 日均成交量 | 代表品种 | 最大建议仓位 |
|------|-----------|---------|-------------|
| A 高流动性 | > 100 万手 | RB, I, MA, TA, AG, AU | 不受限 |
| B 中流动性 | 10-100 万手 | J, JM, NI, CU, AL, ZN, SC | 日成交量的 2% |
| C 低流动性 | 1-10 万手 | SN, BC, SS, SP, LU | 日成交量的 1% |
| D 极低流动性 | < 1 万手 | 部分新上市品种 | 不建议大资金交易 |

**AlphaForge 已有保护：** 单笔成交量不超过该 bar 成交量的 10%。但这只是回测保护，实盘中低流动性品种的实际滑点会远超模型假设。

**策略容量评估：**
- 策略盈利 ≠ 可以无限放大。当仓位占日成交量 > 5% 时，市场冲击成本会显著侵蚀 alpha
- 高频策略（5min/10min）对流动性要求更高
- 评估容量时用保守估计：假设实际滑点是回测的 2-3 倍

## 实战经验参考

详见 `research_log/lessons_learned.md`（独立文档，记录已验证的实战经验和教训）。

## 注意事项

- **所有策略必须使用预计算模式**（on_init_arrays + context.bar_index 查表）
- 指标是纯函数（numpy in → numpy out），在 on_init_arrays 中传入完整数组一次计算
- 预计算与 Optuna 兼容：每次 trial 参数变化时，on_init_arrays 自动重新调用
- 所有价格计算用复权价（context.get_full_close_array），下单用原始价（context.close_raw）
- 信号在下一个 bar 的 open 执行，不是当前 bar
- 同品种不可同时持有多空仓位
- 保证金不足会被拒绝开仓，权益低于维持保证金会被强平
- FIFO 平仓：先平昨仓（便宜），再平今仓（贵）
- 单笔成交量不超过该 bar 成交量的 10%
