# Step 2: 策略开发（Strategy Development）

> 从指标池中选 1-3 个指标，用预计算架构写出完整策略。

## 指标选择规则

每个策略使用 **1-3 个指标**，只要有效就行，不强制用满 3 个。

**选择思路（参考，不强制）：**
- 趋势策略常见组合: 趋势判断 + 动量确认 + 波动率过滤
- 震荡策略常见组合: 振荡器 + 波动率 + 量价确认
- 单指标策略也完全可行，如果该指标本身信号足够强

**核心原则：力求高效，不堆砌。** 1 个强指标 > 3 个弱指标。

## 方向约束

- `strong_trend/`、`medium_trend/` → **只做多，可加仓减仓，总仓位 ≥ 0（不可为负，即不可开空）**
- `weak_trend/`、`mean_reversion/` → 视策略而定
- `all_time/` → **多空均可**（做多 + 做空，灵活应对所有行情）

**趋势策略仓位规则（strong_trend / medium_trend）：**
- 允许加仓（金字塔递减）和减仓（分层止盈 / 信号弱化）
- 减仓后仓位可以是 1 手、半仓等，但**绝不能减到负仓位**
- `context.close_long()` 平多，**禁止使用 `context.sell()` 开空**
- 完全平仓后仓位归零，等待下一个做多信号

**全时间策略仓位规则（all_time/）：**
- 可以 `context.buy()` 做多，也可以 `context.sell()` 做空
- 同一时间只能持有单一方向（不可同时多空）
- 从多头切换到空头时，先 `close_long()` 再 `sell()`
- 仓位可以为正（多）、零（空仓）、负（空）

## 仓位管理规则（加仓 / 减仓 / 止损）

**所有规则已集成到策略模板中（见预计算模式章节）。以下是规则总结：**

### 加仓：金字塔递减

配置：`SCALE_FACTORS = [1.0, 0.5, 0.25]`，最多 3 层（`MAX_SCALE = 3`）。

**加仓三大前提（缺一不可）：**
1. **盈利前提** — 浮盈 ≥ 1×ATR，绝不在亏损时加仓
2. **信号确认** — 策略指标仍然给出加仓信号
3. **最小间隔** — 两次加仓至少间隔 10 根 bar

### 减仓：分层止盈 + 信号弱化

| 浮盈达到 | 动作 | 剩余仓位 |
|---------|------|---------|
| 3×ATR | 减仓 1/3 | 67% |
| 5×ATR | 再减 1/3 | 33% |
| 追踪止损触发 | 平掉剩余 | 0% |

**信号弱化减仓：** 主退出信号触发 → 全部平仓

**关键原则：**
- 不要一次全部减完 — 至少保留 25% 底仓直到主退出信号
- 减仓后的止损只能收紧，不能放宽

### 止损：ATR 追踪止损

**每个策略必须内置 ATR 止损。**

- **初始止损：** 入场价 ± N×ATR（默认 N=3.0，Optuna 范围 2.0-5.0）
- **追踪止损：** 最高价（多头）或最低价（空头）回撤 N×ATR 触发
- **只收紧不放宽**
- 加仓后止损不变，追踪止损继续跟随
- 空头止损逻辑对称（仅 all_time 策略）

### on_bar 内执行顺序

1. 极端行情过滤（展期日/低量/涨跌停）
2. **止损检查（保命第一）**
3. 分层止盈检查
4. 信号弱化减仓/主退出
5. 入场逻辑
6. 加仓逻辑

## 策略目录结构

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

**市场状态定义：**

| 状态 | 涨幅范围 | 持续时间 | 参考数据 |
|------|---------|---------|---------|
| 强趋势 | >100% | 6个月+ | trend/RALLIES.md |
| 中趋势 | 20-80% | 2-4个月 | trend/MEDIUM_TRENDS.md |
| 弱趋势 | 5-20% | 1-3个月 | 待整理 |
| 震荡 | ±5%内 | 不定 | 待整理 |

**全时间策略开发要点：**
- 策略需要内置行情状态判断（趋势/震荡/突破），在不同状态下切换逻辑
- 可以参考市场状态策略的有效指标组合，整合到一个策略中
- 训练和测试都在该品种自身的全历史数据上进行
- 需要覆盖该品种经历过的各种行情（牛市、熊市、震荡、极端事件）

**策略是品种无关的。** 运行时通过 `--symbols` 指定品种：
```bash
# 同一个强趋势策略，跑不同品种
./run.sh strategies/strong_trend/v1.py --symbols AG --freq daily --start 2022
./run.sh strategies/strong_trend/v1.py --symbols J --freq daily --start 2016
./run.sh strategies/strong_trend/v1.py --symbols ZC --freq daily --start 2021
```

## 策略文档要求

每个策略文件必须在类的 docstring 中包含：
1. **策略简介** — 一句话描述核心逻辑
2. **使用指标** — 列出所有指标及其作用
3. **进场条件** — 做多/做空的具体触发条件
4. **出场条件** — 止损、止盈、信号反转的具体条件
5. **优点** — 该策略的核心优势
6. **缺点** — 已知的局限性和弱点

## 策略命名

新策略统一以 v1-v500 命名，优先扩展新版本而非修改旧版本。

## 数据分割标准

**All-Time 单品种策略（all_time/）：**
- **训练集：** 2022 年之前的全部历史数据
- **测试集：** 2022 年及之后的数据
- 此分割适用于所有 `all_time/` 下的品种策略

**趋势策略（strong_trend / medium_trend 等）：**
- 训练集和测试集由 Simon 在开发前单独定义
- 不使用上述时间分割，而是基于品种 + 行情时段分割

用 `trend/RALLIES.md` 和 `trend/MEDIUM_TRENDS.md` 中的行情时段作为参考：

```python
# 示例（以 Simon 定义为准）：
# 训练品种（强趋势）：J, ZC, JM, I, NI, SA 等
# 测试品种（留出）：AG, EC
# 在训练品种的强趋势时段上优化参数
# 在测试品种的对应时段上验证
```

**Walk-forward（最严格）：**

```python
# 5年滚动训练→1年测试
for year in range(2018, 2026):
    train: year-5 ~ year-1
    test: year
```

选择哪种方法取决于策略类型。趋势策略推荐方法 1（因为趋势不是每年都有），震荡策略推荐方法 2。

## 快速可行性验证（7 项检查）

写完代码后、跑优化前，必须先做：
1. 在训练集上跑 1-2 个品种的基础回测
2. 确认策略有交易信号、不报错
3. 交易次数 ≥ 10（排除僵尸策略）
4. 做多/做空方向与指标信号一致
5. 止损正常生效
6. 信号频率合理（不过密也不过稀）
7. 训练集单品种 Sharpe > 0（否则不值得进入优化）

## 策略去重/冗余检查

策略数量扩展到 v200+ 后，新策略容易与已有策略高度相关。**开发新策略前必须检查冗余：**

```python
# 在训练集上跑新策略和现有策略，计算日收益相关性
from numpy import corrcoef

new_returns = backtest(new_strategy, train_data)
for existing in portfolio_strategies:
    existing_returns = backtest(existing, train_data)
    corr = corrcoef(new_returns, existing_returns)[0, 1]
    if abs(corr) > 0.7:
        print(f"WARNING: {new_strategy} 与 {existing} 相关性 {corr:.2f}，不值得开发")
```

**判断标准：**
- 相关性 > 0.7 → 高度冗余，不开发
- 相关性 0.4-0.7 → 部分重叠，可开发但 portfolio 价值有限
- 相关性 < 0.4 → 有分散价值，优先开发

**快速预判（不需要跑回测）：**
- 使用相同核心指标（如都用 ADX+Supertrend）→ 大概率冗余
- 使用相同分类的指标（如都是 trend 类）→ 中等概率冗余
- 跨分类组合（如 trend + volume + regime）→ 低概率冗余

## Warmup 期计算规范

**warmup 值必须 ≥ 所用指标中最大 period 的 2-3 倍。**

```python
# 示例：策略用了 ema(60) + adx(14) + atr(14)
# 最大 period = 60
# warmup = 60 * 3 = 180
class MyStrategy(TimeSeriesStrategy):
    warmup = 180  # 不是 60！
```

**各频率推荐 warmup：**

| 频率 | 最小 warmup | 说明 |
|------|:---:|------|
| daily | 120-250 bars | 约 6 个月-1 年 |
| 4h | 200-500 bars | 约 3-6 个月 |
| 1h | 500-1000 bars | 约 2-4 个月 |
| 5min/10min | 1000-2000 bars | 约 1-2 周 |

**多周期策略：** warmup 按主频计算，但要确保大周期指标也有足够数据。如 5min 主频 + 4h 方向，warmup 至少 48 * 大周期指标 period * 2（48 = 4h 包含的 5min bar 数）。

## 极端行情处理

**所有策略必须处理以下极端情况（已集成到策略模板中）：**

1. **展期日跳过** — `is_rollover` 日复权跳变导致信号失真
2. **低量 bar 跳过** — 成交量 < 20 日均量的 10%，流动性不足
3. **涨跌停附近不开仓** — AlphaForge 已自动拒绝，策略层也应避免

**连续涨跌停风险：**
- 连续涨停时无法开空、无法平多（被锁住）
- 连续跌停时无法开多、无法平空
- 策略无法控制此风险，但应在 research_log 中记录品种的历史涨跌停频率
- 高频发生涨跌停的品种（如小品种），需要更宽的止损或更低的仓位

## 统计显著性标准

交易次数门槛应根据策略频率调整。门槛设计原则：**降低到"能反映策略是否有效"的最低水平**，过高的门槛会杀掉在短训练段上有效的低频趋势策略（如 daily 策略在 3 个月中趋势段上可能只有 3-5 笔交易）。

| 策略频率 | 最低交易次数 | 说明 |
|---------|:---:|------|
| daily | ≥ 10 笔 | 趋势策略月均 1-2 笔，10 笔 ≈ 半年 |
| 4h | ≥ 20 笔 | 每周 2-3 笔，20 笔 ≈ 2 个月 |
| 1h | ≥ 30 笔 | CLT 最低门槛 |
| 30min | ≥ 50 笔 | |
| 5min/10min/15min | ≥ 80 笔 | 高频交易多，80 笔足够 |

**额外检查（已内置到目标函数 S_quality 维度）：** 如果策略 80% 的利润来自 top 10% 的天数，即使 Sharpe 很高也不可靠。目标函数会自动惩罚利润集中度 > 0.5 的参数组合。

## 策略退化与退役标准

策略上线后需要持续监控，触发以下条件时考虑退役：

| 条件 | 动作 |
|------|------|
| 滚动 6 个月 Sharpe < 0 | 标记为"观察"，降低权重至原来的 50% |
| 连续 3 个月亏损 | 标记为"观察"，降低权重至原来的 50% |
| 滚动 12 个月 Sharpe < -0.5 | 从 portfolio 中移除 |
| 最大回撤超过回测最大回撤的 1.5 倍 | 立即移除，调查原因 |
| 市场结构性变化（如品种规则改变、交易时间变更） | 重新回测验证，不通过则移除 |

**退役流程：**
1. 从 portfolio 权重中移除
2. 在 `research_log/` 中记录退役原因和日期
3. 策略文件保留（不删除），但在文件头标注 `# RETIRED: YYYY-MM-DD 原因`
4. 重新运行 portfolio 构建，重新分配权重

## 预计算模式（on_init_arrays）

**所有策略必须使用预计算模式。** 传统模式（每 bar 重算指标）在高频数据上慢 20-50x。

引擎调用顺序：
```
strategy.on_init(context)              # 1. 初始化状态变量
strategy.on_init_arrays(context, bars) # 2. 预计算所有指标（一次性）
for each bar:
    strategy.on_bar(context)           # 3. 查表 + 交易逻辑
strategy.on_stop(context)              # 4. 清理
```

**策略模板选择：**

| 模板 | 文件 | 适用场景 | 说明 |
|------|------|---------|------|
| **Simple** | `template_simple.py` | **90% 的策略** | 单周期、标准止损/止盈/加仓，推荐首选 |
| **Full** | `template_full.py` | 复杂策略 | 多周期、自定义退出逻辑、regime 切换等高级场景 |
| **Mean Reversion** | `template_mean_reversion.py` | 均值回归策略 | RSI 超买超卖 + 利润目标 + 固定止损 + 时间止损 |
| **Volatility Target** | `template_volatility_target.py` | 波动率管理策略 | 波动率缩放仓位 + 百分比止损，适合波动率驱动的策略 |
| **Time-Based** | `template_time_based.py` | 纯时间退出策略 | 固定持仓周期退出 + 紧急止损，适合事件驱动或季节性策略 |

**5 个模板共覆盖所有常见策略类型。** 选择指南：

| 你想做的 | 选择模板 |
|---------|---------|
| 趋势跟踪（大部分场景） | **Simple** |
| 多周期信号 / regime 切换 | **Full** |
| RSI/BB 均值回归 / 超买超卖 | **Mean Reversion** |
| 波动率目标 / vol-scaling 仓位 | **Volatility Target** |
| 固定持仓天数 / 事件驱动 | **Time-Based** |

大部分策略应使用 `template_simple.py`，它已包含所有必要组件（ATR 止损、分层止盈、金字塔加仓、极端行情过滤）。只有在需要多周期信号、自定义退出逻辑或 regime 切换等复杂功能时，才使用 `template_full.py`。均值回归、波动率目标和时间退出模板各自针对特定策略类型做了专门优化。

**标准策略模板（单周期，包含所有必要组件）：**

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
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

# 加仓配置
SCALE_FACTORS = [1.0, 0.5, 0.25]   # 首仓100% → 加仓50% → 加仓25%
MAX_SCALE = 3                        # 最多3层


class MyStrategy(TimeSeriesStrategy):
    """
    策略简介：ADX 趋势强度 + ROC 动量确认的趋势跟踪策略。

    使用指标：
    - ADX(14): 趋势强度判断，> threshold 时确认趋势存在
    - ROC(20): 动量方向确认，> 0 时确认上涨动量
    - ATR(14): 止损距离计算

    进场条件（做多）：
    - ADX > adx_threshold（趋势足够强）
    - ROC > 0（动量向上）

    出场条件：
    - ATR 追踪止损触发（最高价回撤 N×ATR）
    - 分层止盈（3ATR 减 1/3，5ATR 再减 1/3）
    - 信号反转（ROC < 0）

    优点：双重确认（趋势+动量），减少假信号
    缺点：震荡市中 ADX 可能滞后，导致入场偏晚
    """
    name = "my_strategy"
    warmup = 60   # max(adx_period, roc_period) * 3 = 20 * 3 = 60
    freq = "daily"

    # 可调参数（Optuna 优化范围见注释）
    adx_period: int = 14
    roc_period: int = 20
    adx_threshold: float = 25.0    # Optuna: 15-40
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._adx = None
        self._roc = None
        self._atr = None
        self._avg_volume = None

    def on_init(self, context):
        # 持仓状态
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0          # 当前加仓层数
        self.bars_since_last_scale = 0   # 距上次加仓的bar数
        # 分层止盈标记
        self._took_profit_3atr = False
        self._took_profit_5atr = False

    def on_init_arrays(self, context, bars):
        """一次性预计算所有指标。"""
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()
        volumes = context.get_full_volume_array()

        self._adx = adx(highs, lows, closes, period=self.adx_period)
        self._roc = rate_of_change(closes, self.roc_period)
        self._atr = atr(highs, lows, closes, period=14)

        # 预计算平均成交量（用于低量过滤）
        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)  # 200x faster than Python loop

    def on_bar(self, context):
        i = context.bar_index
        price = context.close_raw
        side, lots = context.position

        # ── 极端行情过滤 ──
        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return  # 展期日跳过
        vol = context.volume
        if self._avg_volume[i] is not None and not np.isnan(self._avg_volume[i]):
            if vol < self._avg_volume[i] * 0.1:
                return  # 低量跳过

        # ── 查表 ──
        adx_val = self._adx[i]
        roc_val = self._roc[i]
        atr_val = self._atr[i]
        if np.isnan(adx_val) or np.isnan(roc_val) or np.isnan(atr_val):
            return

        self.bars_since_last_scale += 1

        # ── 1. 止损检查（保命第一）──
        if side == 1:
            self.highest_since_entry = max(self.highest_since_entry, price)
            trailing_stop = self.highest_since_entry - self.atr_stop_mult * atr_val
            self.stop_price = max(self.stop_price, trailing_stop)  # 只收紧
            if price <= self.stop_price:
                context.close_long()
                self._reset_state()
                return

        # ── 2. 分层止盈检查 ──
        if side == 1 and self.entry_price > 0:
            profit_atr = (price - self.entry_price) / atr_val
            if profit_atr >= 5.0 and not self._took_profit_5atr:
                lots_to_close = max(1, lots // 3)
                context.close_long(lots=lots_to_close)
                self._took_profit_5atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return
            elif profit_atr >= 3.0 and not self._took_profit_3atr:
                lots_to_close = max(1, lots // 3)
                context.close_long(lots=lots_to_close)
                self._took_profit_3atr = True
                self.position_scale = max(0, self.position_scale - 1)
                return

        # ── 3. 信号弱化减仓 ──
        if side == 1 and roc_val < 0:
            context.close_long()  # 主退出信号：动量反转
            self._reset_state()
            return

        # ── 4. 入场逻辑 ──
        if side == 0 and adx_val > self.adx_threshold and roc_val > 0:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # ── 5. 加仓逻辑（金字塔递减）──
        elif side == 1 and self._should_add(price, atr_val, adx_val):
            add_lots = self._calc_add_lots(self._calc_lots(context, atr_val))
            if add_lots > 0:
                context.buy(add_lots)
                self.position_scale += 1
                self.bars_since_last_scale = 0

    def _should_add(self, price, atr_val, adx_val):
        """检查是否满足加仓条件（三大前提缺一不可）"""
        if self.position_scale >= MAX_SCALE:
            return False
        if self.bars_since_last_scale < 10:         # 最小间隔
            return False
        if price < self.entry_price + atr_val:      # 盈利 ≥ 1ATR
            return False
        if adx_val < self.adx_threshold:             # 信号仍然确认
            return False
        return True

    def _calc_add_lots(self, base_lots):
        """金字塔递减手数"""
        factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS) - 1)]
        return max(1, int(base_lots * factor))

    def _calc_lots(self, context, atr_val):
        """标准仓位公式：2% 权益风险"""
        from alphaforge.data.contract_specs import ContractSpecManager
        spec = ContractSpecManager().get(context.symbol)
        stop_distance = self.atr_stop_mult * atr_val * spec.multiplier
        if stop_distance <= 0:
            return 0
        risk_lots = int(context.equity * 0.02 / stop_distance)
        # 保证金上限 30%
        margin_per_lot = context.close_raw * spec.multiplier * spec.margin_rate
        if margin_per_lot <= 0:
            return 0
        max_lots = int(context.equity * 0.30 / margin_per_lot)
        return max(1, min(risk_lots, max_lots))

    def _reset_state(self):
        """平仓后重置所有状态"""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_since_entry = 0.0
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
```

**注意：** 以上是趋势策略模板（只做多）。all_time 策略需额外加入做空逻辑（`context.sell()` + 空头止损 + 空头分层止盈）。

**性能对比：**

| 周期 | ~Bars (5年) | 传统模式 | 预计算模式 | 加速比 |
|------|------------|---------|-----------|--------|
| 1min | ~1,500K | >10min | ~8-15s | ~50x |
| 5min | ~350K | ~98s | ~3-5s | ~25x |
| 30min | ~25K | ~7s | <1s | ~10x |
| 1h | ~12K | ~3s | <0.5s | ~6x |
| daily | ~1.2K | <0.3s | <0.1s | ~3x |

## 多周期支持（Multi-Timeframe）

允许且鼓励在策略中组合不同周期的信号。典型模式：

### V6.0 多周期 API（推荐）

AlphaForge V6.0 引入了 `resample_freqs` 类属性 + `context.get_resampled_bars()` API，替代手动 reshape：

```python
class MultiTFTrend(TimeSeriesStrategy):
    name = "multi_tf_trend"
    freq = "30min"
    resample_freqs = ["4h"]  # V6: 声明需要的重采样频率
    warmup = 960

    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        highs = context.get_full_high_array()
        lows = context.get_full_low_array()

        # 30min 指标
        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

        # V6: 引擎自动重采样，无需手动 reshape
        bars_4h = context.get_resampled_bars("4h")
        _, self._st_dir_4h = supertrend(
            bars_4h.high, bars_4h.low, bars_4h.close,
            self.st_period, self.st_mult)

        # V6: 引擎提供索引映射（避免前视偏差）
        self._4h_map = context.get_resampled_index_map("4h")

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]  # 30min bar → 最近完成的 4h bar
        trend_dir = self._st_dir_4h[j]
        # ... 其余逻辑不变
```

**V6 多周期 vs 旧方式对比：**

| 方面 | 旧方式（手动 reshape） | V6 `resample_freqs` |
|------|----------------------|---------------------|
| 声明 | 无 | `resample_freqs = ["4h"]` |
| 数据获取 | `closes[:trim].reshape(n_4h, step)` | `context.get_resampled_bars("4h")` |
| 索引映射 | `np.maximum(0, (np.arange(n)+1)//step-1)` | `context.get_resampled_index_map("4h")` |
| 边界处理 | 手动 trim | 引擎自动处理 |
| 前视偏差 | 需自行确保 | 引擎保证 |

**常见多周期组合：**

| 大周期（方向） | 小周期（进场） | 适用场景 |
|---------------|---------------|---------|
| daily | 4h | 中长线趋势，日内择时 |
| 4h | 30min/1h | 波段交易 |
| 4h | 5min/10min | 精确入场，紧止损 |
| 1h | 5min | 日内趋势跟踪 |

**注意：多周期不是强制的。** 单周期策略（如 v1-v20 强趋势策略）完全有效。多周期是进阶技术，用于提升 alpha 或降低回撤。

**50策略实验结论：** 单周期平均 Sharpe 1.15 > 多周期平均 Sharpe 0.62。v41-v45 多周期策略远低于单周期平均。小周期噪音是主因（v43 5min 唯一训练集负 Sharpe）。增加复杂度但未增加 alpha。**除非有明确理由，优先用单周期 daily 策略。**

## 失败分析工具

当大量策略优化后表现不佳时，可使用 `analyze_failures.py` 分析失败模式，指导后续策略开发：

```bash
python strategies/all_time/ag/analyze_failures.py
```

**功能：**
- 按策略类别（趋势/均值回归/突破/多周期/自适应）统计失败率
- 识别常见失败模式（信号过稀、参数边界、Sharpe 负值等）
- 输出可操作的改进建议（如哪类策略值得继续开发、哪些应放弃）

**典型用法：** 在完成一批策略的粗调优化后运行，快速定位最有价值的改进方向，避免在低回报的策略类型上浪费时间。

## AlphaForge V6.0 性能优化（必须遵守）

**当前版本：V6.0**（1505 tests） — 5min 回测从 77s → 0.56s（137x 加速），QBase 实测 140-200x 加速。

### P0 必须改（不改就慢 100x）

**1. `context.is_rollover` 替代 `context.current_bar.is_rollover`**

```python
# ❌ 旧方式（每 bar 创建 17 字段 Bar namedtuple）
if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:

# ✅ 新方式（直接数组访问，零对象创建）
if context.is_rollover:
```

同样可用：`context.origin_symbol`（替代 `context.current_bar.origin_symbol`）。

**2. `ContractSpecManager()` 缓存到模块级别**

```python
# ❌ 旧方式（每次交易读 YAML — 6000 笔 = 6000 次文件 IO）
def _calc_lots(self, context, atr_val):
    from alphaforge.data.contract_specs import ContractSpecManager
    spec = ContractSpecManager().get(context.symbol)

# ✅ 新方式（模块级别单例，只读一次）
from alphaforge.data.contract_specs import ContractSpecManager
_SPEC_MANAGER = ContractSpecManager()

class MyStrategy(TimeSeriesStrategy):
    def _calc_lots(self, context, atr_val):
        spec = _SPEC_MANAGER.get(context.symbol)
```

**全局替换命令：**
```bash
# 1. 替换 is_rollover
grep -rl "context.current_bar.is_rollover" strategies/ | xargs sed -i '' \
  's/hasattr(context.current_bar, '\''is_rollover'\'') and context.current_bar.is_rollover/context.is_rollover/g'

# 2. ContractSpecManager 需手动改为模块级缓存
grep -rn "ContractSpecManager()" strategies/ --include="*.py"
```

### P1 推荐改

**3. `signal_mask()` 向量化预筛选（稀疏策略额外 10-50x 加速）**

```python
from alphaforge.indicators import crossover, crossunder

class MyStrategy(TimeSeriesStrategy):
    def signal_mask(self, context, bars):
        """只在信号 bar 调用 on_bar，跳过 95%+ 无信号 bar"""
        return crossover(self._sma_fast, self._sma_slow) | crossunder(self._sma_fast, self._sma_slow)
```

规则：返回 `None` = 不启用 | `np.ndarray[bool]` = 启用 | **有持仓时始终调用 on_bar** | Mask 应保守

**4. `on_init_arrays_static()` — ML 指标只算一次**

```python
class MyStrategy(TimeSeriesStrategy):
    def on_init_arrays_static(self, context, bars):
        """Optuna 优化前只调一次，放重的 ML 指标"""
        features = np.column_stack([rsi_arr, adx_arr, atr_arr])
        self._regime = kmeans_regime(features, period=120)  # ~3s，只算一次

    def on_init_arrays(self, context, bars):
        """每个 trial 调。self._regime 已从缓存注入。"""
        self._sma = sma(closes, self.fast_period)  # 参数相关，每次重算
```

效果：ML 指标 3s × 200 trials = 600s → 只算 1 次 = 3s。

**5. Numba 加速指标库**

```python
from alphaforge.indicators import sma, ema, rsi, atr, macd, bollinger_bands, supertrend, crossover, crossunder
```

全部 `@njit(cache=True)`，C 速度。QBase 自己的 `indicators/` 库仍可用。

**6. 零拷贝数组访问**

```python
closes = context.get_close_array_view(20)  # 只读 view，零拷贝（不可修改）
```

### Portfolio 新功能

```python
config = PortfolioConfig(
    total_capital=5_000_000,
    allocations=[...],
    n_workers=4,                        # 并行回测
    min_capital_per_strategy=500_000,   # 防小权重开不了仓
)
```

### BacktestConfig 新选项（V6.0 扩展至 ~30 参数）

```python
config = BacktestConfig(
    dynamic_margin=True,        # 交割月阶梯保证金（T-3:+3%, T-2:+5%, T-1:+8%）
    time_varying_spread=True,   # 盘口时段价差（开盘/收盘加宽）
    # V6.0 新增
    margin_check_mode="daily",       # 保证金检查: "bar" | "daily"
    rollover_window_bars=20,         # 渐进式换仓
    detect_locked_limit=True,        # 涨跌停锁仓检测
    volume_adaptive_spread=True,     # 成交量自适应价差
    asymmetric_impact=True,          # 非对称市场冲击
    overnight_gap=True,              # 隔夜跳空
    auction_spread=True,             # 集合竞价价差
)
```

已配置动态保证金：I、RB、CU、AG、M。详见 [AlphaForge API — BacktestConfig](../reference/alphaforge-api.md#backtestconfigv60-扩展至-30-参数)。

### QBase 实测性能

| 策略 | 类型 | Before V4 | After V4 | 加速 |
|------|------|:---:|:---:|:---:|
| v1 | 纯 5min | 5.58s | 0.04s | **140x** |
| v121 | 多周期 (daily+5min) | 1.34s | 0.03s | **45x** |
| v14 | 跨品种 (AG+AU) | 1.07s | 0.02s | **54x** |

结果完全一致，纯性能提升零行为变更。

## AlphaForge Context API

```python
# === on_init_arrays 中可用 ===
context.get_full_close_array()      # 完整复权close (np.ndarray, shape=(N,))
context.get_full_high_array()       # 完整high
context.get_full_low_array()        # 完整low
context.get_full_open_array()       # 完整open
context.get_full_volume_array()     # 完整volume
context.get_full_oi_array()         # 完整持仓量
context.get_full_close_raw_array()  # 完整原始close（未复权）

# === V6.0 多周期 API（on_init_arrays 中可用）===
# 需先在类上声明: resample_freqs = ["4h"]
bars_4h = context.get_resampled_bars("4h")       # 重采样后的 BarArray
idx_map = context.get_resampled_index_map("4h")   # 主频→重采样频率的索引映射

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

## 组合回测 (PortfolioBacktester)

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

## 运行回测

```bash
# 单品种单频率
./run.sh strategies/strong_trend/v1.py --symbols AG --freq daily --start 2022

# 多品种测试
./run.sh strategies/strong_trend/v1.py --symbols AG,J,ZC --freq daily --start 2020

# 不同频率
./run.sh strategies/strong_trend/v9.py --symbols AG --freq 60min --start 2024
```

## Report 生成规范

### 目录结构

所有 HTML 报告按策略类型和品种组织：

```
reports/
├── strong_trend/
│   ├── ag/
│   │   ├── portfolio.html      # AG 强趋势 Portfolio 报告
│   │   ├── v7.html ~ v49.html  # 各策略单独报告
│   └── lc/
│       ├── portfolio.html      # LC 强趋势 Portfolio 报告
│       └── v5.html ~ v34.html
├── medium_trend/
│   └── <symbol>/
├── all_time/
│   └── ag/
└── ...
```

### 必须使用优化参数

**生成回测报告或 portfolio 报告时，必须传入 Optuna 优化后的参数。不能用默认参数。**

```python
# ❌ 错误：用默认参数（未经优化）
strat = StrategyV20()
result = engine.run(strat, {'AG': bars})
reporter.generate(result, 'reports/v20.html')  # 结果不准确！

# ✅ 正确：传入优化后的参数
from strategies.strong_trend.optimizer import create_strategy_with_params
import json

with open('strategies/strong_trend/optimization_results.json') as f:
    opt_results = json.load(f)
params = {r['version']: r['best_params'] for r in opt_results}

strat = create_strategy_with_params('v20', params['v20'])
result = engine.run(strat, {'AG': bars}, warmup=strat.warmup)
reporter.generate(result, 'reports/strong_trend/ag/v20.html')
```

**教训（v20 参数 bug）：** 默认参数 v20 AG Sharpe=-0.78，优化后为 1.80。差异巨大。

### 单策略报告

**必须传 `bar_data` 和 `freq` 才有 K 线图 + 进出场标记。不传就没有 K 线。**

```python
from alphaforge.report import HTMLReportGenerator

reporter = HTMLReportGenerator()
reporter.generate(
    result,
    'reports/strong_trend/ag/v20.html',
    bar_data={'AG': bars},  # 必须传这个才有 K-line
    freq='daily',           # K 线频率
)
```

### Portfolio 报告（新 API）

**必须传入 `slot_results` 和 `kline_bars`**，这样报告中会有：
- 每个策略的单独报告 + K 线图 + 进出场标记
- 策略链接和权重信息
- 组合级净值曲线和回撤

```python
from alphaforge.engine.portfolio import PortfolioConfig, StrategyAllocation, PortfolioBacktester
from alphaforge.report import HTMLReportGenerator

# 1. 加载优化参数
with open('strategies/strong_trend/optimization_results.json') as f:
    opt_results = json.load(f)
params_map = {r['version']: r.get('best_params', {}) for r in opt_results}

# 2. 构建 PortfolioConfig（必须传入 params）
config = PortfolioConfig(
    total_capital=3_000_000,
    allocations=[
        StrategyAllocation('v20', 'strategies/strong_trend/v20.py', 'AG', 'daily',
                           weight=0.20, params=params_map['v20']),
        StrategyAllocation('v12', 'strategies/strong_trend/v12.py', 'AG', 'daily',
                           weight=0.144, params=params_map['v12']),
        # ... 其他策略
    ],
    rebalance='none',
)

# 3. 运行回测
runner = PortfolioBacktester(data_dir=get_data_dir())
report = runner.run(config, start='2025-01-01', end='2026-03-01')

# 4. 构建 kline_bars（从 slot_results 提取 K 线数据）
kline_bars = {}
for s in report.slot_results:
    if s.bar_data:
        sym = s.symbol.split(',')[0].strip().upper()
        if sym in s.bar_data:
            kline_bars[sym] = s.bar_data[sym]

# 5. 生成报告（必须传 slot_results + kline_bars）
reporter = HTMLReportGenerator()
reporter.generate_portfolio_report(
    [s.result for s in report.slot_results],
    report.combined_result,
    'reports/strong_trend/ag/portfolio.html',
    slot_results=report.slot_results,     # 关键：单策略报告+链接+权重
    kline_bars=kline_bars,                # 关键：K 线图+进出场标记
    kline_freq='daily',                   # K 线频率
)
```

**不传 `slot_results` 就没有单策略报告和 K 线。这是必须的。**

### 当前 Portfolio 报告

**Strong Trend 使用统一通用 Portfolio（Portfolio C），同一组策略和权重跑任何品种：**

| 权重 | 策略 | 指标组合 | 频率 |
|:---:|------|---------|:---:|
| 25% | v12 | Aroon + PPO + Volume Momentum | daily |
| 20% | v8 | LinReg + Choppiness + OBV | daily |
| 20% | v11 | Vortex + ROC + OI Momentum | daily |
| 15% | v34 | McGinley + PPO + OI Momentum | daily |
| 20% | v31 | TEMA + Fisher + OBV | 4h |

| 品种 | Sharpe | Return | MaxDD | 路径 |
|------|:------:|:------:|:-----:|------|
| AG | 2.58 | 66.13% | -12.80% | `reports/strong_trend/ag/portfolio.html` |
| LC | 2.37 | 19.19% | -3.99% | `reports/strong_trend/lc/portfolio.html` |

## 频率体系

**支持频率：** AlphaForge 支持任意频率：`<N>min`（如 `1min`, `5min`, `20min`）、`<N>h`（如 `1h`, `4h`）、`daily`

**最低频率: 5min。** 不使用 1min（噪音太大，回测慢）。

**开发原则：从小周期到大周期**

策略开发时**优先从小周期开始**，逐步放大到大周期验证：

```
5min → 10min → 30min → 1h → 4h → daily
```

理由：小周期产生更多交易信号，alpha 更细腻。如果一个逻辑在 5min 上有效，再在 30min/daily 上验证稳健性。反之，如果只在 daily 有效，可能只是偶然。

**频率选择指南：**

| 策略类型 | 推荐频率 | 理由 |
|---------|---------|------|
| 强趋势 | daily / 4h | 持仓周期长，高频噪音无意义 |
| 中趋势 | 4h / 1h | 需要更快的入场/出场响应 |
| 弱趋势/波段 | 1h / 30min | 捕捉短期波动 |
| 震荡/均值回归 | 30min / 5min | 快进快出，需要精确时机 |
| 多周期协作 | 5min（主频）+ 4h（方向） | 精确入场 + 趋势过滤 |

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

## 单策略开发流程

1. **确定市场状态** — strong_trend / medium_trend / weak_trend / mean_reversion / all_time
2. **选 1-3 个指标** — 从 320 个中选（可 research 新指标），确保互补，参考 `research_log/` 中已验证有效的组合
3. **选频率** — 从小周期开始 (5min → 10min → 30min → 1h → 4h → daily)，可考虑多周期协作
4. **写策略** — 继承 `TimeSeriesStrategy`，放入对应目录，包含完整文档（简介、指标、进出场、优缺点）
5. **快速可行性验证（训练集）** — 7 项检查（见上方）
6. **Optuna 优化** — `optimizer.py --strategy vN --trials 150`，多品种平均 Sharpe
7. **测试集验证** — `validate_and_iterate.py`，在测试集上验证
8. **记录结果** — 写入 `research_log/`，包含指标、参数、结果、结论

## 批量开发流程（推荐）

一次性开发 10-20 个不同指标组合的策略版本，然后批量优化筛选：

```bash
# 1. 开发 v1-v20（可用 subagent 并行）
# 2. 批量优化
python strategies/strong_trend/optimizer.py --strategy all --trials 150
# 3. 测试集验证 + 迭代
python strategies/strong_trend/validate_and_iterate.py
# 4. 淘汰 Sharpe < 0 的，保留最优
```
