# QBase — Agent 开发指南

QBase 是量化策略研究工作区。指标库 + 按品种组织的策略开发，回测统一用 AlphaForge。

## 项目结构

```
QBase/
├── indicators/                  # 320个指标（纯numpy函数），详见 indicators/indicators.md
│   ├── momentum/                # 35个动量/振荡类
│   ├── trend/                   # 35个趋势类
│   ├── volatility/              # 35个波动率类
│   ├── volume/                  # 39个成交量/持仓类
│   ├── microstructure/          # 15个市场微观结构
│   ├── ml/                      # 65个机器学习/统计学习
│   ├── regime/                  # 33个行情状态识别
│   ├── seasonality/             # 15个季节性/日历效应
│   ├── spread/                  # 25个跨品种价差/比值
│   └── structure/               # 23个持仓结构分析
├── portfolio/                   # 通用 Portfolio 构建工具
│   ├── builder.py               # 组合构建器（穷举/贪心 + HRP + Bootstrap）
│   └── scorer.py                # 组合评分器（4维12指标，0-100分）
├── strategies/                  # 策略库
│   ├── strong_trend/            # 强趋势策略 (v1-v50 + optimizer.py)
│   │   ├── v1.py ~ v50.py      # 50个策略（不同指标组合）
│   │   ├── optimizer.py         # Optuna 参数优化器
│   │   ├── validate_and_iterate.py  # 测试集验证+迭代优化
│   │   ├── optimization_results.json
│   │   └── portfolio/           # 构建结果（weights_ag.json, scores.json 等）
│   ├── medium_trend/            # 中趋势策略（品种无关，待开发）
│   ├── weak_trend/              # 弱趋势策略（品种无关，待开发）
│   ├── mean_reversion/          # 震荡策略（品种无关，待开发）
│   └── all_time/                # 全时间策略（按品种，覆盖所有行情状态）
│       ├── ag/                  # 白银全时间策略 (v1-v100 + optimizer.py)
│       │   ├── v1.py ~ v100.py # 100个策略（多空，5类：趋势/均值回归/突破/多周期/混合）
│       │   ├── optimizer.py    # Optuna 自动参数检测 + 优化
│       │   ├── build_portfolio.py  # Portfolio 构建入口
│       │   ├── optimization_results.json
│       │   └── portfolio/      # 构建结果
│       ├── i/                   # 铁矿石（待开发）
│       └── .../                 # 其他品种按需添加
├── trend/                       # 行情时段参考数据
│   ├── RALLIES.md               # 强趋势时段表（19品种，涨幅99%-907%）
│   └── MEDIUM_TRENDS.md         # 中趋势时段表（40段，涨幅20%-80%）
├── screener/                    # 品种筛选器
├── fundamental/                 # 基本面量化（预留）
├── research_log/                # 实验记录 + 经验教训
│   ├── strong_trend.md          # 强趋势策略完整结果
│   └── lessons_learned.md       # 50策略实战经验总结
├── reports/                     # HTML 回测报告
├── tests/                       # 指标单元测试
├── tasks/                       # 开发进度
│   └── todo.md                  # 待办事项
├── config.yaml                  # AlphaForge路径等配置
├── conftest.py                  # sys.path 配置（pytest自动加载）
└── run.sh                       # 策略运行入口
```

## 开发流水线概览

QBase 的策略开发遵循 **5 步流水线**，每一步有明确的输入、输出和标准：

```
第一步         第二步           第三步            第四步           第五步
指标池  ──→  策略开发  ──→  单策略优化  ──→  测试集验证  ──→  Portfolio 构建
(320+个)     (编写策略)     (训练集调参)     (只读验证)      (组合+赋权)
```

| 步骤 | 输入 | 输出 | 关键原则 |
|------|------|------|---------|
| 第一步：指标池 | 研究/灵感 | 纯 numpy 指标函数 | 纯函数、分类入库 |
| 第二步：策略开发 | 1-3 个指标 | 完整策略代码 | 预计算模式、可行性验证 |
| 第三步：单策略优化 | 训练集数据 | 最优参数 | 只用训练集、参数 ≤ 5 |
| 第四步：测试集验证 | 锁定参数 | 验证报告 | 测试集只读、不淘汰 |
| 第五步：Portfolio 构建 | 多策略收益曲线 | 权重分配 | 互补 > 堆砌、评分 ≥ B+ |

---

## 第一步：指标池（Indicator Pool）

### 指标库概览

**完整指标目录见 [`indicators/indicators.md`](indicators/indicators.md)。**

10 大类 320 个指标，纯 numpy 函数（numpy in → numpy out）：

| 分类 | 数量 | 说明 |
|------|:---:|------|
| momentum | 35 | 动量/振荡类 |
| trend | 35 | 趋势类 |
| volatility | 35 | 波动率类 |
| volume | 39 | 成交量/持仓类 |
| microstructure | 15 | 市场微观结构 |
| ml | 65 | 机器学习/统计学习 |
| regime | 33 | 行情状态识别 |
| seasonality | 15 | 季节性/日历效应 |
| spread | 25 | 跨品种价差/比值 |
| structure | 23 | 持仓结构分析 |

### 新指标开发规则

**指标来源不限于 320 个库存指标。** 可以随时 research 并开发新指标加入库中，例如：
- 品种特有指标（如铁煤比 I/J ratio、金银比 AU/AG ratio）
- 跨品种价差/比值指标
- 基本面衍生指标（库存、持仓结构等）
- 自定义统计量
- 网上 research 发现的有效指标

开发新指标时**必须**：
1. 放入 `indicators/` 对应分类，保持纯函数风格（numpy in → numpy out）
2. **更新 `indicators/indicators.md`**，添加新指标的名称、文件、函数签名

### 分类规则（10 个现有分类）

| 分类 | 放入 | 示例 |
|------|------|------|
| 品种比值/价差 | `indicators/spread/` | 铁煤比、金银比 |
| 跨品种相关性 | `indicators/spread/` | 板块动量扩散 |
| 波动率衍生 | `indicators/volatility/` | 自定义波动率模型 |
| 量价/持仓衍生 | `indicators/volume/` | OI 结构分析 |
| 趋势衍生 | `indicators/trend/` | 自定义趋势评分 |
| 市场微观结构 | `indicators/microstructure/` | 流动性、价格冲击 |
| 机器学习特征 | `indicators/ml/` | Kalman、PCA、聚类 |
| 行情状态识别 | `indicators/regime/` | 趋势/震荡判断 |
| 季节性/日历 | `indicators/seasonality/` | 月效应、节假日 |
| 持仓结构 | `indicators/structure/` | OI 分析、资金流向 |

如果现有 10 个分类都不合适，可以新建分类文件夹。

### 各分类使用规范

**原始 4 类（momentum/trend/volatility/volume）：** 标准 OHLCV 数据即可，直接在 `on_init_arrays` 中使用。

**spread 类：** 需要多品种数据。策略中需额外加载第二品种的价格数组：
```python
# 示例：在 AG 策略中使用金银比
from indicators.spread.gold_silver_ratio import gold_silver_ratio
# 需要在 on_init_arrays 中同时获取 AU 的 close 数组
```

**seasonality 类：** 需要 datetime 数组。通过 `context.get_full_datetime_array()` 获取：
```python
from indicators.seasonality.weekday_effect import weekday_effect
datetimes = context.get_full_datetime_array()
effect = weekday_effect(closes, datetimes, lookback=252)
```

**ml 类：** 多数需要 features_matrix（多列特征矩阵）。通常先计算若干基础指标，再组合成矩阵传入：
```python
from indicators.ml.kmeans_regime import kmeans_regime
features = np.column_stack([rsi_arr, adx_arr, atr_arr])
regime = kmeans_regime(features, period=120, n_clusters=3)
```

**regime 类：** 用于判断当前行情状态（趋势/震荡/突破），适合作为策略的过滤层而非核心信号。

**structure 类：** 依赖 OI（持仓量）数据，通过 `context.get_full_oi_array()` 获取。适合期货品种，非期货品种不可用。

**microstructure 类：** 分析市场微观结构（流动性、价格冲击等），适合高频策略（5min/10min）的辅助过滤。

---

## 第二步：策略开发（Strategy Development）

### 指标选择规则

每个策略使用 **1-3 个指标**，只要有效就行，不强制用满 3 个。

**选择思路（参考，不强制）：**
- 趋势策略常见组合: 趋势判断 + 动量确认 + 波动率过滤
- 震荡策略常见组合: 振荡器 + 波动率 + 量价确认
- 单指标策略也完全可行，如果该指标本身信号足够强

**核心原则：力求高效，不堆砌。** 1 个强指标 > 3 个弱指标。

### 方向约束

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

### 仓位管理规则（加仓 / 减仓 / 止损）

**所有规则已集成到策略模板中（见预计算模式章节）。以下是规则总结：**

#### 加仓：金字塔递减

配置：`SCALE_FACTORS = [1.0, 0.5, 0.25]`，最多 3 层。

**加仓三大前提（缺一不可）：**
1. **盈利前提** — 浮盈 ≥ 1×ATR，绝不在亏损时加仓
2. **信号确认** — 策略指标仍然给出加仓信号
3. **最小间隔** — 两次加仓至少间隔 10 根 bar

#### 减仓：分层止盈 + 信号弱化

| 浮盈达到 | 动作 | 剩余仓位 |
|---------|------|---------|
| 3×ATR | 减仓 1/3 | 67% |
| 5×ATR | 再减 1/3 | 33% |
| 追踪止损触发 | 平掉剩余 | 0% |

**信号弱化减仓：** 主退出信号触发 → 全部平仓

**关键原则：**
- 不要一次全部减完 — 至少保留 25% 底仓直到主退出信号
- 减仓后的止损只能收紧，不能放宽

#### 止损：ATR 追踪止损

**每个策略必须内置 ATR 止损。**

- **初始止损：** 入场价 ± N×ATR（默认 N=3.0，Optuna 范围 2.0-5.0）
- **追踪止损：** 最高价（多头）或最低价（空头）回撤 N×ATR 触发
- **只收紧不放宽**
- 加仓后止损不变，追踪止损继续跟随
- 空头止损逻辑对称（仅 all_time 策略）

**on_bar 内执行顺序：**
1. 极端行情过滤（展期日/低量/涨跌停）
2. **止损检查（保命第一）**
3. 分层止盈检查
4. 信号弱化减仓/主退出
5. 入场逻辑
6. 加仓逻辑

### 策略目录结构

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

### 策略文档要求

每个策略文件必须在类的 docstring 中包含：
1. **策略简介** — 一句话描述核心逻辑
2. **使用指标** — 列出所有指标及其作用
3. **进场条件** — 做多/做空的具体触发条件
4. **出场条件** — 止损、止盈、信号反转的具体条件
5. **优点** — 该策略的核心优势
6. **缺点** — 已知的局限性和弱点

### 策略命名

新策略统一以 v1-v500 命名，优先扩展新版本而非修改旧版本。

### 数据分割标准

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

### 快速可行性验证（7 项检查）

写完代码后、跑优化前，必须先做：
1. 在训练集上跑 1-2 个品种的基础回测
2. 确认策略有交易信号、不报错
3. 交易次数 ≥ 10（排除僵尸策略）
4. 做多/做空方向与指标信号一致
5. 止损正常生效
6. 信号频率合理（不过密也不过稀）
7. 训练集单品种 Sharpe > 0（否则不值得进入优化）

### 策略去重/冗余检查

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

### Warmup 期计算规范

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

### 极端行情处理

**所有策略必须处理以下极端情况（已集成到策略模板中）：**

1. **展期日跳过** — `is_rollover` 日复权跳变导致信号失真
2. **低量 bar 跳过** — 成交量 < 20 日均量的 10%，流动性不足
3. **涨跌停附近不开仓** — AlphaForge 已自动拒绝，策略层也应避免

**连续涨跌停风险：**
- 连续涨停时无法开空、无法平多（被锁住）
- 连续跌停时无法开多、无法平空
- 策略无法控制此风险，但应在 research_log 中记录品种的历史涨跌停频率
- 高频发生涨跌停的品种（如小品种），需要更宽的止损或更低的仓位

### 统计显著性标准

交易次数门槛应根据策略频率调整：

| 策略频率 | 最低交易次数 | 说明 |
|---------|:---:|------|
| daily | ≥ 30 笔 | 约 2-3 年数据 |
| 4h | ≥ 50 笔 | 约 1-2 年数据 |
| 1h | ≥ 80 笔 | 约 6 个月-1 年 |
| 30min | ≥ 100 笔 | 约 3-6 个月 |
| 5min/10min | ≥ 150 笔 | 约 1-3 个月 |

**为什么不同频率门槛不同：**
- 高频策略交易多，30 笔可能只覆盖极短时间段，不能代表多种市场环境
- 低频策略交易少，30 笔已经跨越多年不同行情
- 核心原则：交易样本必须覆盖至少 1 个完整的牛熊周期

**额外检查：** 如果策略 80% 的利润来自 1-2 笔交易，即使 Sharpe 很高也不可靠（运气成分大）。检查方法：去掉最盈利的 10% 交易后，Sharpe 是否仍 > 0。

### 策略退化与退役标准

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

### 预计算模式（on_init_arrays）

**所有策略必须使用预计算模式。** 传统模式（每 bar 重算指标）在高频数据上慢 20-50x。

引擎调用顺序：
```
strategy.on_init(context)              # 1. 初始化状态变量
strategy.on_init_arrays(context, bars) # 2. 预计算所有指标（一次性）
for each bar:
    strategy.on_bar(context)           # 3. 查表 + 交易逻辑
strategy.on_stop(context)              # 4. 清理
```

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

### 多周期支持（Multi-Timeframe）

允许且鼓励在策略中组合不同周期的信号。典型模式：

**多周期策略模板（30min 主频 + 4h 方向，包含所有必要组件）：**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # QBase root
import conftest

import numpy as np
from alphaforge.strategy.base import TimeSeriesStrategy
from indicators.momentum.rsi import rsi
from indicators.trend.supertrend import supertrend
from indicators.volatility.atr import atr
from strategies.all_time.ag.strategy_utils import fast_avg_volume, compute_tradeable_mask

SCALE_FACTORS = [1.0, 0.5, 0.25]
MAX_SCALE = 3


class MultiTFTrend(TimeSeriesStrategy):
    """
    策略简介：4h Supertrend 定方向 + 30min RSI 超卖入场的多周期趋势策略。

    使用指标：
    - Supertrend(10, 3.0) [4h]: 大周期趋势方向过滤
    - RSI(14) [30min]: 小周期超卖入场信号
    - ATR(14) [30min]: 止损距离计算

    进场条件（做多）：
    - 4h Supertrend 方向 = 1（上升趋势）
    - 30min RSI < 30（超卖回调入场）

    出场条件：
    - ATR 追踪止损 / 分层止盈 / 4h 趋势反转

    优点：大周期过滤噪音，小周期精确入场，回撤更可控
    缺点：多周期信号可能错位，warmup 需要更多数据
    """
    name = "multi_tf_trend"
    freq = "30min"
    warmup = 960  # 4h Supertrend 需要 step(8) * period(10) * 3 ≈ 240，取更保守值

    # 可调参数
    rsi_period: int = 14
    rsi_entry: float = 30.0         # Optuna: 20-40
    st_period: int = 10
    st_mult: float = 3.0
    atr_stop_mult: float = 3.0     # Optuna: 2.0-5.0

    def __init__(self):
        super().__init__()
        self._rsi = None
        self._atr = None
        self._avg_volume = None
        self._st_dir_4h = None
        self._4h_map = None

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
        n = len(closes)
        step = 8  # 30min × 8 = 4h

        # 30min 指标
        self._rsi = rsi(closes, self.rsi_period)
        self._atr = atr(highs, lows, closes, period=14)

        # 预计算平均成交量
        window = 20
        self._avg_volume = fast_avg_volume(volumes, window)  # 200x faster than Python loop

        # 聚合 → 4h bars
        n_4h = n // step
        trim = n_4h * step
        closes_4h = closes[:trim].reshape(n_4h, step)[:, -1]
        highs_4h = highs[:trim].reshape(n_4h, step).max(axis=1)
        lows_4h = lows[:trim].reshape(n_4h, step).min(axis=1)

        _, self._st_dir_4h = supertrend(
            highs_4h, lows_4h, closes_4h, self.st_period, self.st_mult)

        # 映射: 30min bar i → 最近完成的 4h bar index（避免前视偏差）
        self._4h_map = np.maximum(0, (np.arange(n) + 1) // step - 1)

    def on_bar(self, context):
        i = context.bar_index
        j = self._4h_map[i]
        price = context.close_raw
        side, lots = context.position

        # ── 极端行情过滤 ──
        if hasattr(context.current_bar, 'is_rollover') and context.current_bar.is_rollover:
            return
        vol = context.volume
        if not np.isnan(self._avg_volume[i]) and vol < self._avg_volume[i] * 0.1:
            return

        # ── 查表 ──
        rsi_val = self._rsi[i]
        atr_val = self._atr[i]
        trend_dir = self._st_dir_4h[j]
        if np.isnan(rsi_val) or np.isnan(atr_val):
            return

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

        # ── 3. 主退出信号：4h 趋势反转 ──
        if side == 1 and trend_dir != 1:
            context.close_long()
            self._reset_state()
            return

        # ── 4. 入场 ──
        if side == 0 and trend_dir == 1 and rsi_val < self.rsi_entry:
            base_lots = self._calc_lots(context, atr_val)
            if base_lots > 0:
                context.buy(base_lots)
                self.entry_price = price
                self.stop_price = price - self.atr_stop_mult * atr_val
                self.highest_since_entry = price
                self.position_scale = 1
                self.bars_since_last_scale = 0

        # ── 5. 加仓 ──
        elif side == 1 and self.position_scale < MAX_SCALE:
            if (self.bars_since_last_scale >= 10
                    and price > self.entry_price + atr_val
                    and trend_dir == 1 and rsi_val < 40):
                factor = SCALE_FACTORS[min(self.position_scale, len(SCALE_FACTORS)-1)]
                add = max(1, int(self._calc_lots(context, atr_val) * factor))
                context.buy(add)
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
        self.position_scale = 0
        self.bars_since_last_scale = 0
        self._took_profit_3atr = False
        self._took_profit_5atr = False
```

**注意：** 以上是趋势策略模板（只做多）。all_time 多周期策略需额外加入做空逻辑。

**常见多周期组合：**

| 大周期（方向） | 小周期（进场） | 适用场景 |
|---------------|---------------|---------|
| daily | 4h | 中长线趋势，日内择时 |
| 4h | 30min/1h | 波段交易 |
| 4h | 5min/10min | 精确入场，紧止损 |
| 1h | 5min | 日内趋势跟踪 |

**注意：** 多周期不是强制的。单周期策略（如 v1-v20 强趋势策略）完全有效。多周期是进阶技术，用于提升 alpha 或降低回撤。

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

### 频率体系

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

### 品种筛选器

```python
from screener.scanner import scan

# 找趋势最强的品种（训练用）
trend_top = scan(mode="trend", start="2024-01-01", end="2025-01-01", top_n=10)

# 找震荡特征明显的品种
mr_top = scan(mode="mean_reversion", start="2024-01-01", top_n=10)

# 找波动率收缩的品种（潜在突破）
breakout_top = scan(mode="breakout", start="2024-01-01", top_n=10)
```

### 单策略开发流程

1. **确定市场状态** — strong_trend / medium_trend / weak_trend / mean_reversion / all_time
2. **选 1-3 个指标** — 从 320 个中选（可 research 新指标），确保互补，参考 `research_log/` 中已验证有效的组合
3. **选频率** — 从小周期开始 (5min → 10min → 30min → 1h → 4h → daily)，可考虑多周期协作
4. **写策略** — 继承 `TimeSeriesStrategy`，放入对应目录，包含完整文档（简介、指标、进出场、优缺点）
5. **快速可行性验证（训练集）** — 7 项检查（见上方）
6. **Optuna 优化** — `optimizer.py --strategy vN --trials 150`，多品种平均 Sharpe
7. **测试集验证** — `validate_and_iterate.py`，在测试集上验证
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

---

## 第三步：单策略优化（Single Strategy Optimization）

**所有优化在训练集上完成，测试集不参与任何优化决策。**

### 可调参数分类与优先级

| 优先级 | 参数类别 | 示例 | 说明 |
|:---:|------|------|------|
| **必须优化** | 信号阈值 | ADX > 25、RSI < 30 | 对策略表现最敏感 |
| **必须优化** | 止损参数 | ATR 止损倍数 (2.0-5.0) | 直接影响风险收益比 |
| **可以优化** | 指标 period | RSI period (10-20) | 范围要窄，不给太大空间 |
| **尽量不动** | 加仓减仓结构 | 金字塔 1.0/0.5/0.25 | 固定结构，避免过拟合 |
| **尽量不动** | 分层止盈倍数 | 3ATR/5ATR | 通用规则，不需要按策略调 |

**总可调参数 ≤ 5 个，范围窄（2-3x）。** 参数越少、范围越窄，过拟合风险越低。

### 分阶段优化（粗调 → 精调）

**不要直接 200 trials 全范围搜索。** 分两阶段效率更高：

**第一阶段 — 粗调（30 trials）：**
- 大步长（period 步长 5，阈值步长 0.05）
- 快速锁定参数大致范围
- 内置 probe 机制：先跑 5 个 trials，如果全部报错（-999）则跳过该策略

```bash
# 粗调：大范围、大步长、30 trials（含 5 trial probe）
python optimizer.py --strategy v1 --trials 30 --phase coarse

# 批量粗调
python optimizer.py --strategy v1-v50 --trials 30 --phase coarse

# 全部粗调
python optimizer.py --strategy all --trials 30 --phase coarse
```

**第二阶段 — 精调（50 trials）：**
- 在粗调最优附近缩小范围至 30%、步长减半
- 精确找到最优参数

```bash
# 精调：自动读取粗调结果，窄范围、细步长
python optimizer.py --strategy v1 --trials 50 --phase fine
```

### 优化效率注意事项

**速度瓶颈分析：**
- 数据加载：每个策略 ~3-5 秒（StrategyOptimizer 内部只加载一次）
- ML 指标计算：每个 trial 的 `on_init_arrays` 都要重算（参数变了），HMM/K-Means 等较慢
- 回测本身：daily ~4s/trial，4h ~4.5s/trial，1h ~7s/trial

**并行建议：**
- 不同策略之间用多进程并行（2-3 个进程，不要超过 CPU 核数的 50%）
- 同一策略内 Optuna 不建议并行（n_jobs=1），因为 TPE 采样是顺序依赖的

### 早停与淘汰标准

**Probe 早停机制（optimizer.py 内置）：**
- 每个策略先跑 5 个 probe trials
- 如果 5 个 trials 全部返回 -999（策略代码报错），立即跳过，不浪费剩余 trials
- 节省大量时间（有 bug 的策略不需要跑完全部 30 trials）

**Sharpe 负值不跳过：** 粗调 Sharpe < 0 的策略直接用粗调最优参数，不进精调（节省时间），但保留在策略池中（portfolio 可能用作 hedge）。

**优化阶段唯一淘汰标准：** 训练集交易次数不达标（见统计显著性标准表）则淘汰。Sharpe 为负不淘汰——portfolio 构建时负 Sharpe + 负相关的策略可能比正 Sharpe + 高相关的更有价值。

### 目标函数

**趋势策略（多品种）：**
```python
# 基础目标：多品种平均 Sharpe
objective = mean(sharpe across training_symbols)

# 加一致性 bonus：奖励所有品种都盈利的参数
objective = mean_sharpe + 0.3 * min_sharpe
```

**All-Time 单品种：**
```python
# 单品种训练集 Sharpe
objective = sharpe_on_training_set
```

**复合目标（推荐）：**

纯 Sharpe 最大化可能选出回撤很大但收益更大的参数。加入回撤惩罚：

```python
# Sharpe - 回撤惩罚
objective = sharpe - 0.5 * max(0, abs(max_drawdown) - 0.20)
# MaxDD > 20% 时开始惩罚，每超 1% 扣 0.5 分
```

### 参数交互处理

**参数之间可能耦合：** 比如 RSI period 和 RSI 阈值——period 短了，波动大，阈值要调松。

- Optuna TPE 能捕捉部分交互，但参数 > 5 个时交互空间爆炸
- 如果发现两个参数强耦合，考虑固定其中一个或合并为比值参数

### 稳健性检查（参数高原 vs 参数尖峰）

Optuna 找到最优参数后，不是直接用，而是检查周围邻域：

```python
# 最优参数邻域检查
best_params = study.best_params  # 如 adx_threshold=25

# 在 ±20% 范围内采样 10 组邻近参数
for delta in [-2, -1, 0, 1, 2]:
    neighbor = {**best_params, "adx_threshold": 25 + delta}
    sharpe = evaluate(neighbor)
    print(f"adx={25+delta}: Sharpe={sharpe:.3f}")

# 如果最优 Sharpe=1.5 但邻居都 < 0.3 → 噪音峰值，不可靠
# 应该选一片区域都 Sharpe > 0.8 的"高原"中心
```

**判断标准：**
- 邻域内 80% 参数组合 Sharpe > 最优的 50% → 稳健，可用
- 邻域内 Sharpe 方差 > 最优的 50% → 不稳健，考虑放弃或简化策略

### 频率也是优化维度

同一策略在不同频率上表现可能差很多。**在 fine-tune 参数前先选频率：**

```bash
# 用默认参数在多频率上快速跑一遍
for freq in 5min 30min 1h 4h daily; do
    ./run.sh strategy.py --symbols AG --freq $freq --start 2013 --end 2021
done

# 选 Sharpe 最高的频率，再在该频率上 fine-tune 参数
```

### 并行优化

- 不同策略之间完全独立，可以用 subagent 并行优化
- 同一策略内 Optuna 支持 `n_jobs` 并行 trials
- 批量优化：`optimizer.py --strategy all --trials 150`

### 优化器位置与配置

每个市场状态目录下有独立的 `optimizer.py`：

```
strategies/strong_trend/optimizer.py     # 强趋势优化器
strategies/all_time/ag/optimizer.py      # AG 全时间优化器
strategies/medium_trend/optimizer.py     # 中趋势优化器（待开发）
```

**优化结果保存：**
- `optimization_results.json` — 每个策略的最优参数和 Sharpe
- 粗调和精调的结果都保存，方便对比

### 非标准频率支持

优化器自动处理 4h/1h 等非原生频率：
- `4h`: 加载 60min 数据，每 4 根合成 1 根 4h bar
- `1h`: 直接用 60min
- `20min`: 加载 10min，每 2 根合成

---

## 第四步：测试集验证（Test Set Validation）

### 核心原则

**测试集是只读的。不能因为测试集结果反过来修改任何参数或逻辑。**

### 验证规则

1. **参数锁定** — 用训练集优化出的参数原封不动跑测试集，不管结果好坏都不回去改
2. **不淘汰策略** — 测试集 Sharpe 为负不淘汰（portfolio 层面负相关策略有对冲价值），唯一淘汰条件仍然是交易次数不达标
3. **如果看了测试集结果再调参，测试集就变成了第二个训练集，失去验证意义**

### 详细结果记录

每个策略的测试集验证结果必须完整记录到 `research_log/` 中：

**收益指标：**
- 总收益率、年化收益率
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Profit Factor（总利润 / 总亏损）

**风险指标：**
- 最大回撤（幅度 + 持续天数）
- CVaR 95%（最差 5% 情况的平均损失）
- 最大单日亏损
- 年化波动率

**交易统计：**
- 总交易次数
- 胜率
- 平均盈利 / 平均亏损（盈亏比）
- 最大单笔盈利 / 最大单笔亏损
- 平均持仓时间（bar 数）
- 最长持仓 / 最短持仓
- 做多次数 / 做空次数（all_time 策略）

**与训练集对比：**
- 测试集 Sharpe / 训练集 Sharpe 比值
- 训练集 vs 测试集的 MaxDD 对比
- 训练集 vs 测试集的交易次数对比
- 训练集 vs 测试集的平均持仓时间对比

**行为一致性检查：**
- 平均持仓时间是否与训练集一致（差异 > 3 倍需标注）
- 交易频率是否与训练集一致（差异 > 3 倍需标注）
- 做多/做空比例是否与训练集一致（all_time 策略）
- 如有异常行为，在 research_log 中标注原因分析

### 结果解读（仅标注，不淘汰）

| 训练集 vs 测试集 | 解读 | 标注 |
|------|------|------|
| 测试集 Sharpe ≥ 训练集 50% | 正常，泛化能力可以 | 无需特别标注 |
| 测试集 Sharpe 远低于训练集 | 可能过拟合 | 标注"疑似过拟合" |
| 测试集 Sharpe 反而更好 | 可能是运气或市场环境恰好匹配 | 标注"测试集偏高，需观察" |
| 测试集交易行为异常 | 策略逻辑在新数据上不稳定 | 标注"行为异常" + 具体描述 |

### 记录格式模板

```markdown
## YYYY-MM-DD <品种> <策略名> 测试集验证

### 策略信息
- **版本**: vN
- **指标**: 指标1(参数) + 指标2(参数) + 指标3(参数)
- **频率**: daily / 4h / 1h / ...
- **优化参数**: param1=X, param2=Y, param3=Z

### 训练集结果（参考）
- **期间**: YYYY-MM-DD ~ YYYY-MM-DD
- **Sharpe**: X.XX | **MaxDD**: X.X% | **Trades**: N
- **年化收益**: X.X% | **胜率**: X.X%

### 测试集结果
- **期间**: YYYY-MM-DD ~ YYYY-MM-DD
- **收益**: 总收益=X.X%, 年化=X.X%
- **风险调整**: Sharpe=X.XX, Sortino=X.XX, Calmar=X.XX
- **风险**: MaxDD=X.X% (持续N天), CVaR95=X.X%, 最大单日亏损=X.X%
- **交易**: N笔, 胜率=X.X%, 盈亏比=X.XX, 平均持仓=N bars
- **Omega**: X.XX | **Profit Factor**: X.XX

### 训练集/测试集对比
- **Sharpe 比值**: 测试/训练 = X.XX
- **行为一致性**: 正常 / 异常（描述）

### 标注
- 无 / 疑似过拟合 / 测试集偏高 / 行为异常

### 结论
一句话总结
```

---

## 第五步：Portfolio 构建（Portfolio Construction）

### 核心原则

**组合的唯一目标：最大化组合风险调整收益（Sharpe + CVaR 约束），同时控制回撤。**

不是把好策略堆在一起，而是找到一组**互补**的策略。一个负 Sharpe 但与其他策略负相关的策略，可能比一个正 Sharpe 但高度相关的策略更有价值。

### 数据分割（三段式）

构建 Portfolio 本身也是一种"模型选择"，必须防止选择偏差：

```
全部数据
├── 训练集 (60%)    → 策略参数优化（Optuna）
├── 验证集 (20%)    → 策略选择 + 权重优化（Step 1-5）
└── 测试集 (20%)    → 最终评估（不改任何东西，只看结果）
```

**绝不在测试集上做任何决策。** 验证集上选策略、算权重；测试集只用于最终报告。如果测试集表现远差于验证集，说明过拟合了，需要回去简化。

### 构建流程

```
开发 N 个策略 → 全部在验证集回测 → 穷举/贪心组合优化（含回撤重叠分析）
→ 协方差收缩 + Sharpe 加权 HRP → 权重上限裁剪
→ Leave-one-out 检验 → Bootstrap 稳健性验证
→ 测试集最终评估 → 最终 portfolio
```

### Step 1: 组合优化（策略选择）

**不使用 Sharpe 门槛过滤，不使用硬性相关性阈值。** 以组合 Sharpe 最大化为目标选择策略子集。

**方法选择：**

| 策略池大小 | 方法 | 理由 |
|-----------|------|------|
| N ≤ 20 | **穷举搜索** | 2^20 ≈ 100 万种组合，几分钟内完成，保证全局最优 |
| 20 < N ≤ 50 | **双向贪心**（先加后删交替迭代） | 比单向贪心更接近全局最优 |
| N > 50 | **遗传算法/模拟退火** | 近似全局最优 |

**穷举搜索（推荐，N ≤ 20）：**

```python
from itertools import combinations

best_sharpe = -np.inf
best_subset = None

for size in range(3, len(pool) + 1):  # 至少 3 个策略
    for subset in combinations(pool, size):
        portfolio_sharpe = calc_portfolio_sharpe(subset)
        if portfolio_sharpe > best_sharpe:
            best_sharpe = portfolio_sharpe
            best_subset = subset
```

**双向贪心（N > 20 时的后备方案）：**

```python
# Phase 1: 前向贪心 — 逐个加入
pool = sort_by_sharpe_descending(all_strategies)
selected = [pool[0]]
for strategy in pool[1:]:
    candidate = selected + [strategy]
    if calc_portfolio_sharpe(candidate) > calc_portfolio_sharpe(selected):
        selected.append(strategy)

# Phase 2: 后向删除 — 逐个尝试移除
improved = True
while improved:
    improved = False
    for s in list(selected):
        without = [x for x in selected if x != s]
        if len(without) >= 3 and calc_portfolio_sharpe(without) > calc_portfolio_sharpe(selected):
            selected.remove(s)
            improved = True

# Phase 3: 重复 Phase 1-2 直到稳定（通常 2-3 轮）
```

**关键：负 Sharpe + 负相关性的策略会被自然纳入。** 比如一个 Sharpe=-0.3 但与组合相关性 -0.5 的策略，加入后可能让组合 MaxDD 从 -10% 降到 -6%，整体 Sharpe 反而提升。

**组合 Sharpe 计算时使用回撤重叠惩罚：**

```python
def calc_portfolio_sharpe(subset, penalty_weight=0.1):
    """组合 Sharpe，带回撤重叠惩罚。"""
    base_sharpe = weighted_sharpe(subset)  # 标准组合 Sharpe

    # 回撤重叠率：策略两两之间回撤期的 Jaccard 相似度
    overlap = mean_drawdown_overlap(subset)
    # overlap ∈ [0, 1]，0=完全不重叠，1=完全重叠

    return base_sharpe * (1 - penalty_weight * overlap)
```

回撤重叠比收益率相关性更重要 — 两个策略日收益相关性 0.3 但回撤在同一周发生，组合在那一周就会爆。

### Step 2: 协方差收缩 + Sharpe 加权 HRP

**Step 2a: 协方差矩阵收缩估计**

原始样本协方差在 N 接近 T 时噪音极大（17 个策略、几百个日收益观测）。必须做收缩：

```python
from sklearn.covariance import LedoitWolf

# Ledoit-Wolf 收缩估计（自动选择最优收缩系数）
lw = LedoitWolf().fit(returns_matrix)
cov_shrunk = pd.DataFrame(lw.covariance_, index=strategies, columns=strategies)
corr_shrunk = cov_to_corr(cov_shrunk)

# 用收缩后的协方差做 HRP
w_hrp = hrp_weights(cov_shrunk, corr_shrunk)
```

**Step 2b: Sharpe 加权**

```python
# HRP 基础权重（基于收缩协方差）
w_hrp = hrp_weights(cov_shrunk, corr_shrunk)

# Sharpe 调整因子
sharpe_factor = {v: max(0.1, sharpe) for v, sharpe in strategy_sharpes.items()}
# 对负 Sharpe 策略（因负相关被纳入）给予最低权重 0.1

# 混合权重 = HRP × Sharpe 调整，然后归一化
w_final = {v: w_hrp[v] * sharpe_factor[v] for v in strategies}
w_final = normalize(w_final)  # 归一化到 sum=1
```

**效果：** 高 Sharpe + 低相关的策略获得最多权重。低 Sharpe 的"对冲策略"获得适当但不过多的权重。

**角色化分析（参考，不强制约束权重）：**

可以在报告中将策略标注角色，辅助理解组合结构，但不作为硬性约束：

| 角色 | 特征 | 参考权重 |
|------|------|---------|
| **核心层** | Sharpe ≥ 中位数，与组合正相关 | 通常占 50-60% |
| **卫星层** | Sharpe > 0，与核心层低相关 | 通常占 25-35% |
| **对冲层** | 与组合负/零相关（含负 Sharpe） | 通常占 10-15% |

如果 Sharpe 加权 HRP 的结果明显偏离上述参考范围（如对冲层占 40%），可以作为复查信号，但不自动调整。

### Step 3: 权重上限裁剪

**单策略最大权重 20%。** 超出部分按比例重新分配给其他策略：

```python
MAX_WEIGHT = 0.20

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

**为什么 20%：** 实测中 v41 在 AG 上拿了 24%、v31 在 LC 上拿了 22%，单策略集中度过高。20% 上限保证至少 5 个策略共同承担风险，同时给优质策略足够空间。

### Step 4: Leave-one-out 边际检验

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

**这是最后的清理步骤。** 穷举/贪心构建时选入的策略，在其他策略也加入后可能变成冗余。Leave-one-out 检验能发现并清理这些冗余。

### Step 5: Bootstrap 稳健性验证

对最终组合做 Bootstrap 检验，量化结果的可信度：

```python
n_bootstrap = 1000
bootstrap_sharpes = []

for _ in range(n_bootstrap):
    # 有放回重抽样日收益
    sample = portfolio_daily_returns.sample(frac=1.0, replace=True)
    bootstrap_sharpes.append(calc_sharpe(sample))

ci_lower = np.percentile(bootstrap_sharpes, 2.5)
ci_upper = np.percentile(bootstrap_sharpes, 97.5)
print(f"Portfolio Sharpe: {portfolio_sharpe:.3f}  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

**通过标准：**
- 95% CI 下界 > 0（Sharpe 显著大于零）
- CI 宽度 < 1.5（结果足够稳定）
- 如果 CI 跨零，说明结果不可靠，需要简化组合或增加数据

**权重稳定性测试：**

```python
# 随机去掉 10% 数据，重新计算权重，重复 100 次
weight_samples = []
for _ in range(100):
    subset = portfolio_returns.sample(frac=0.9)
    w = full_pipeline(subset)  # 重新跑 Step 2-3
    weight_samples.append(w)

# 每个策略权重的变异系数 (CV)
for strategy in portfolio:
    weights_list = [ws[strategy] for ws in weight_samples]
    cv = np.std(weights_list) / np.mean(weights_list)
    print(f"  {strategy}: mean={np.mean(weights_list):.3f}  CV={cv:.2f}")
    # CV > 0.5 的策略权重不稳定，考虑降权或移除
```

### Step 6: Portfolio 质量要求

**Portfolio 评分必须达到 B+ (75分) 以上才可使用。**

**评分维度（4 维 12 指标）：**

| 维度 | 权重 | 包含指标 |
|------|:----:|---------|
| 收益风险比 | 40% | Sharpe, Calmar, MaxDD, DD Duration, **CVaR 95%** |
| 组合质量 | 25% | 平均相关性, **回撤重叠率**, Portfolio vs Best Single, 正 Sharpe 比例 |
| 实操性 | 20% | 策略数量, 最大单策略权重, 频率多样性, **Bootstrap CI 宽度** |
| 稳定性 | 15% | 收益一致性, 权益曲线稳定性, **Tail Ratio** |

**尾部风险指标（必须纳入评估）：**

| 指标 | 含义 | 最低要求 | 理想目标 |
|------|------|---------|---------|
| CVaR 95% | 最差 5% 情况平均损失 | > -3% (日) | > -1.5% (日) |
| Tail Ratio | 右尾95分位 / |左尾5分位| | > 0.8 | > 1.2 |
| 最大单日亏损 | 极端事件暴露 | > -5% | > -3% |
| Omega Ratio | 收益分布完整比较 | > 1.5 | > 2.0 |

**质量门槛汇总：**

| 指标 | 最低要求 | 理想目标 |
|------|---------|---------|
| 综合评分 | ≥ 75 (B+) | ≥ 85 (A) |
| Portfolio Sharpe / 最佳单策略 | ≥ 0.8 | ≥ 1.0 |
| Max Drawdown | < -15% | < -5% |
| 策略间平均相关性 | < 0.5 | < 0.3 |
| 回撤重叠率 | < 0.5 | < 0.3 |
| 最大单策略权重 | ≤ 20% | ≤ 10% |
| Bootstrap 95% CI 下界 | > 0 | > 0.5 |

如果评分低于 B+，必须在 `research_log/` 中分析原因并迭代：
- 组合 Sharpe 低于单策略？→ 检查 Sharpe 加权和角色分配是否合理
- 回撤过大？→ 检查回撤重叠率，补充对冲层策略
- 相关性过高？→ 扩大策略池（加入不同频率/指标类型）
- 某策略权重过高？→ 降低权重上限
- Bootstrap CI 跨零？→ 数据不足或过拟合，简化组合

### Step 7: Portfolio 止损标准

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

### Step 8: 重平衡（待定）

重平衡机制后续讨论确定。

### 活跃度过滤（所有 portfolio 必用）

**构建 portfolio 时不允许有僵尸策略。** 僵尸策略（测试期几乎不交易、收益接近 0）会导致：
1. 与其他策略的相关性为 NaN（方差为 0 无法计算 correlation）
2. 被错误地当作"对冲策略"选入 portfolio（因为 NaN 相关性被视为 0 相关）
3. 稀释资金利用率，占用权重但不产生收益

**所有策略类型都必须加活跃度过滤：**

```bash
# 所有 portfolio 构建都加 --min-activity
python portfolio/builder.py --symbol AG --min-activity 0.001  # strong_trend
python strategies/all_time/ag/build_portfolio.py --min-activity 0.001  # all_time
```

```python
# 过滤逻辑
active = [d for d in all_data
          if d["primary_sharpe"] is not None
          and abs(d["primary_return"] or 0) > min_activity]
```

**经验值：** 所有策略类型统一使用 `--min-activity 0.001`（abs(return) > 0.1%）。

**教训（AG v26 事件）：** v26 (TTM Squeeze + ADX + Force Index) 在 AG 测试期仅 4 笔交易、收益 -0.19%，日收益率几乎全为 0。被 portfolio builder 选为 "hedge"（因 NaN 相关性），占 9.3% 权重。加入活跃度过滤后被正确剔除，portfolio 从 7 策略变 6 策略，Sharpe 反而从 3.05 提升到 3.08。

### Portfolio 构建工具使用

**通用工具位置：** `portfolio/builder.py` + `portfolio/scorer.py`

**Strong Trend Portfolio（品种无关策略）：**
```bash
python portfolio/builder.py --symbol AG --start 2025-01-01 --end 2026-03-01
python portfolio/scorer.py strategies/strong_trend/portfolio/weights_ag.json
```

**All-Time AG Portfolio（品种专用策略）：**
```bash
python strategies/all_time/ag/build_portfolio.py
python portfolio/scorer.py strategies/all_time/ag/portfolio/weights_ag.json
```

`build_portfolio.py` 是一个薄包装器，将 `portfolio/builder.py` 的策略加载接口替换为 `all_time/ag/optimizer.py`，并设置正确的默认参数（测试期 2022-2026、活跃度过滤 0.1%）。

### 多频率策略混合

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

**策略容量约束：**
- 高频策略（5min/10min）在低流动性品种上有容量限制
- 分配资金不能超过策略容量上限（以滑点敏感性测试为准）
- 滑点敏感性：2x 滑点下 Sharpe 下降 >30% 的策略，权重上限额外收紧至 10%

**信号冲突处理：**
- 不同频率策略可能对同一品种产生相反信号（如 daily 做多，5min 做空）
- 在 `all_time/` 中这是允许的（多空均可），但需要注意净头寸
- 在 `strong_trend/` 中不会冲突（都是只做多）

### All-Time AG 策略架构

**100 个策略，5 类，多空全天候：**

| 类别 | 版本 | 数量 | 方向 | 特点 |
|------|------|:---:|:---:|------|
| 趋势跟踪 | v1-v20 | 20 | 多空 | Supertrend/ADX/Aroon/EMA/Ichimoku 等 |
| 均值回归 | v21-v40 | 20 | 多空 | RSI/Stochastic/BB/Z-Score/CCI 等 |
| 突破 | v41-v60 | 20 | 多空 | Donchian/TTM Squeeze/Keltner/NR7 等 |
| 多周期+量价 | v61-v80 | 20 | 多空 | 4h+1h 多周期, OBV/MFI/Klinger 等 |
| 自适应/混合 | v81-v100 | 20 | 多空 | Regime 切换/多指标投票/双策略模式 |

**与 strong_trend 的区别：**

| 维度 | strong_trend | all_time/ag |
|------|-------------|-------------|
| 方向 | 只做多 | **多空均可** |
| 品种 | 品种无关（跨品种训练） | **AG 专用**（单品种训练） |
| 行情覆盖 | 只做强趋势行情 | **全天候（趋势+震荡+突破）** |
| 训练数据 | J/ZC/JM/I/NI/SA 多品种 | AG 2012-2021 全历史 |
| 测试数据 | AG/EC 强趋势时段 | AG 2022-2026（含各种行情） |
| 策略数量 | 50 | 100 |
| optimizer | 手动定义参数空间 | **自动检测参数空间** |

**优化器特性（`optimizer.py`）：**
- 自动从策略类的 type annotations 检测参数及范围
- 特殊参数（atr_trail_mult、rsi_period 等）有预定义范围
- ValueError 安全捕获（多空冲突时返回 -1.0 惩罚）
- 支持批量优化：`--strategy v1-v20 --trials 150`

---

## 附录

### A. Portfolio 评分体系

每个 Portfolio 用 **0-100 综合评分**，覆盖 4 个维度。适用于单品种 all_time 和多品种组合。

**评分维度与权重：**

| 维度 | 权重 | 包含指标 | 关注点 |
|------|:----:|---------|--------|
| **收益风险比** | 40% | Sharpe, Calmar, MaxDD, 回撤持续时间 | 赚多少、亏多少、亏多久 |
| **组合质量** | 25% | 平均相关性, Portfolio/最佳单策略比, 正Sharpe占比 | 分散有效吗、组合有增值吗 |
| **实操性** | 20% | 策略数量, 最大单策略权重, 频率多样性 | 能不能实际跑起来 |
| **稳定性** | 15% | 收益一致性, 权益曲线稳定度 | 是不是靠运气 |

**各指标评分标准 (0-10)：**

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

**等级映射：**

| 分数 | 等级 | 含义 |
|:----:|:----:|------|
| 90+ | A+ | 卓越 — 可直接进入模拟盘 |
| 85-89 | A | 优秀 — 小幅优化后可上线 |
| 80-84 | A- | 良好 — 需要补充 walk-forward 验证 |
| 75-79 | B+ | 中上 — 核心指标过关，部分维度待提升 |
| 70-74 | B | 中等 — 可用但有明显弱点 |
| 60-69 | C | 勉强 — 需要重大改进 |
| <60 | D/F | 不合格 — 重新构建 |

**当前 Portfolio 评分：**

| Portfolio | Sharpe | MaxDD | 策略数 | 相关性 | 主要改进 |
|-----------|:------:|:-----:|:-----:|:-----:|---------|
| **AG Strong Trend** | **3.08** | **-3.52%** | 6 | 0.118 | Sharpe 2.43→3.08, MaxDD -5.92%→-3.52%, 相关性 0.45→0.12 |
| **LC Strong Trend** | **3.23** | **-0.75%** | 4 | 0.022 | Sharpe 2.91→3.23, MaxDD -3.07%→-0.75%, Portfolio/Best 1.29x |

**运行评分：**

```bash
python portfolio/scorer.py
```

### B. 防过拟合规范

1. **品种级分割** — 训练品种和测试品种完全分开。优化时只碰训练品种，测试品种从不参与参数选择
2. **多品种平均** — Optuna 目标函数 = mean(Sharpe across 6+ 训练品种)，不是单品种最优
3. **参数数量限制** — 每策略可调参数 ≤ 5 个。参数范围窄（2-3x），不给优化器太大空间
4. **指标参数** — 优先用指标默认参数。Optuna 调的是策略参数（阈值、周期），不是指标内部参数
5. **迭代验证** — `validate_and_iterate.py` 自动 3 轮迭代：验证 → 重优化表现差的 → 再验证
6. **Walk-forward** — 对关键策略做滚动验证（5年训练→1年测试），确认参数稳定性
7. **一致性奖励** — 重优化时目标函数加入 `min_sharpe * 0.3` bonus，奖励所有品种都盈利的参数

### C. 版本管理

**策略版本：**
- **v1 → v2**: 核心逻辑变更（换指标、改信号方向、改仓位模型）
- **同版本调参**: 参数微调不升版本，在 research_log 中记录
- **新策略**: 不同逻辑思路 = 新文件（如 `trend_v1.py` 和 `breakout_v1.py`）

**研究记录（必须）：**

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

**Git Commit 规范：**

```
[品种] 策略类型: 简短描述

示例:
[AG] trend_v1: initial strategy with ADX+ROC+ATR
[AG] trend_v1: tune ADX threshold from 25 to 20
[AG] portfolio: add risk parity weights v1
[indicator] add parabolic SAR to trend category
```

### D. 风险管理框架

**单笔风险：**

每笔交易最大风险 = 账户权益的 **2%**。这是仓位计算的基础：

```python
# 标准仓位公式
lots = (equity * 0.02) / (stop_distance * contract_multiplier)

# 示例：AG 白银，权益 100万，ATR=50，止损 3.5×ATR
# lots = 1,000,000 * 0.02 / (50 * 3.5 * 15) = 7.6 → 7 手
```

**总敞口限制：**

| 限制 | 数值 | 说明 |
|------|------|------|
| 单策略最大持仓 | 权益的 **30%** 保证金占用 | 含所有加仓层 |
| 单品种最大持仓 | 权益的 **40%** 保证金占用 | 同品种多策略合计 |
| 总账户最大持仓 | 权益的 **80%** 保证金占用 | 留 20% 现金缓冲 |
| 单笔最大亏损 | 权益的 **2%** | 通过仓位公式控制 |
| 单日最大亏损 | 权益的 **5%** | 触发后当日停止交易 |

**合约参数：动态获取，不要硬编码**

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

### E. 中国期货特殊规则

**夜盘归属：**

中国期货夜盘（21:00-次日02:30）归属**下一个交易日**。AlphaForge 的 `context.trading_day` 已正确处理此逻辑。

注意事项：
- 周五夜盘 → 属于下周一交易日
- 节假日前最后一个夜盘 → 属于节后第一个交易日
- 判断日内/隔夜时用 `trading_day` 而非 `datetime`

**涨跌停：**
- 触及涨跌停时 AlphaForge 拒绝成交
- 不同品种涨跌停幅度不同（一般 4%-8%，通过 `spec.price_limit` 获取）
- 策略不应在涨跌停附近下单（成交概率低）

**主连数据与换仓移仓：**

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

**数据可用范围：**

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

### F. 前视偏差（Look-ahead Bias）检查

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

### G. 策略生命周期

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

### H. 流动性与策略容量

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

### I. 实战经验参考

详见 `research_log/lessons_learned.md`（独立文档，记录已验证的实战经验和教训）。

### J. 注意事项速查

- **所有策略必须使用预计算模式**（on_init_arrays + context.bar_index 查表）
- 指标是纯函数（numpy in → numpy out），在 on_init_arrays 中传入完整数组一次计算
- 预计算与 Optuna 兼容：每次 trial 参数变化时，on_init_arrays 自动重新调用
- 所有价格计算用复权价（context.get_full_close_array），下单用原始价（context.close_raw）
- 信号在下一个 bar 的 open 执行，不是当前 bar
- 同品种不可同时持有多空仓位
- 保证金不足会被拒绝开仓，权益低于维持保证金会被强平
- FIFO 平仓：先平昨仓（便宜），再平今仓（贵）
- 单笔成交量不超过该 bar 成交量的 10%

### K. 策略质量门槛

策略必须通过以下标准才能进入 portfolio：

| 指标 | 最低要求 |
|------|---------|
| 样本外 Sharpe | > 0.3 |
| 最大回撤 | < -35% |
| 样本外测试期 | ≥ 2 年 |
| 交易次数 | ≥ 30 笔（统计显著性） |
| 训练集/测试集 Sharpe 比 | 测试集 ≥ 训练集的 50% |

不满足的策略可以保留在品种目录供参考，但不进 portfolio。
