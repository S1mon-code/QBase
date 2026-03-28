# Step 3: 单策略优化（Single Strategy Optimization）

> **所有优化在训练集上完成，测试集不参与任何优化决策。**

核心代码：[`strategies/optimizer_core.py`](../../strategies/optimizer_core.py)

---

## 优化架构

所有优化逻辑统一由 `optimizer_core.py` 提供，各优化器调用它：

```
optimizer_core.py (共享基础设施, 25 函数, ~1200 行)
├── 参数自动发现    (auto_discover_params)
├── 复合目标函数    (composite_objective)
├── 两阶段优化      (optimize_two_phase: 粗调→探针验证→精调→稳健性检查)
├── 多种子验证      (optimize_multi_seed: 3种子→取中位数)
├── 稳健性检查      (check_robustness: 邻域采样→高原/尖峰判断)
├── 统一回测接口    (run_single_backtest: 频率映射+bar重采样)
├── 统一结果构建    (build_result_entry: 统一 JSON schema)
├── 策略状态检测    (detect_strategy_status: active/dead/error/import_error)
├── 死策略跳过      (is_strategy_dead: 重跑时跳过已确认死亡的策略)
└── 边界保护        (narrow_param_space: 参数靠近边界时自动扩展范围)

strategies/strong_trend/optimizer.py     → 调用 optimizer_core（多品种评估）
strategies/all_time/ag/optimizer.py      → 调用 optimizer_core（单品种评估）
strategies/all_time/i/optimizer.py       → 调用 optimizer_core（单品种评估）
strategies/medium_trend/optimizer.py     → 调用 optimizer_core（多品种评估）
strategies/all_time/ag/boss_optimizer.py → Boss 优化器（scoring_mode 已修复）
```

---

## 可调参数分类与优先级

| 优先级 | 参数类别 | 示例 | 说明 |
|:---:|------|------|------|
| **必须优化** | 信号阈值 | ADX > 25、RSI < 30 | 对策略表现最敏感 |
| **必须优化** | 止损参数 | ATR 止损倍数 (2.0-5.0) | 直接影响风险收益比 |
| **可以优化** | 指标 period | RSI period (10-20) | 范围要窄，不给太大空间 |
| **尽量不动** | 加仓减仓结构 | 金字塔 1.0/0.5/0.25 | 固定结构，避免过拟合 |
| **尽量不动** | 分层止盈倍数 | 3ATR/5ATR | 通用规则，不需要按策略调 |

**总可调参数 <= 5 个，范围窄（2-3x）。** 参数越少、范围越窄，过拟合风险越低。

---

## 参数自动发现

**不再需要手动定义参数搜索空间。** `optimizer_core.auto_discover_params()` 从策略类的 type annotations + 默认值自动推导：

```python
class StrongTrendV1(TimeSeriesStrategy):
    st_period: int = 10          # 自动推导: range [4, 30]
    st_mult: float = 3.0         # 已知参数: range [1.5, 5.0]
    roc_period: int = 20         # 自动推导: range [8, 60]
    roc_threshold: float = 5.0   # 自动推导: range [1.5, 15.0]
    atr_trail_mult: float = 3.0  # 已知参数: range [2.0, 5.5]
```

### 已知参数映射（精确范围）

| 参数名 | 固定范围 |
|--------|---------|
| `atr_trail_mult` / `atr_stop_mult` | 2.0 - 5.5 |
| `st_mult` | 1.5 - 5.0 |
| `kc_mult` | 1.0 - 3.0 |
| `chand_mult` | 2.0 - 5.0 |
| `psar_af_step` | 0.01 - 0.04 |
| `psar_af_max` | 0.1 - 0.3 |
| `t3_vfactor` | 0.5 - 0.9 |
| `alma_offset` | 0.7 - 0.95 |

### 通用规则（按参数名模式）

- **Period 类**（含 period/lookback/fast/slow/signal/tenkan 等）：`default x 0.4 ~ default x 3.0`
- **Threshold/Mult 类**：`default x 0.3 ~ default x 3.0`
- **其他**：`default x 0.4 ~ default x 3.0`（int），`default x 0.3 ~ default x 3.0`（float）

**跳过的属性：** `name`, `warmup`, `freq`, `contract_multiplier`

---

## 目标函数（composite_objective）

**加权多维评分系统（0-10 分制）。** 每个维度独立评分 0-10，加权求和。Sharpe 是主指标（60%），但风险、质量、稳定性共同决定最终得分，防止高 Sharpe 高回撤的过拟合参数胜出。

```python
final_score = 0.60 * S_sharpe + 0.15 * S_risk + 0.15 * S_quality + 0.10 * S_stability
```

**设计原理：** 旧的加法惩罚（`sharpe - penalty`）无法解决 Sharpe 碾压问题——Sharpe 范围 0-3，惩罚项只有 0-0.4，高 Sharpe 永远赢。新系统将所有维度归一化到 0-10，通过权重控制各维度影响力。

### 维度 1: S_sharpe（权重 60%）— 收益能力

**双模式评分：** 粗调阶段用 tanh（防过拟合），精调阶段用 linear（追求绝对 Sharpe）。

```python
# 粗调 (coarse phase): tanh — 递减回报，防止追求极端 Sharpe
S_sharpe = 10 * tanh(0.7 * sharpe)

# 精调 (fine phase): linear — 等额奖励每个 Sharpe 增量
S_sharpe = min(10, sharpe * 10 / 3)
```

| Sharpe | tanh (粗调) | linear (精调) | 说明 |
|:---:|:---:|:---:|------|
| 0.0 | 0.0 | 0.0 | 无收益 |
| 0.5 | 3.4 | 1.7 | 初级 |
| 1.0 | 6.0 | 3.3 | 中等偏上 |
| 1.5 | 7.8 | 5.0 | 优秀 |
| 2.0 | 8.9 | 6.7 | 卓越 |
| 2.5 | 9.4 | 8.3 | 精调在此开始给更高分 |
| 3.0 | 9.7 | 10.0 | 精调满分 |

**设计原理：**
- **粗调用 tanh** — Sharpe 2.0->3.0 只给 0.85 分激励，防止优化器为追求高 Sharpe 牺牲稳健性
- **精调用 linear** — 同样 Sharpe 2.0->3.0 给 3.33 分激励（4 倍），在已验证稳健的参数邻域内追求绝对最优
- 粗调找到"对的方向"，精调在那个方向上走到最远

**多品种一致性处理：** 多品种评估时，如果最差品种 Sharpe 为负，effective_sharpe 会被缩减（乘以一致性系数 0.5-1.0），然后再评分。

### 维度 2: S_risk（权重 15%）— 风险控制

基于 MaxDD 线性评分：

| MaxDD | 评分 | 说明 |
|:---:|:---:|------|
| <= 5% | 10.0 | 极低风险 |
| 10% | 8.6 | 低风险 |
| 15% | 7.1 | 可接受 |
| 20% | 5.7 | 中等风险 |
| 25% | 4.3 | 偏高 |
| 30% | 2.9 | 高风险 |
| >= 40% | 0.0 | 不可接受 |

### 维度 3: S_quality（权重 15%）— 利润质量

基于利润集中度（top 10% 天数贡献的利润比例）：

| 集中度 | 评分 | 说明 |
|:---:|:---:|------|
| <= 0.30 | 10.0 | 利润分布健康 |
| 0.50 | 6.9 | 可接受 |
| 0.70 | 3.9 | 偏集中 |
| 0.80 | 2.3 | 脆弱 |
| >= 0.95 | 0.0 | 完全依赖少数天 |

**优雅降级：** 无权益曲线数据时给 5 分中性值，不影响评分。

### 维度 4: S_stability（权重 10%）— 稳定性

基于月度胜率（每 21 个交易日统计一次）：

| 月度胜率 | 评分 | 说明 |
|:---:|:---:|------|
| >= 65% | 10.0 | 非常稳定 |
| 55% | 7.1 | 良好 |
| 45% | 4.3 | 偏弱 |
| <= 30% | 0.0 | 大起大落 |

**优雅降级：** 无权益曲线数据时给 5 分中性值。

### 硬过滤: 交易次数门槛

低于门槛的回测结果直接排除（返回 -10.0），不参与评分。

门槛设计原则：**降低到"能反映策略是否有效"的最低水平**，而非追求统计学完美。过高的门槛会杀掉低频趋势策略（daily 策略在 3 个月训练段上可能只有 3-5 笔交易，旧门槛 30 会全部过滤掉）。

| 频率 | 最低交易次数 | 说明 |
|------|:---:|------|
| daily | 10 | 趋势策略月均 1-2 笔，10 笔约半年 |
| 4h | 20 | 每周 2-3 笔，20 笔约 2 个月 |
| 1h / 60min | 30 | CLT 最低门槛 |
| 30min | 50 | |
| 15min / 10min / 5min | 80 | 高频交易多，80 笔足够 |

### 评分效果对比

| 策略类型 | Sharpe | MaxDD | 集中度 | 月胜率 | **得分** |
|---------|:---:|:---:|:---:|:---:|:---:|
| **均衡型**（理想） | 1.2 | -12% | 0.35 | 55% | **7.41** |
| 过拟合型 | 1.9 | -30% | 0.40 | 45% | 7.34 |
| 稳健型 | 1.0 | -8% | 0.40 | 58% | 7.07 |
| 保守型 | 0.8 | -5% | 0.30 | 60% | 6.91 |
| **高风险型**（最差） | 2.5 | -45% | 0.60 | 40% | **6.74** |

**关键效果：** 均衡型（Sharpe=1.2, DD=-12%）以 7.41 胜过过拟合型（Sharpe=1.9, DD=-30%）的 7.34。高风险型虽然 Sharpe=2.5 但总分最低。这确保优化器不会为追求高 Sharpe 而忽视风险和稳定性。

---

## 两阶段优化（optimize_two_phase）

默认优化流程，自动执行粗调->探针验证->精调->稳健性检查四步：

```
Phase 1: 粗调 (coarse)     Probe 验证            Phase 2: 精调 (fine)         Phase 3: 稳健性
───────────────────────   ──────────────────     ──────────────────────       ──────────────────
全范围 TPE 搜索            粗调最优参数验证        围绕粗调最优 ±15% 范围       最优参数 ±15% 邻域
30 trials (含 5 probe)     检测是否在死区          50 trials                    max(20, n_params*5) 采样
  |                          |                      |                           |
  ├─ probe 全失败 → 早停     ├─ dead → 跳过精调     ├─ 取 coarse vs fine 较优    ├─ >=60% 邻居 > 最优50%
  └─ Sharpe <= -5 → 验证    └─ active → 继续       └─ 步长减半，精确寻优          → PLATEAU (稳健)
                                                                                 → SPIKE (噪音)
```

### Probe 验证（精调前置检查）

粗调完成后、进入精调前，先对粗调最优参数做一次 `detect_strategy_status()` 检查。如果粗调最优参数处于"死区"（strategy status = dead），则跳过精调阶段，避免浪费算力。

**解决的问题：** I All-Time 策略池中大量策略粗调最优 Sharpe 极低（如 -3.0），精调在这些参数附近搜索毫无意义。Probe 验证在精调前拦截这些情况。

### Strong Trend 用法

```bash
# 默认两阶段优化（推荐，80 trials = 27 coarse + 27 fine + robustness）
python strategies/strong_trend/optimizer.py --strategy v1 --trials 80

# 批量优化
python strategies/strong_trend/optimizer.py --strategy all --trials 80

# 最终 top 候选启用多种子验证（3x 算力，更可靠）
python strategies/strong_trend/optimizer.py --strategy v12 --trials 80 --multi-seed
```

### All-Time AG 用法

```bash
# 粗调
python strategies/all_time/ag/optimizer.py --strategy v1 --trials 50 --phase coarse

# 精调（自动读取粗调结果）
python strategies/all_time/ag/optimizer.py --strategy v1 --trials 100 --phase fine

# 批量
python strategies/all_time/ag/optimizer.py --strategy v1-v50 --trials 50 --phase coarse

# 多种子
python strategies/all_time/ag/optimizer.py --strategy v1 --trials 50 --multi-seed
```

---

## 稳健性检查（check_robustness）

**每次优化自动执行。** 找到最优参数后，在 ±15% 邻域随机采样：

- **采样数量：** `max(20, n_params * 5)` — 3 个参数采 20 个，7 个参数采 35 个
- **判断标准：** >=60% 邻居得分 > 最优的 50% -> PLATEAU（稳健），否则 -> SPIKE（噪音）
- **输出标记：** 结果 JSON 中 `robustness.is_robust` 字段

如果判定为 PLATEAU，可能使用邻域中发现的更优参数（高原中心可能比 Optuna 的最优更好）。

---

## 多种子验证（optimize_multi_seed）

**默认关闭，用 `--multi-seed` 手动开启。** 适用于最终 portfolio 入选的 top 10-15 个策略。

- 3 个种子 (42, 123, 456) 各自独立跑完整两阶段
- 最终取**中位数**结果（最稳定，避免选到运气好/差的种子）
- 跨种子一致性检查：`std < 50% * mean` -> CONSISTENT，否则 -> INCONSISTENT

**3x 算力成本**，不建议对全部策略开启。

---

## 边界保护（narrow_param_space）

精调阶段围绕粗调最优参数 ±15% 缩小搜索范围。但如果最优参数靠近原始搜索空间的边界，±15% 会被截断，导致精调空间不对称或过窄。

**边界保护机制：** `narrow_param_space` 检测参数是否在原始范围边界附近（距边界 < 总范围的 10%），如果是，自动向边界外扩展搜索范围，确保精调有足够的探索空间。

```
原始范围: [2.0, 5.5]
粗调最优: 5.3 (靠近上边界)
无保护:   [4.5, 5.5]  ← 被截断，只向下搜索
有保护:   [4.5, 5.8]  ← 向上扩展，双向搜索
```

---

## 新增核心函数

### build_result_entry() — 统一 JSON Schema 构建器

所有 5 个优化器现在通过 `build_result_entry()` 生成统一格式的结果 JSON，确保输出字段一致：

```python
entry = build_result_entry(
    version="v12",
    best_params={...},
    best_sharpe=2.45,
    best_score=7.83,
    status="active",         # active/dead/error/import_error
    n_trials=80,
    phase="two_phase",
    robustness={...},
)
```

### detect_strategy_status() — 策略状态自动分类

根据优化结果自动判定策略状态：

| 状态 | 条件 | 含义 |
|------|------|------|
| `active` | Sharpe > 阈值且有有效交易 | 正常策略，可进入精调/portfolio |
| `dead` | Sharpe 极低或交易次数不足 | 策略在该品种/频率上无效 |
| `error` | 回测过程中抛出异常 | 代码错误，需修复 |
| `import_error` | 策略类无法导入 | 文件/依赖问题 |

### is_strategy_dead() — 重跑时跳过死策略

读取已有优化结果文件，如果策略的 status 为 `dead`，在批量重跑时自动跳过，节省算力。适用于增量优化场景（新增策略后只跑新策略 + 非 dead 策略）。

---

## 早停与淘汰标准

### Probe 早停机制（自动）

- 每次优化先跑 5 个 probe trials
- 如果 5 个 trials 全部返回 -999（策略代码报错），立即跳过
- 多种子模式下，第一个种子 probe 失败则跳过剩余种子

### Sharpe 负值不跳过

Sharpe <= -5.0 跳过精调（节省时间），但保留在策略池中。portfolio 构建时负 Sharpe + 负相关的策略可能比正 Sharpe + 高相关的更有价值。

### 交易次数门槛

内置在目标函数中自动执行（见上方交易次数门槛表格）。

---

## 参数交互处理

**参数之间可能耦合：** 比如 RSI period 和 RSI 阈值——period 短了，波动大，阈值要调松。

- Optuna TPE 能捕捉部分交互，但参数 > 5 个时交互空间爆炸
- 如果发现两个参数强耦合，考虑固定其中一个或合并为比值参数

---

## 频率也是优化维度

同一策略在不同频率上表现可能差很多。**在 fine-tune 参数前先选频率：**

```bash
# 用默认参数在多频率上快速跑一遍
for freq in 5min 30min 1h 4h daily; do
    ./run.sh strategy.py --symbols AG --freq $freq --start 2013 --end 2021
done

# 选 Sharpe 最高的频率，再在该频率上 fine-tune 参数
```

---

## 优化效率注意事项

### 速度瓶颈分析

- **数据加载：** 每个策略 ~3-5 秒
- **ML 指标计算：** 每个 trial 的 `on_init_arrays` 都要重算（参数变了），HMM/K-Means 等较慢
- **回测本身：** daily ~4s/trial，4h ~4.5s/trial，1h ~7s/trial

### 并行建议

- 不同策略之间完全独立，可以用 subagent 并行优化
- 同一策略内 Optuna TPE 是顺序依赖的（n_jobs=1）
- 批量优化：`optimizer.py --strategy all --trials 80`

---

## 优化器位置与配置

```
strategies/optimizer_core.py             # 共享优化基础设施（25 函数, ~1200 行）
strategies/strong_trend/optimizer.py     # 强趋势优化器（多品种评估）
strategies/all_time/ag/optimizer.py      # AG 全时间优化器（单品种评估）
strategies/all_time/ag/boss_optimizer.py # AG Boss 优化器（scoring_mode bug 已修复）
strategies/all_time/i/optimizer.py       # I 全时间优化器（单品种评估）
strategies/medium_trend/optimizer.py     # 中趋势优化器（全频率优化中）
```

### 优化结果保存

- `optimization_results.json` — 每个策略的最优参数、Sharpe、Score、状态、稳健性标记
- 粗调结果保存在 `optimization_coarse.json`（All-Time AG/I）

### 统一输出字段（所有 5 个优化器）

所有优化器通过 `build_result_entry()` 输出统一 JSON schema：

- `version`, `best_sharpe`, `best_score`, `best_params`, `n_trials` — 基本信息
- `status` — 策略状态（`active` / `dead` / `error` / `import_error`）
- `robustness` — `{is_robust, neighbor_mean, neighbor_std, above_threshold_pct}`
- `is_consistent` — 多种子一致性（仅 `--multi-seed` 时）
- `phase` — 优化阶段（`two_phase` / `coarse_only` / `probe_failed`）

**注意：** `best_sharpe` 和 `best_score` 都会输出。`best_sharpe` 是纯 Sharpe Ratio，`best_score` 是复合目标函数得分（含风险、质量、稳定性权重）。Portfolio 构建时通常参考 `best_score`。

### 当前优化成果

| 策略池 | 策略数 | 精调数 | Sharpe 范围 | Score 范围 | 说明 |
|--------|:-----:|:-----:|:-----------:|:----------:|------|
| AG All-Time | 100 | Top 50 | 2.22 - 7.43 | — | 精调完成 |
| Medium Trend | 200 | Top 45 | — | 1.93 - 9.26 | 43/45 robust |
| I All-Time | 200 | 进行中 | — | — | probe 验证已修复 |

---

## BacktestConfig Industrial-Grade 设置

V6.0 扩展了 BacktestConfig 至约 30 个参数。**各频率应在不同阶段使用不同模式，详见上方"回测模式选择"章节。**

```python
# Industrial 模式完整配置
config = BacktestConfig(
    initial_capital=10_000_000,
    volume_adaptive_spread=True,
    dynamic_margin=True,
    time_varying_spread=True,
    rollover_window_bars=20,
    asymmetric_impact=True,
    detect_locked_limit=True,
    margin_check_mode="daily",
    margin_call_grace_bars=3,
)
```

**关键规则：** 1h+ 策略精调必须用 Industrial 模式，Daily 策略可全程 Basic 但最终须 Industrial 验证一次。详见 [AlphaForge API 参考](../reference/alphaforge-api.md#两种回测模式)。

---

## 回测模式选择（Basic vs Industrial）

V6 引擎提供两种回测模式，对优化结果影响显著。**精调阶段所有频率都必须使用 Industrial 模式。**

### 实测 Sharpe 衰减数据

| 策略 | 频率 | Basic Sharpe | Industrial Sharpe | 衰减 | 成交量 Basic→Industrial |
|------|------|:-----------:|:-----------------:|:----:|:-----------------------:|
| v12 | daily | 3.09 | 2.84 | -8.1% | 18→18 |
| v9 | 1h | 2.15 | 1.00 | -53.7% | 261→43 |

**关键发现：** Daily 策略几乎不受影响（-8%），但 1h 策略损失超过一半 Sharpe（-54%），因为 V6 Industrial 模式过滤了不真实的成交（volume-adaptive spread、locked-limit detection 等）。

### 各阶段模式选择规则

| 频率 | 粗调（Coarse） | 精调（Fine） | 最终验证 |
|------|:-------------:|:-----------:|:--------:|
| daily | Basic（快速） | **Industrial（必须）** | **Industrial** |
| 4h | Basic（快速） | **Industrial（必须）** | **Industrial** |
| 1h | Basic（快速） | **Industrial（必须）** | **Industrial** |
| 5min | Basic（快速） | **Industrial（必须）** | **Industrial** |

### 优化流程（含 Industrial 验证）

```
粗调 (Basic, 快速筛选)
  → Probe 验证（跳过死区）
    → 精调 (所有频率均用 Industrial 模式)
      → 稳健性检查
        → 通过 → 进入 Step 4
        → 失败（Industrial Sharpe < 0）→ 策略不可靠，不进入 Portfolio
```

### Industrial 验证判定标准

| Industrial 衰减 | 判定 | 动作 |
|:---------------:|------|------|
| < 10% | 正常（daily 典型值） | 直接通过 |
| 10-30% | 可接受（4h 典型值） | 通过，记录衰减比 |
| 30-50% | 警告 | 需要在 Industrial 模式下重新精调 |
| > 50% | 不可靠 | 策略 alpha 来自不真实成交假设，不入 Portfolio |

### 注意事项

- **粗调阶段所有频率都可以用 Basic**（速度优先，只要方向对就行）
- **精调阶段所有频率都必须用 Industrial**（确保参数在真实成本下仍然有效）
- Daily 策略在 Industrial 下衰减通常 5-10%，属正常
- 1h+ 策略在 Basic 下的精调结果完全不可靠（v9 案例：261 笔→43 笔，Sharpe -54%）

---

## 非标准频率支持

优化器自动处理 4h/1h 等非原生频率：

- `4h`: 加载 60min 数据，每 4 根合成 1 根 4h bar（或 V6 `resample_freqs`）
- `1h`: V6.0 原生支持（不再需要映射到 60min）
- `20min`: 加载 10min，每 2 根合成
