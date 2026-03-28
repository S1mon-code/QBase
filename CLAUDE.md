# QBase — Agent 开发指南

QBase 是量化策略研究工作区。320 个指标 + 350+ 策略，回测引擎由 AlphaForge V6.0 提供。1505 个单元测试覆盖全链路。

## 项目结构

```
QBase/
├── indicators/          # 320个指标（10大类，纯numpy函数）
├── attribution/         # 归因分析（信号ablation + 行情regime + batch/coverage/drawdown/decay）
├── portfolio/           # Portfolio 构建（穷举/贪心 + HRP + Bootstrap + 稳定性测试）
├── strategies/
│   ├── optimizer_core.py    # 共享优化基础设施（25函数, ~1200行）
│   ├── walk_forward.py      # Walk-Forward 滚动验证工具
│   ├── template_simple.py   # 简单策略模板（90%场景）
│   ├── template_full.py     # 完整策略模板（复杂场景）
│   ├── template_mean_reversion.py  # 均值回归模板（RSI + 固定止损 + 时间止损）
│   ├── template_volatility_target.py # 波动率目标模板（vol-scaling + 百分比止损）
│   ├── template_time_based.py      # 纯时间退出模板（时间止损 + 紧急止损）
│   ├── strong_trend/        # 50个强趋势策略 (v1-v50)
│   ├── all_time/ag/         # AG 100个全时间策略（含 analyze_failures.py）
│   ├── all_time/i/          # I 200个全时间策略
│   ├── medium_trend/        # 200个中趋势策略（V4迁移完成）
│   └── mean_reversion/      # 待开发
├── trend/               # 行情时段数据（RALLIES.md, MEDIUM_TRENDS.md）
├── screener/            # 品种筛选器
├── research_log/        # 实验记录 + 归因报告
├── reports/             # HTML 回测报告
├── tests/               # 单元测试（1505 tests）
│   └── robustness/      # 鲁棒性测试（滑点敏感性 + Monte Carlo 压力测试）
├── docs/
│   ├── pipeline/        # 流水线各阶段详细文档
│   └── reference/       # API、风控、期货规则等参考
├── config.yaml          # AlphaForge 路径配置
├── conftest.py          # sys.path 配置
└── run.sh               # 策略运行入口
```

## 开发流水线（5.5 步）

```
指标池 ──→ 策略开发 ──→ 单策略优化 ──→ 测试集验证 ──→ 归因分析 ──→ Portfolio 构建
```

| 步骤 | 输入 | 输出 | 详细文档 |
|------|------|------|---------|
| **Step 1: 指标池** | 研究/灵感 | 纯 numpy 函数 | [docs/pipeline/step1-indicators.md](docs/pipeline/step1-indicators.md) |
| **Step 2: 策略开发** | 1-3 个指标 | 完整策略代码 | [docs/pipeline/step2-strategy-dev.md](docs/pipeline/step2-strategy-dev.md) |
| **Step 3: 单策略优化** | 训练集数据 | 最优参数 | [docs/pipeline/step3-optimization.md](docs/pipeline/step3-optimization.md) |
| **Step 4: 测试集验证** | 锁定参数 | 验证报告 | [docs/pipeline/step4-validation.md](docs/pipeline/step4-validation.md) |
| **Step 4.5: 归因分析** | 验证结果 | 归因报告 | [docs/pipeline/step4.5-attribution.md](docs/pipeline/step4.5-attribution.md) |
| **Step 5: Portfolio** | 多策略曲线 | 权重分配 | [docs/pipeline/step5-portfolio.md](docs/pipeline/step5-portfolio.md) |

## 参考文档

| 文档 | 内容 |
|------|------|
| [AlphaForge API](docs/reference/alphaforge-api.md) | Context API、BacktestResult、性能优化、Report 生成 |
| [风险管理](docs/reference/risk-management.md) | 仓位公式、敞口限制、Portfolio 止损、策略退化标准 |
| [中国期货规则](docs/reference/china-futures.md) | 夜盘、涨跌停、换仓、数据范围、流动性分级 |
| [防过拟合](docs/reference/anti-overfit.md) | 数据分割、前视偏差、策略生命周期、质量门槛 |

## 全局规则（所有阶段适用）

### 回测模式（Basic vs Industrial）

QBase 回测有两种模式，适用场景不同：

**Basic 模式**（默认）— 开发和快速迭代用：
```python
config = BacktestConfig(initial_capital=10_000_000)
# 固定 1-tick 滑点，总是成交，瞬时换仓，仅开盘时检查保证金
```

**Industrial 模式**（V6 真实仿真）— 进 Portfolio 前必须验证：
```python
config = BacktestConfig(
    initial_capital=10_000_000,
    volume_adaptive_spread=True,     # 低量→更宽价差
    dynamic_margin=True,             # 交割月阶梯保证金
    time_varying_spread=True,        # 开盘/收盘时段更宽价差
    rollover_window_bars=20,         # 20 bar 渐进换仓
    asymmetric_impact=True,          # 顺势冲击低，逆势冲击高
    detect_locked_limit=True,        # 涨跌停 + 低量→拒绝全部订单
    margin_check_mode="daily",       # 每日结算价检查保证金
    margin_call_grace_bars=3,        # 追保宽限 3 bar
)
```

**实测 Sharpe 衰减（V6 真实数据）：**

| 策略 | 频率 | Basic Sharpe | Industrial Sharpe | 衰减 | 成交量变化 |
|------|------|:-----------:|:-----------------:|:----:|:----------:|
| v12 | daily | 3.09 | 2.84 | -8.1% | 18→18 |
| v9 | 1h | 2.15 | 1.00 | -53.7% | 261→43 |

**规则：**
- **任何策略进入 Portfolio 前，必须在 Industrial 模式下验证 Sharpe 仍为正**
- **高频策略（1h 及以上）必须在 Industrial 模式下优化，不能只用 Basic**
- Daily 策略可全程用 Basic 开发（衰减 < 10%），最终 Industrial 验证一次即可
- 1h+ 策略 Basic 模式下的优化结果不可靠，必须在 Industrial 模式下精调

### 预计算模式（必须）

**所有策略必须使用 `on_init_arrays` 预计算模式。** 指标在初始化时一次性全数组计算，`on_bar` 通过 `context.bar_index` 查表。

```python
def on_init_arrays(self, context, bars):
    closes = context.get_full_close_array()
    self._indicator = my_indicator(closes, period=14)

def on_bar(self, context):
    val = self._indicator[context.bar_index]  # O(1) 查表
```

### AlphaForge V6.0 必须遵守

1. **`context.is_rollover`** — 不要用 `context.current_bar.is_rollover`
2. **`ContractSpecManager` 模块级缓存** — 不要在 `_calc_lots` 中每次 import
3. **多周期用 `resample_freqs`** — 不要手动 reshape，用 `context.get_resampled_bars("4h")` + `context.get_resampled_index_map("4h")`
4. **`"1h"` 原生支持** — 不再需要用 `"60min"`，直接用 `"1h"`

### 核心原则

- **简单优先** — 1 个强指标 > 3 个弱指标（v46 仅 Supertrend 排名 16/50）
- **量价是 alpha 核心** — v12 归因：Volume Momentum 贡献 51.7%，Aroon+PPO 合计 < 3%
- **参数 ≤ 5 个** — 范围窄（2-3x），防过拟合
- **测试集只读** — 不能因测试结果修改参数
- **归因先行** — 进 Portfolio 前必须通过归因分析

### 策略命名

v1-v500 顺序命名，优先扩展新版本而非修改旧版本。

### Git Commit 规范

```
[品种/模块] 类型: 简短描述

示例:
[AG] trend_v12: initial strategy with Aroon+PPO+VolMom
[optimizer] dual-mode Sharpe scoring for fine phase
[attribution] v12 AG attribution report
[indicator] add parabolic SAR to trend category
```

### 运行命令速查

```bash
# 回测
./run.sh strategies/strong_trend/v12.py --symbols AG --freq daily --start 2022

# 优化
python strategies/strong_trend/optimizer.py --strategy v12 --trials 80

# 验证 + 归因
python strategies/strong_trend/validate_and_iterate.py

# 单策略归因
python -c "
from attribution.signal import run_signal_attribution
from attribution.report import generate_attribution_report
# ...
"

# 批量归因（Portfolio 全策略）
python -m attribution.batch --portfolio <weights_file>

# Regime 覆盖矩阵（RED FLAG 检测）
python -m attribution.coverage --portfolio <weights_file>

# 回撤归因
python -m attribution.drawdown --portfolio <weights_file>

# Alpha 衰减检测
python -m attribution.decay --strategy v12 --symbol AG

# Walk-Forward 验证
python strategies/walk_forward.py --strategy strong_trend/v12 --symbol AG

# 失败模式分析
python strategies/all_time/ag/analyze_failures.py

# 测试（1505 tests）
pytest tests/

# 滑点敏感性测试（LOW/MODERATE/HIGH 判定）
python tests/robustness/slippage_test.py --strategy <strategy> --symbol <symbol>

# Monte Carlo 压力测试（1000x bootstrap，ROBUST/ACCEPTABLE/FRAGILE 判定）
python tests/robustness/stress_test.py --strategy <strategy> --symbol <symbol>

# Portfolio 选择稳定性测试（CORE/SATELLITE/EDGE 分类）
python portfolio/stability_test.py --portfolio <weights_file>

# Portfolio 构建 + 稳定性测试
python portfolio/builder.py --symbol AG --stability-test 100
```

## 关键经验（详见 research_log/lessons_learned.md）

- **宽止损** (ATR > 4.0) 是强趋势中最重要的单一因素
- **量价/OI 是 alpha 的几乎全部来源** — v12 归因：VolMom 51.7%，其他 < 3%
- **多周期策略不推荐** — 增加复杂度但未增加 alpha
- **极简策略出人意料地强** — v46 仅 Supertrend 排名 16/50
- **daily 频率占据 Top 10 中 9 席**
- **一致性 > 绝对值** — 两品种都强 > 单品种极高
