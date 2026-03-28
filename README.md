# QBase

量化研究工作区 — 指标库 + 策略开发 + 归因分析，回测引擎由 [AlphaForge](../AlphaForge) 提供。

## 当前成果

- **320 个指标** — 10 大类（momentum / trend / volatility / volume / ml / regime / spread / structure / microstructure / seasonality）
- **350+ 策略** — 50 强趋势 + 100 AG全时间 + 200 I全时间
- **49/50 测试集正 Sharpe** — AG (+254%) 和 EC (+907%) 强趋势验证
- **归因分析 (Step 4.5)** — 信号 ablation + 行情 regime 归因，自动化集成到流水线
- **Portfolio C (通用)** — AG Sharpe 2.58, LC Sharpe 2.37
- **Top 3**: v12 (Aroon+PPO+VolMom, Sharpe 2.92), v11 (Vortex+ROC, 2.53), v34 (McGinley+PPO, 2.45)

### v12 归因发现

| 指标 | 贡献 | 占比 |
|------|:---:|:---:|
| Volume Momentum | +1.60 | **51.7%** |
| PPO Histogram | +0.09 | 2.8% |
| Aroon Oscillator | +0.003 | 0.1% |

**Volume Momentum 是唯一真正的 alpha 来源。** 强趋势下胜率 100%，弱趋势下 0%。

## 结构

```
indicators/          # 320个指标（10大类，纯numpy函数）
attribution/         # 归因分析模块 (Step 4.5)
├── signal.py        # 信号归因（ablation test）
├── regime.py        # 行情归因（regime 标注 + 统计）
└── report.py        # Markdown 报告生成
strategies/
├── strong_trend/    # 50个强趋势策略 (v1-v50)，只做多
│   ├── v1.py ~ v50.py
│   ├── optimizer.py         # Optuna 参数优化器
│   ├── validate_and_iterate.py  # 验证 + 归因分析
│   └── portfolio/           # Portfolio C 构建结果
├── all_time/        # 全时间策略（按品种）
│   ├── ag/          # 白银 100 策略
│   └── i/           # 铁矿石 200 策略
├── medium_trend/    # 中趋势策略（待开发）
├── weak_trend/      # 弱趋势策略（待开发）
└── mean_reversion/  # 震荡策略（待开发）
portfolio/           # 通用 Portfolio 构建工具
trend/               # 行情时段参考数据（19 强趋势 + 40 中趋势）
research_log/        # 实验记录 + 归因报告
tests/               # 指标 + 归因单元测试
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/

# 运行策略（通过 AlphaForge）
./run.sh strategies/strong_trend/v12.py --symbols AG --freq daily --start 2022

# Optuna 参数优化（单策略 150 trials）
python strategies/strong_trend/optimizer.py --strategy v12 --trials 150

# 全量优化（50策略）
python strategies/strong_trend/optimizer.py --strategy all --trials 150

# 测试集验证 + 归因分析（自动运行 Step 4 + Step 4.5）
python strategies/strong_trend/validate_and_iterate.py

# 单策略归因分析
python -c "
from attribution.signal import run_signal_attribution
from attribution.regime import run_regime_attribution
from attribution.report import generate_attribution_report
import importlib

cls = importlib.import_module('strategies.strong_trend.v12').StrongTrendV12
params = {'aroon_period': 20, 'ppo_fast': 19, 'ppo_slow': 29, 'vol_mom_period': 25, 'atr_trail_mult': 4.814}
sig = run_signal_attribution(cls, params, 'AG', '2025-01-01', '2026-03-01')
reg = run_regime_attribution(cls, params, 'AG', '2025-01-01', '2026-03-01')
generate_attribution_report(sig, reg, 'research_log/attribution/v12_AG.md')
"
```

## 指标使用

```python
import numpy as np
from indicators.momentum.roc import rate_of_change
from indicators.trend.adx import adx

closes = np.array([...])
roc_values = rate_of_change(closes, period=12)
```

## 策略架构

所有策略使用 **on_init_arrays 预计算模式**（~20x 加速）：

```python
class MyStrategy(TimeSeriesStrategy):
    def on_init_arrays(self, context, bars):
        closes = context.get_full_close_array()
        self._indicator = indicator_func(closes, ...)

    def on_bar(self, context):
        i = context.bar_index
        val = self._indicator[i]  # O(1) 查表
        # 交易逻辑...
```
