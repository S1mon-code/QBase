# QBase

量化研究工作区 — 指标库 + 策略开发，回测引擎由 [AlphaForge](../AlphaForge) 提供。

## 当前成果

- **100 个指标** — momentum (25) / trend (25) / volatility (25) / volume (25)
- **50 个强趋势策略** — 全部使用 on_init_arrays 预计算架构，Optuna 优化
- **49/50 测试集正 Sharpe** — AG (+254%) 和 EC (+907%) 强趋势验证
- **Top 3**: v12 (Aroon+PPO, Sharpe 2.92), v11 (Vortex+ROC, 2.53), v34 (McGinley+PPO, 2.45)

## 结构

```
indicators/          # 100个量价指标（momentum / volatility / trend / volume）
strategies/
├── strong_trend/    # 50个强趋势策略 (v1-v50)，只做多
│   ├── v1.py ~ v50.py
│   ├── optimizer.py         # Optuna 参数优化器
│   ├── validate_and_iterate.py
│   └── optimization_results.json
├── medium_trend/    # 中趋势策略（待开发）
├── weak_trend/      # 弱趋势策略（待开发）
├── mean_reversion/  # 震荡策略（待开发）
├── all_time/        # 全时间策略（待开发）
│   ├── ag/
│   └── i/
└── template.py      # 策略模板（预计算架构）
trend/               # 行情时段参考数据（19 强趋势 + 40 中趋势）
screener/            # 品种筛选器（待开发）
research_log/        # 实验记录 + 经验教训
tests/               # 指标单元测试
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

# 测试集验证 + 迭代
python strategies/strong_trend/validate_and_iterate.py
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
