# QBase

量化研究工作区 — 指标库 + 策略开发，回测引擎由 [AlphaForge](../AlphaForge) 提供。

## 结构

```
indicators/          # 量价指标库（momentum / volatility / trend / volume）
strategies/          # 按品种组织的策略（ag/, au/, I/ ...），命名: {类型}_v{版本}.py
screener/            # 品种筛选器（按趋势/震荡/波动率等维度排序）
fundamental/         # 基本面量化（预留）
research_log/        # 实验记录
tests/               # 指标单元测试
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/

# 运行策略（通过 AlphaForge）
./run.sh strategies/ag/trend_v1.py --symbols AG --freq daily --start 2022
```

## 指标使用

```python
import numpy as np
from indicators.momentum.roc import rate_of_change
from indicators.trend.adx import adx

closes = np.array([...])
roc_values = rate_of_change(closes, period=12)
```
