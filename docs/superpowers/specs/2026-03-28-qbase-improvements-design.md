# QBase 架构改进设计文档

**日期**: 2026-03-28
**状态**: Approved

---

## 概览

4 个子项目，分 2 个 Phase 实施：

- **Phase 1（并行）**: A 归因扩展 + B 数据分割统一
- **Phase 2（并行）**: C 模板重构+失败分析 + D 回撤归因+衰减监控

## 子项目 A：归因扩展 + Regime 覆盖矩阵

### 目标
从 v12 单策略归因扩展到任意策略批量归因，生成 Portfolio 级别 regime 覆盖矩阵。

### 新增文件
- `attribution/batch.py` — 批量归因编排器，输入 Portfolio 权重 JSON，自动对所有策略运行信号+行情归因
- `attribution/coverage.py` — Regime 覆盖矩阵生成器，输出 Markdown + JSON 双格式

### 输入/输出
- 输入: `strategies/*/portfolio/weights_*.json`
- 输出: `research_log/attribution/` 下各策略报告 + `portfolio_*_coverage.md` + `portfolio_*_coverage.json`

### 覆盖矩阵格式
每种 regime（Strong/Weak/No Trend × High/Low Vol × Active/Quiet）下，标注每个策略的胜率和盈亏。Portfolio 覆盖度 = 该 regime 下有几个策略盈利。目标：每种 regime ≥ 2 个策略盈利，否则标记 RED FLAG。

### 集成点
- 复用现有 `attribution/signal.py` 和 `attribution/regime.py`
- 未来 `portfolio/builder.py` 在选策略时可读取覆盖矩阵 JSON 发出警告

---

## 子项目 B：数据分割统一 + Walk-Forward

### 目标
1. 统一分割标准，新 Portfolio 必须三段式（按周期切）
2. 引入 Walk-Forward 验证
3. 简化质量门槛为两层

### 新增文件
- `strategies/walk_forward.py` — Walk-Forward 验证器

### Walk-Forward 设计
- 滚动窗口：训练 5 年 → 测试 1 年，步进 1 年
- 输出：每窗口 Sharpe + 聚合统计（Mean, Std, Worst, Win Rate）+ 行情标注
- 用法：`python strategies/walk_forward.py --strategy v12 --symbol AG --train-years 5 --test-years 1`

### 三段式分割标准
| 策略类型 | 训练 | 验证 | 测试 |
|---------|------|------|------|
| strong_trend | 6 品种 2016-2024 | — | AG/EC 跨品种（保持两段） |
| all_time/ag | AG 2012-2021 | AG 2022-2024 | AG 2025-2026 |
| all_time/i | I 2013-2021 | I 2022-2024 | I 2025-2026 |
| 未来新品种 | ≥2 完整周期 | ≥1 完整周期 | 标注 regime |

### 质量门槛（两层）
策略级：验证集 Sharpe > 0 + 归因完成 + 交易次数达标 + Walk-Forward win rate ≥ 50%
组合级：评分 ≥ 75 (B+) + Regime 覆盖无 RED FLAG + Bootstrap 95% CI 下界 > 0

### 文档更新
更新 docs/pipeline/step3, step4, step5 和 docs/reference/anti-overfit 中的分割标准和门槛描述。

---

## 子项目 C：策略模板重构 + All-Time 失败分析

### 目标
1. 拆分模板为 simple/full 两版，修复 V4 反模式
2. 分析 all_time 失败策略共性

### 模板重构
- `strategies/template_simple.py` — 极简版：1-2 指标 + ATR 止损，无加仓/止盈/多周期
- `strategies/template_full.py` — 完整版：金字塔加仓 + 分层止盈 + 多周期支持
- `strategies/template.py` — 保留，import simple 版（向后兼容）
- 两个模板都修复：ContractSpecManager 模块级缓存 + context.is_rollover

### 失败分析
- `strategies/all_time/ag/analyze_failures.py` — 读 optimization_results.json，按策略类型统计失败率，提取共性特征
- 输出：失败分布、共性特征、改进建议

---

## 子项目 D：回撤归因 + Alpha 衰减监控

### 目标
1. 分析 Portfolio 回撤发生条件
2. 检测指标有效性衰减
3. 预留实时监控接口

### 新增文件
- `attribution/drawdown.py` — 回撤归因：识别 MaxDD 期间各策略贡献 + 行情条件
- `attribution/decay.py` — Alpha 衰减检测：滚动 IC（Information Coefficient）检测指标有效性趋势
- `attribution/monitor.py` — 空壳接口，预留实时监控（当前 raise NotImplementedError）

### 衰减检测设计
- 核心指标：滚动 IC = corr(indicator_value, future_N_day_return)
- 滚动窗口：252 天（1年）
- 告警阈值：Mean IC < 0.05 连续 2 年
- 支持多品种多指标批量运行

---

## 依赖关系与实施顺序

```
Phase 1（并行）：A + B
Phase 2（并行）：C + D（D 依赖 A 的 regime 基础设施）
```

## 测试策略
- 每个新模块需要对应的 pytest 测试
- batch.py 用 Portfolio C (5策略) 做集成测试
- walk_forward.py 用 v12 + AG 做端到端验证
- 失败分析脚本用 AG all_time 100 策略验证
