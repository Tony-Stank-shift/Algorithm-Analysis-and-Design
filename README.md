# 图算法中的「白盒」与「黑盒」：从 Dijkstra 到图神经网络

《算法设计与分析》课程大作业（题目 2）。核心问题：**可解释性（Interpretability）vs. 性能（Performance）**——用同一个任务（学习型最短路）把两条路放进同一张表，从**准确率 / 鲁棒性 / 可解释性**三个轴量化比较。

报告：`report/main.pdf`（12 页）。所有数字都可由本仓库的脚本重跑得到。

## 一句话结论

> Bellman-Ford 的第 $k$ 轮就是「至多 $k$ 条边的最短路」，而 $k$ 层消息传递的感受野恰好是 $k$ 跳——两者是同一个式子换了记号。
> 在最短路这类**算法已知**的问题上，白盒在准确率、鲁棒性、可解释性上同时占优；**用有限深度的监督去「学」这个算子反而是负收益**：未训练的恒等 min-sum（即 Bellman-Ford）误差 $4.4\times10^{-7}$，训练 2 个参数后反而退化到 $0.024$。

## 环境

Python 3.12 + numpy + networkx + matplotlib + torch（**纯 CPU 即可**，无需 CUDA、无需 PyG），不下载任何外部数据集。

```bash
python exp/check_env.py
```

## 一键复现

```bash
python exp/run_all.py              # 全量，约 10 分钟（含 48 次训练）
python exp/run_all.py --skip-train # 跳过训练，约 2 分钟（复用已有的 results/black.json）
```

脚本会按依赖顺序跑完 15 个步骤，逐步打印判据，最后给出汇总表；任何一步失败，整条命令返回非 0。

单独运行某一步也可以，例如 `python exp/run_interp.py`。所有图由固定种子生成，同一命令重跑得到相同数字（计时列除外）。

## 目录

```
src/                库代码（无副作用）
  algorithms.py       Dijkstra / 同步 Bellman-Ford / Floyd / 路径还原
  graphs.py           ER / BA / grid / SBM 四类可控图，固定 seed 逐字节复现
  gnn.py              纯 torch 消息传递（sum / mean / min-sum 三种聚合）
  train.py            监督学距离的训练管线（含断点保存/加载）
  explain.py          路径解码、梯度显著性、Fidelity / 精度 / 任务敏感度
  timing.py           稳健计时（批内循环，避免亚毫秒噪声）
  viz.py              统一绘图风格（中文字体）
exp/                可执行脚本，每个对应报告里的一处结论
  check_*.py          验收脚本（跑不过就返回非 0）
  run_*.py            实验脚本（产出 results/）
  analyze_black.py    非配对统计（Mann-Whitney + bootstrap）
  run_all.py          一键复现
  _trained.py         按需训练并缓存 L1/L2 模型
results/            json / csv / 图（全部可重跑）
report/             LaTeX 源文件与 PDF
docs/               STATE.md（进度与已知问题）、DECISIONS.md（22 条决策记录）
```

## 脚本 ↔ 报告章节

| 报告 | 脚本 | 产物 |
|---|---|---|
| §2 白盒算法 | `check_algorithms.py` | 1000 图对拍 0 差异；Floyd 全源对拍 |
| §2 复杂度 | `run_complexity.py` | `complexity.csv`、`fig/complexity.png` |
| §3 鲁棒性命题 | `run_white_robust.py` | `robust_white.json`（500 例，0 违反） |
| §4 消息传递 = BF | `check_gnn.py` | 逐层逐点一致（float32 容差 1e-5） |
| §4 深度下界 | `run_depth.py` | `depth_bound.json`、`fig/depth_bound.png` |
| §6 准确率 | `run_black.py` + `run_accuracy.py` | `black.json`、`accuracy.json` |
| §6 训练偏差 | `analyze_black.py`、`run_failures.py` | `black_stats.json`、`fig/fail_g_bias.png` |
| §6 鲁棒性 | `run_robust.py` | `robust.json`、`fig/robust.png` |
| §6 可解释性 | `run_interp.py` | `interp.json`、`fig/interp.png` |
| — 判据是否有效 | `check_negative_controls.py` | 9 个桩实现必须让对应脚本失败 |

## 复现性与负控制

- 每个验收脚本都配有「桩实现必须失败」的负控制：把 Dijkstra 换成恒返回 0、把解释换成空集等，脚本必须返回非 0。当前 9/9 正确失败。
- 负控制运行前会备份 `results/`、结束后恢复，避免桩实现把真实结果覆盖成假数据。
- 统计用 Mann-Whitney U（非配对）+ bootstrap 95% CI，不用配对检验（8 个 seed 是 8 个不同模型，没有自然配对关系）。

## 已知边界

报告的 §7.3 逐条列出了结论的边界，其中最重要的两条：

1. **黑盒基线没有调到最好**：8 个种子里 6 个塌缩成常数预测。因此本文的强结论只到「未调好的 mean-pool 基线做不到」，不是「GNN 学不会最短路」。
2. **迭代预算不对称**：结构臂推理跑 128 轮（迭代到收敛），黑盒只有 8 层。这不是黑盒的固有属性，是我们的设计选择；同深度对照见报告表 6，SBM 上结构臂反而更差。

## 课程信息

算法设计与分析（JBK2634001，宋锐），2026 秋 第 9–16 周，2.5 学分。
