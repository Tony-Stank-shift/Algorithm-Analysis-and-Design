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
- 训练固定为单线程（`torch.set_num_threads(1)`）：多线程下 `index_add_` / `scatter_reduce_` 的归约顺序不固定，浮点加法不满足结合律，同一个种子跨进程会跑出不同结果（实测某 seed 两次得到 MAE 2.49 与 2.20）。单线程后同种子逐位可复现，代价是训练慢一些。
- 统计用 Mann-Whitney U（非配对）+ bootstrap 95% CI，不用配对检验（8 个 seed 是 8 个不同模型，没有自然配对关系）。

## 已知边界

报告的 §7.3 逐条列出了结论的边界，其中最重要的两条：

1. **黑盒基线没有调到最好**：8 个种子里 6 个塌缩成常数预测。因此本文的强结论只到「未调好的 mean-pool 基线做不到」，不是「GNN 学不会最短路」。
2. **迭代预算不对称**：结构臂推理跑 128 轮（迭代到收敛），黑盒只有 8 层。这不是黑盒的固有属性，是我们的设计选择；同深度对照见报告表 6，SBM 上结构臂反而更差。

## 课程信息

算法设计与分析（JBK2634001，宋锐），2026 秋 第 9–16 周，2.5 学分。

## 补充实验（B 线）：节点分类 + 可解释性 + 攻击

主报告的结论有个前提：**算法已知**。节点分类没有白盒算法，正好拿来检验"没有白盒时，解释还可不可信"。
完整内容见 `report/appendixB.tex`（可单独编译：`cd report && tectonic appB-main.tex`）。

### 快速开始

```bash
python exp/fetch_cora.py              # 下载 8 个 Planetoid 文件到 data/cora/（约 570 KB，不入库）
python exp/check_cora.py              # 数据体检（形状/划分/同配率）
python exp/check_node_gnn.py          # GCN 与手算对照 + 三条回归断言
python exp/run_node_class.py          # 训练 GCN 与 MLP（8 种子）
python exp/check_node_repro.py        # 同种子跨进程复现 + 与 JSON 对账
python exp/run_node_planted.py        # SBM 上的解释真值检验（消融）
python exp/run_node_explain.py        # Cora 上的解释与跨模型稳定性
python exp/run_node_attack.py         # 随机加边 vs FGA
python exp/run_node_attack_explain.py # 攻击前后的解释稳定性
python exp/check_b_negative_controls.py  # 三条负控制
python exp/plot_b.py                  # 结果图
```

### 关键数字

| 项 | 结果 |
|---|---|
| GCN（公开划分） | **0.8162**（原论文 81.5%） |
| MLP（无结构对照） | 0.5539 |
| 随机加边 10% / FGA 10% | −0.017 / **−0.119** |
| 解释边 vs 随机边（SBM 消融真值） | Δ **0.0673 vs 0.0044**，Wilcoxon p=5e-11 |
| oracle 效率 | 中位 **0.939**（41/44 个节点可算） |
| 无法判定 | 1/45（掩码并列） |
| Cora 上的掩码解释器 | **88% 可判定**（105/120） |
| Cora 充分性 / 必要性 | 1.31 vs 随机 1.21 / 删解释边 −0.123 vs 删随机边 −0.019 |
| 攻击后的解释 | Jaccard **1.000**——解释对 FGA 完全不敏感 |
| Cora 上的掩码解释器 | **0/120 可判定**——候选上百条时失效 |
| Cora 上的 fidelity | 1.211 vs 随机 1.208，**无判别力** |

### 已知限制

- 解释器实现踩过一个致命 bug（归一化算子度数重复累加、自环写成 d 而非 d²、只传单方向边表），已修；修前所有解释结论作废
- 仍有 1/45 个 SBM 节点排不出序；跳过节点的候选池与度数分布未落盘，是否有偏无法审计
- 只测了 ε=5% 一个攻击预算与 k=6 一个解释预算，未做敏感性曲线
- 跨模型 Jaccard 0.714 与"攻击前后 Jaccard 1.000"一并说明：解释稳定，但未必跟踪模型当前的证据
