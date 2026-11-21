# 图算法中的「白盒」与「黑盒」：从 Dijkstra 到图神经网络

《算法设计与分析》课程大作业（题目 2）。核心问题：**可解释性（Interpretability）vs. 性能（Performance）**，在**准确率 / 鲁棒性 / 可解释性**三个轴上量化比较。

报告：`report/main.pdf`（**17 页**）。所有数字都可由本仓库的脚本重跑得到。

## 结论

> **“可解释性 vs. 性能”的权衡曲线，只在算法未知时才真正出现。**

报告分两部分，用两个任务把这个判断做实：

**第一部分（算法已知，最短路）**：白盒三轴全胜，而且**训练是负收益**。$k$ 层 min-sum 消息传递与同步 Bellman-Ford 的第 $k$ 轮逐点等价；未训练的恒等 min-sum 误差 $\sim10^{-6}$，训练 2 个参数后反而退化到 $0.024$。白盒的鲁棒性不是“更抗打”，而是三条**可证**命题（500 例零违反）。

**第二部分（算法未知，Cora 节点分类）**：没有白盒可对照，问题变成“能不能验证一个解释”。GCN 0.8162 对 MLP 0.5539（参数量相同），FGA 攻击比随机强 7 倍；由于真实数据没有“正确的解释”，另造带真值的 SBM 环境，测得解释器达到 oracle 的 93.9\%。一个反直觉的发现：**攻击把准确率打掉 7.5 个点，解释却几乎逐条不变**（Jaccard 中位 1.000）——稳定，但不跟踪模型改变的主意。

## 环境

Python 3.12 + numpy + networkx + matplotlib + torch（**纯 CPU**，无需 CUDA、无需 PyG）。
第二部分额外需要 scipy，以及一份 Cora（脚本自动下载，约 570 KB，不入库）。

```bash
python exp/check_env.py
```

## 一键复现

```bash
# 第一部分：最短路（15 步，约 12 分钟；含 48 次训练）
python exp/run_all.py
python exp/run_all.py --skip-train     # 跳过训练，约 2 分钟

# 第二部分：节点分类（需先下载 Cora）
python exp/fetch_cora.py
python exp/run_all_b.py                # 约 4 分钟
python exp/run_all_b.py --quick        # 只跑验收脚本，约 40 秒
```

两个脚本都按依赖顺序执行、逐步打印判据、最后给汇总表；任何一步失败，整条命令返回非 0。

## 目录

```
src/                库代码（无副作用）
  algorithms.py       Dijkstra / 同步 Bellman-Ford / Floyd / 路径还原
  graphs.py           ER / BA / grid / SBM 四类可控图，固定 seed 逐字节复现
  gnn.py              纯 torch 消息传递（sum / mean / min-sum）
  train.py            监督学距离的训练管线
  explain.py          路径解码、梯度显著性、Fidelity / 精度 / 任务敏感度
  cora.py             Planetoid 读取、边表去重、公开划分
  node_gnn.py         2 层 GCN 与同参数量 MLP
  explain_node.py     边掩码解释器 + 梯度显著性（子图归一化与训练一致）
  attack.py           随机加边与 FGA（带符号梯度）
  sbm.py              随机块模型（带真值的合成环境）
  timing.py / viz.py  计时与绘图
exp/                可执行脚本，每个对应报告里的一处结论
  check_*.py          验收脚本（跑不过就返回非 0）
  run_*.py            实验脚本（产出 results/）
  run_all.py          A 线一键复现
  run_all_b.py        B 线一键复现
  _trained.py         按需训练并缓存 L1/L2 模型
results/            json / csv / 图 / 模型缓存（全部可重跑）
report/             LaTeX 源（main.tex + m01–m09）与 16 页 PDF
docs/               STATE.md（进度与已知问题）、DECISIONS.md（决策记录）
```

## 脚本 ↔ 报告章节

| 报告 | 脚本 | 产物 |
|---|---|---|
| §2 白盒算法 | `check_algorithms.py` | 1000 图对拍 0 差异；Floyd 全源对拍 |
| §2 复杂度 | `run_complexity.py` | `complexity.csv`、`fig/complexity.png` |
| §3 鲁棒性命题 | `run_white_robust.py` | `robust_white.json`（500 例，0 违反） |
| §4 消息传递 = BF | `check_gnn.py` | 逐层逐点一致（float32 容差 1e-5） |
| §4 深度下界 | `run_depth.py` | `depth_bound.json`、`fig/depth_bound.png` |
| §5 准确率 | `run_black.py` + `run_accuracy.py` | `black.json`、`accuracy.json` |
| §5 训练偏差 | `analyze_black.py`、`run_failures.py` | `black_stats.json`、`fig/fail_g_bias.png` |
| §5 鲁棒性 | `run_robust.py` | `robust.json`、`fig/robust.png` |
| §5 可解释性 | `run_interp.py` | `interp.json`、`fig/interp.png` |
| §6 数据与模型 | `check_cora.py`、`check_node_gnn.py` | 形状/划分/同配率体检、GCN 手算对照 |
| §6 分类 | `run_node_class.py`、`check_node_repro.py` | `node_class.json`（GCN 0.8162 / MLP 0.5539） |
| §7 真值检验 | `run_node_planted.py` | `node_planted.json`、`fig/b_explain.png` |
| §7 Cora 解释 | `run_node_explain.py` | `node_explain.json` |
| §7 攻击 | `run_node_attack.py`、`run_node_attack_explain.py` | `node_attack*.json` |
| — 判据是否有效 | `check_negative_controls.py`（9 条）、`check_b_negative_controls.py`（3 条） | 桩实现必须让对应脚本失败 |

## 复现性与负控制

- 每个验收脚本都配有「桩实现必须失败」的负控制：把 Dijkstra 换成恒返回 0、把解释换成空集、用 MLP 顶替 GCN、用 bottom-$k$ 顶替 top-$k$。当前 **12/12 正确失败**。
- 负控制运行前会备份 `results/`、结束后恢复，避免桩实现把真实结果覆盖成假数据。
- 训练固定单线程（`torch.set_num_threads(1)`）：多线程下 `index_add_` / `scatter_reduce_` 的归约顺序不固定，浮点加法不满足结合律，同一个种子跨进程会跑出不同结果（实测某 seed 两次得到 MAE 2.49 与 2.20）。单线程后同种子逐位可复现。
- 统计用 Mann-Whitney U（非配对）+ bootstrap 95% CI，不用配对检验（8 个 seed 是 8 个不同模型，没有自然配对关系）。

## 已知边界

报告的 §8.4 逐条列出，最重要的四条：

1. **A 线黑盒基线没有调到最好**：8 个种子里 6 个塌缩成常数预测。强结论只到「未调好的 mean-pool 基线做不到」，不是「GNN 学不会最短路」；Nerem 等人（arXiv:2503.19173）已证明带合适正则的 GNN 能精确实现 Bellman-Ford。
2. **迭代预算不对称**：结构臂推理跑 128 轮，黑盒只有 8 层。同深度对照见报告表 4：SBM 上结构臂反而更差（35.38 对 28.76）。
3. **B 线的“真值”来自合成图**：SBM 的社区结构与真实图差距很大，有真值的代价是不真实。
4. **B 线预算单一**：只测了 ε=5% 一个攻击预算与 k=6 一个解释预算，没有敏感性曲线。

## 复现踩过的坑（写在这里，免得后人重蹈）

- A 线：`collate` 最初只建单向边，消息传递死路；玩具例子恰好所有边由小号指向大号，骗过了测试。
- B 线：子图归一化算子的三个细节（度数只累加一次、自环贡献 $d^2$、边表双向）写错两处，导致掩码塌成常数、解释完全失效；修复前所有解释结论作废。
- B 线：FGA 若对梯度取绝对值，会同时选中「最伤害」与「最补」的边，净效果反而提升准确率（0.808 → 0.875）。

## 课程信息

算法设计与分析（JBK2634001，宋锐），2026 秋 第 9–16 周，2.5 学分。

> 说明：`report/b-main.*` 与 `appendixB.tex` 是 B 线的独立草稿，已被合并版 `report/main.pdf` 取代，保留作过程记录。

