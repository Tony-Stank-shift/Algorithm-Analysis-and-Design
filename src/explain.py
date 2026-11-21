"""可解释性：路径解码、梯度显著性、以及四个可计算指标。

两条路线
--------
- 结构臂（min-sum）：解释就是**前向计算里的 argmin 链**，解出来是一条真实路径。
- 黑盒臂（mean-GNN）：没有内生结构，只能用**梯度显著性**做事后近似：
  score(e) = |∂ d_pred(t) / ∂ w_e|，取 top-k 边当"解释"。

四个指标（对同一预算 k 比较）
--------------------------
- Fidelity+（充分性）：只保留解释边，预测还保得住多少 = 1 - |p_S - p| / max(|p|,1)
- Fidelity-（必要性）：抽掉解释边，预测会变多少      = |p\\S - p| / max(|p|,1)
- Sparsity：解释边数 / 总边数
- Stability：轻微扰动后解释集合的 Jaccard
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import gnn


def restrict(rec: dict, keep: Sequence[int]) -> dict:
    """只保留 keep 里的边（节点集合不变）。"""
    keep = np.asarray(sorted(set(int(i) for i in keep)), dtype=np.int64)
    out = dict(rec)
    out["ei"] = rec["ei"][keep]
    out["w"] = rec["w"][keep]
    out["meta"] = dict(rec.get("meta", {}), E=int(len(keep)))
    return out


def pred_at(model, rec: dict, s: int, t: int, layers: Optional[int] = None) -> float:
    """模型对 (s -> t) 距离的预测（原始单位）。"""
    model.eval()
    b = gnn.collate([(rec, s)])
    with torch.no_grad():
        if isinstance(model, gnn.MinSumMPNN) and layers is not None:
            out = model(b, layers=layers)
        else:
            out = model(b)
    return float(out[t]) * gnn.W_SCALE


def min_sum_explanation(model, rec: dict, s: int, t: int,
                        layers: int) -> Dict:
    """结构臂：解出 argmin 链，并检验它是不是一条合法路径。

    注意「长度」必须用**模型自己的度量** g(w) 算，而不是原始权重 w：
    训练后的模型预测的是 g 下的距离，用 w 去核对会把 0.15% 的参数偏差
    误判成"路径不合法"（实测合法率会从 1.00 掉到 0.18）。
    """
    model.eval()
    b = gnn.collate([(rec, s)])
    _, _, nodes, edges, reached = model.explain(b, s, t, layers=layers)
    if edges:
        ws = torch.from_numpy(np.asarray(rec["w"])[edges]).float() / gnn.W_SCALE
        with torch.no_grad():
            length = float((model.edge_g(ws) * gnn.W_SCALE).sum())
    else:
        length = 0.0
    pred = pred_at(model, rec, s, t, layers=layers)
    structural = bool(reached and nodes and nodes[0] == s)
    valid = bool(structural and abs(length - pred) <= 1e-2 * max(pred, 1.0))
    return {"nodes": nodes, "edges": edges, "reached": bool(reached),
            "path_length": length, "pred": pred,
            "structural_ok": structural, "valid": valid}


def saliency_explanation(model, rec: dict, s: int, t: int, k: int) -> List[int]:
    """黑盒臂：按 |∂pred(t)/∂w_e| 取 top-k 边。"""
    model.eval()
    b = gnn.collate([(rec, s)])
    b["edge_w"] = b["edge_w"].clone().requires_grad_(True)
    out = model(b)
    grad, = torch.autograd.grad(out[t], b["edge_w"])
    E = len(rec["w"])
    score = torch.zeros(E)
    score.scatter_reduce_(0, torch.arange(2 * E) % E, grad.detach().abs(),
                          reduce="amax")
    k = min(k, E)
    return torch.topk(score, k).indices.tolist()


def fidelity(model, rec: dict, s: int, t: int, edges: Sequence[int],
             layers: Optional[int] = None) -> Tuple[float, float]:
    """返回 (Fidelity+, Fidelity-)。预测不可达时按哨兵值计入，并在调用方统计。"""
    p = pred_at(model, rec, s, t, layers=layers)
    denom = max(abs(p), 1.0)
    p_keep = pred_at(model, restrict(rec, edges), s, t, layers=layers)
    rest = [i for i in range(len(rec["w"])) if i not in set(edges)]
    p_drop = pred_at(model, restrict(rec, rest), s, t, layers=layers)
    return 1.0 - abs(p_keep - p) / denom, abs(p_drop - p) / denom


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def task_sensitivity(rec: dict, edges: Sequence[int], s: int, t: int,
                     factor: float = 10.0, cap: float = 10.0) -> Tuple[float, bool]:
    """把这些边的权重放大 factor 倍后，**真实**最短距离改变了多少（相对）。

    为什么不用"删除"：在平均度 3 的稀疏图上，删掉 4–6 条边经常直接让图断开，
    指标会被"不可达"封顶吃掉（实测三臂全撞上限 10，毫无区分度）。
    把权重放大不会破坏连通性，问的是同一个问题——
    "你标出来的这些边，对**任务答案**重不重要"。
    """
    import algorithms

    w = rec["w"].copy()
    if len(edges):
        idx = np.asarray([int(e) for e in edges], dtype=np.int64)
        w[idx] = w[idx] * factor
    adj = algorithms.build_adj(rec["n"], rec["ei"], w)
    d, _, _ = algorithms.dijkstra(rec["n"], adj, s)
    base = float(rec["d_true"][t])
    if base <= 0:
        return 0.0, False
    if not np.isfinite(d[t]):
        return cap, True
    return min(abs(float(d[t]) - base) / base, cap), False


def forms_path(rec: dict, edges: Sequence[int], s: int, t: int) -> bool:
    """这些边能不能连成一条 s->t 的路（结构臂必然可以，黑盒的 top-k 多半不能）。"""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(rec["n"]))
    for e in edges:
        e = int(e)
        G.add_edge(int(rec["ei"][e, 0]), int(rec["ei"][e, 1]))
    return nx.has_path(G, s, t)


def path_precision(rec: dict, edges: Sequence[int], s: int, t: int) -> float:
    """解释边里，有多少条真的落在最短路树上（0 = 与任务答案无关）。"""
    import algorithms

    adj = algorithms.build_adj(rec["n"], rec["ei"], rec["w"])
    d, parent, pedge = algorithms.dijkstra(rec["n"], adj, s)
    nodes = algorithms.path_nodes(parent, s, t)
    if not nodes:
        return 0.0
    on_path = set()
    for a, b in zip(nodes[:-1], nodes[1:]):
        for e in range(len(rec["w"])):
            if {int(rec["ei"][e, 0]), int(rec["ei"][e, 1])} == {int(a), int(b)}:
                on_path.add(e)
                break
    if not edges:
        return 0.0
    return len(on_path & set(int(e) for e in edges)) / len(edges)


def perturb_weights(rec: dict, rel: float, rng: np.random.Generator) -> dict:
    """结构不变，权重乘一个 [1-rel, 1+rel] 的随机因子（用于稳定性测试）。"""
    out = dict(rec)
    out["w"] = np.maximum(rec["w"] * (1.0 + rng.uniform(-rel, rel, len(rec["w"]))), 1e-3)
    out["meta"] = dict(rec.get("meta", {}))
    return out
