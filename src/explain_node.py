"""边掩码解释器（GNNExplainer 的简化版）。

思路：冻结模型，为目标节点学一张边权重掩码 m ∈ [0,1]^E，
      让"只用 m 大的边"时目标节点的预测尽量保持，同时尽量稀疏。

    min_m  -log p(y_v | A ⊙ m) + λ · mean(m)

实现细节（都会在报告里写明，避免被当成黑魔法）：
  · 只在目标节点的 L 跳子图上优化（Cora 上 2 跳约 30–60 个节点），
    局部重新归一化 Â' = D'^{-1/2}(A' + I)D'^{-1/2}。
    这是"子图解释"的常规近似，不是全局精确解。
  · 掩码参数用 logits 表示，经 sigmoid 得到 m；温度从 1 线性退火到 0.1，
    让 m 逐渐逼近 0/1。
  · 取 top-k 条边（k 由调用方给，与主报告解释实验同预算）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)


def _norm_from_weights(rows, cols, w, n):
    """由带权边表构造对称归一化 Â = D^{-1/2}(A+I)D^{-1/2}。"""
    deg = torch.zeros(n, dtype=w.dtype)
    deg = deg.index_add(0, rows, w).index_add(0, cols, w)
    d = (deg + 1.0).clamp_min(1e-8).pow(-0.5)
    vals = w * d[rows] * d[cols]
    idx = torch.stack([rows, cols])
    A = torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()
    di = torch.arange(n)
    A_I = torch.sparse_coo_tensor(torch.stack([di, di]), d, (n, n))
    return (A + A_I).coalesce()


def explain_node(model, features: np.ndarray, edges: np.ndarray, labels: np.ndarray,
                 node: int, k: int, n_hops: int = 2, steps: int = 150,
                 lr: float = 0.05, lam: float = 0.5, seed: int = 0):
    """返回 top-k 解释边（原始边编号）与优化过程信息。

    lam 不能太小：0.01 时模型本来就很自信，"全保留"的损失几乎是 0，
    掩码会退化成全 1（实测 8 个节点里 2 个 std=0）。0.5 会逼它删到快崩，
    此时活下来的边才接近"必要"。
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    adj = {}
    for eid, (u, v) in enumerate(edges):
        adj.setdefault(int(u), []).append((int(v), eid))
        adj.setdefault(int(v), []).append((int(u), eid))

    # L 跳子图
    cur, seen = {node}, {node}
    for _ in range(n_hops):
        nxt = set()
        for u in cur:
            for v, _e in adj.get(u, []):
                if v not in seen:
                    nxt.add(v)
        seen |= nxt
        cur = nxt
    sub = np.array(sorted(seen))
    remap = {int(o): i for i, o in enumerate(sub)}
    sel = np.array([e for e in range(len(edges))
                    if int(edges[e, 0]) in remap and int(edges[e, 1]) in remap])
    if len(sel) == 0:
        return {"edges": [], "mask": np.array([]), "loss": float("nan"),
                "subgraph_size": len(sub)}

    rows = torch.from_numpy(np.array([remap[int(edges[e, 0])] for e in sel]))
    cols = torch.from_numpy(np.array([remap[int(edges[e, 1])] for e in sel]))
    X = torch.from_numpy(features[sub].astype(np.float32))
    tgt = remap[int(node)]
    y_t = int(labels[node])

    logits_m = torch.zeros(len(sel), requires_grad=True)
    opt = torch.optim.Adam([logits_m], lr=lr)
    model.eval()
    losses = []
    for step in range(steps):
        temp = 1.0 - 0.9 * step / max(steps - 1, 1)
        m = torch.sigmoid(logits_m / temp)
        A_hat = _norm_from_weights(rows, cols, m, len(sub))
        out = model(A_hat, X)
        loss = F.cross_entropy(out[tgt].unsqueeze(0), torch.tensor([y_t]))
        loss = loss + lam * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        m_final = torch.sigmoid(logits_m / 0.1).numpy()
    order = np.argsort(-m_final)[:k]
    chosen = sel[order]
    return {
        "edges": chosen.tolist(),
        "mask": m_final[order].tolist(),
        "loss": float(losses[-1]),
        "loss_first": float(losses[0]),
        "subgraph_size": int(len(sub)),
        "n_candidate": int(len(sel)),
    }


def fidelity(model, features: np.ndarray, edges: np.ndarray, labels: np.ndarray,
             node: int, kept: list[int], n_hops: int = 2):
    """只保留 kept 边后，目标节点原预测类概率的变化。

    返回 (p_keep, p_full, fid_plus, fid_minus)：
      fid_plus  = 1 - |p_keep - p_full| / max(p_full, eps)   —— 保留它够不够
      fid_minus = 1 - |p_del  - p_full| / max(p_full, eps)   —— 去掉它少不少
    这里比较的是**原预测类**的概率，不是 max 概率（预测翻转时两者不同）。
    """
    model.eval()
    full = _forward_probs(model, features, edges, node, None, n_hops)
    cls = int(torch.argmax(full))
    p = float(full[cls])
    pk = float(_forward_probs(model, features, edges, node, kept, n_hops)[cls])
    rest = [e for e in range(len(edges)) if e not in set(kept)]
    pd = float(_forward_probs(model, features, edges, node, rest, n_hops)[cls])
    eps = 1e-8
    return pk, p, 1 - abs(pk - p) / max(p, eps), 1 - abs(pd - p) / max(p, eps)


def _forward_probs(model, features, edges, node, keep, n_hops):
    """在 node 的 L 跳子图上前向，返回整条概率向量。keep=None 表示全保留。"""
    n = features.shape[0]
    adj = {}
    for eid, (u, v) in enumerate(edges):
        adj.setdefault(int(u), []).append((int(v), eid))
        adj.setdefault(int(v), []).append((int(u), eid))
    cur, seen = {node}, {node}
    for _ in range(n_hops):
        nxt = set()
        for u in cur:
            for v, _e in adj.get(u, []):
                if v not in seen:
                    nxt.add(v)
        seen |= nxt
        cur = nxt
    sub = np.array(sorted(seen))
    remap = {int(o): i for i, o in enumerate(sub)}
    keep_set = None if keep is None else set(int(e) for e in keep)
    sel = [e for e in range(len(edges))
           if int(edges[e, 0]) in remap and int(edges[e, 1]) in remap
           and (keep_set is None or e in keep_set)]
    if not sel:
        return torch.ones(model.w2.out_features) / model.w2.out_features
    rows = torch.tensor([remap[int(edges[e, 0])] for e in sel])
    cols = torch.tensor([remap[int(edges[e, 1])] for e in sel])
    w = torch.ones(len(sel))
    X = torch.from_numpy(features[sub].astype(np.float32))
    with torch.no_grad():
        A_hat = _norm_from_weights(rows, cols, w, len(sub))
        out = model(A_hat, X)
        prob = F.softmax(out[remap[int(node)]], dim=0)
    return prob
