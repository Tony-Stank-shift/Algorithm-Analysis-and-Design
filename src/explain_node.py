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


def explain_node(model, features: np.ndarray, edges: np.ndarray, node: int,
                 target_class: int, k: int, n_hops: int = 2, steps: int = 200,
                 lr: float = 0.05, lam: float = 0.05, seed: int = 0):
    """返回**全量候选边与掩码**，由调用方决定怎么取 top-k。

    上一版返回 top-k 的掩码值，导致一个致命问题：掩码在温度退火后塌成硬 0/1、
    只留 1–3 条边，而调用方仍按 k=6 取，剩下的槽位从几百个并列的 0 里按索引抽——
    等于把解释器自己判死的边算成"解释边"。

    现在温度固定为 1（不做退火），掩码是软的、几乎无并列；同时返回
    n_kept = #{m > 0.5}，调用方据此判断该节点的解释是否可判定。
    """
    rng = np.random.default_rng(seed)
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
    sel = np.array([e for e in range(len(edges))
                    if int(edges[e, 0]) in remap and int(edges[e, 1]) in remap])
    if len(sel) == 0:
        return {"cand": np.array([], int), "mask": np.array([]), "n_kept": 0,
                "subgraph_size": len(sub)}

    rows = torch.from_numpy(np.array([remap[int(edges[e, 0])] for e in sel]))
    cols = torch.from_numpy(np.array([remap[int(edges[e, 1])] for e in sel]))
    X = torch.from_numpy(features[sub].astype(np.float32))
    tgt = remap[int(node)]

    logits_m = torch.zeros(len(sel), requires_grad=True)
    opt = torch.optim.Adam([logits_m], lr=lr)
    model.eval()
    losses = []
    for _step in range(steps):
        m = torch.sigmoid(logits_m)          # 温度固定 1，避免塌成硬 0/1
        A_hat = _norm_from_weights(rows, cols, m, len(sub))
        out = model(A_hat, X)
        loss = -F.log_softmax(out[tgt], dim=0)[int(target_class)]
        loss = loss + lam * m.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        m_final = torch.sigmoid(logits_m).numpy()
    return {"cand": sel, "mask": m_final, "n_kept": int((m_final > 0.5).sum()),
            "subgraph_size": int(len(sub)), "loss": float(losses[-1]),
            "loss_first": float(losses[0])}


def global_fidelity(model, features: np.ndarray, edges: np.ndarray, n_nodes: int,
                    node: int, cls: int, kept: list[int], n_hops: int = 2):
    """在**全局图**上算保真度：只保留 kept 边，整图重新归一化，跑一次完整前向。

    与 fidelity() 的区别：后者在 2 跳子图上本地重归一化，那个替身会翻类
    （实测 Cora 上 28% 的节点给的是另一个预测类）。这里不建替身，
    概率就是模型给的概率，所以数字可以读作"对模型保真"。

    返回 (p_keep, p_full, fid_plus, fid_minus)。
    """
    model.eval()
    X = torch.from_numpy(features.astype(np.float32))

    def probs(keep):
        idx = np.arange(len(edges)) if keep is None else np.asarray(keep, dtype=int)
        e = edges[idx]
        u = torch.from_numpy(np.asarray(e[:, 0], dtype=np.int64))
        v = torch.from_numpy(np.asarray(e[:, 1], dtype=np.int64))
        rows = torch.cat([u, v])
        cols = torch.cat([v, u])
        A = _norm_from_weights(rows, cols, torch.ones(len(rows)), n_nodes)
        with torch.no_grad():
            return F.softmax(model(A, X)[node], dim=0)

    p = float(probs(None)[cls])
    pk = float(probs(kept)[cls])
    rest = [e for e in range(len(edges)) if e not in set(kept)]
    pd = float(probs(rest)[cls])
    eps = 1e-8
    return pk, p, 1 - abs(pk - p) / max(p, eps), 1 - abs(pd - p) / max(p, eps)


def fidelity(model, features: np.ndarray, edges: np.ndarray,
             node: int, cls: int, kept: list[int], n_hops: int = 2):
    """只保留 kept 边后，**目标类 cls** 在 2 跳子图**本地替身**上的概率变化。

    返回 (p_keep, p_full, fid_plus, fid_minus)。
    注意：这里的概率是 2 跳子图上本地重归一化的结果，与全局前向不总是同一个
    预测（检测实测 6/39 节点 argmax 不同），所以它衡量的是"对局部算子"的保真，
    报告里必须这样标注。
    """
    model.eval()
    p = float(_forward_probs(model, features, edges, node, None, n_hops)[cls])
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
