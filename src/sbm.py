"""合成 SBM：标签由社区决定，于是"真正的判别边"是已知的。

为什么需要它：在 Cora 上没有任何人知道"正确的解释"是什么，只能测保真度——
而主报告已经证明保真度会被常数预测器骗过。SBM 提供了真值：
  · 节点标签 = 所属社区
  · 真正携带标签信号的边 = 社区内部的边
  · 跨社区边是干扰项
于是"解释的 precision"变成一个可以对着真值算的量。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.sparse as sp


def make_sbm(n_per_block: int = 100, n_blocks: int = 3, p_in: float = 0.06,
             p_out: float = 0.003, n_feat: int = 20, noise: float = 0.6,
             seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    labels = np.repeat(np.arange(n_blocks), n_per_block)

    # 特征：社区 one-hot + 高斯噪声。特征本身有信号但不完全可分，
    # 所以 GCN 必须借邻居来"去噪"——这正是要解释的行为。
    feats = np.zeros((n, n_feat), dtype=np.float32)
    feats[np.arange(n), labels % n_feat] = 1.0
    feats += rng.normal(0, noise, size=feats.shape).astype(np.float32)

    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                rows += [i, j]
                cols += [j, i]
    A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(n, n)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    edges = np.unique(np.sort(np.asarray(A.nonzero()).T, axis=1), axis=0)
    same_block = labels[edges[:, 0]] == labels[edges[:, 1]]

    return {
        "n": n,
        "d": n_feat,
        "classes": n_blocks,
        "features": feats,
        "labels": labels,
        "adj": A,
        "edges": edges,
        "edge_is_signal": same_block,     # 真值：这条边是不是判别边
        "seed": seed,
    }


def make_anchor(n: int = 300, n_classes: int = 3, n_noise: int = 2,
                noise_feat: float = 1.0, n_feat: int = 20,
                seed: int = 0) -> Dict:
    """锚点图：每个节点的标签由**唯一一条**锚点边决定，其余是噪声边。

    为什么换掉 SBM：SBM 里每个节点有 ~6 个同社区邻居，单条边只占信号的 1/6，
    删掉谁都无所谓——边级解释在那个设定下没有可解释的信号（实测解释器在
    一半节点上退化成常数掩码）。锚点图相反：每条命定的边都不可替代。

    构造：
      · 前 n_classes 个节点是锚点，特征为 one-hot，标签即其序号
      · 其余节点特征为纯噪声（所以只能靠结构判类）
      · 每个普通节点连 1 条锚点边（连到自己类的锚点）+ n_noise 条随机边
      · 真值 = 那条锚点边
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n).astype(np.int64)
    labels[:n_classes] = np.arange(n_classes)

    feats = rng.normal(0, noise_feat, size=(n, n_feat)).astype(np.float32)
    for c in range(n_classes):
        feats[c] = 0.0
        feats[c, c] = 1.0

    rows, cols = [], []
    anchor_edge = {}
    for v in range(n_classes, n):
        a = int(labels[v])                 # 锚点编号 = 类编号
        rows += [v, a]
        cols += [a, v]
        anchor_edge[v] = len(rows) // 2 - 1 if rows else 0
        for _ in range(n_noise):
            u = int(rng.integers(0, n))
            if u == v:
                continue
            rows += [v, u]
            cols += [u, v]

    A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(n, n)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    edges = np.unique(np.sort(np.asarray(A.nonzero()).T, axis=1), axis=0)

    # 真值标记：把每个节点的锚点边找出来（按端点对匹配）
    is_anchor = np.zeros(len(edges), dtype=bool)
    for v, a in [(v, int(labels[v])) for v in range(n_classes, n)]:
        pair = (min(v, a), max(v, a))
        hit = np.nonzero((edges[:, 0] == pair[0]) & (edges[:, 1] == pair[1]))[0]
        if len(hit):
            is_anchor[hit[0]] = True

    return {
        "n": n, "d": n_feat, "classes": n_classes,
        "features": feats, "labels": labels, "adj": A, "edges": edges,
        "edge_is_signal": is_anchor,
        "anchor_of": {v: int(labels[v]) for v in range(n_classes, n)},
        "seed": seed,
    }


def split(n: int, seed: int = 0, n_train: int = 150, n_val: int = 100):
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(n)
    tr = np.sort(idx[:n_train])
    va = np.sort(idx[n_train:n_train + n_val])
    te = np.sort(idx[n_train + n_val:])
    return {"train_idx": tr, "val_idx": va, "test_idx": te}
