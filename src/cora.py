"""Cora 数据加载与标准划分（Planetoid pickle，零新依赖）。

节点编号规则（Planetoid 原样）：
  · 0..1707   —— allx / ally 覆盖的节点（有特征、有标签）
  · 1708..2707 —— tx / ty 覆盖的 1000 个测试节点，test.index 给出它们的原始编号
  剩余节点（不在上面两块里的）特征全零、是孤立点。

划分（**公开划分**，与 GCN 原论文一致）：
  train = 节点 0..139（Planetoid 的 ind.cora.x 就是这 140 个），val = 140..639，test = 1708..2707。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.sparse as sp

N_NODES = 2708
N_FEATURES = 1433
N_CLASSES = 7
# 边数有两种口径，别混：
#   · 10858 条有向记录 ÷ 2 = 5429 —— 文献里最常引的数字，但它**没去重**
#   · 去重后的无向边 = 5278 —— graph 里有 151 对节点被重复记录了一次
N_EDGES_DIRECTED = 10858
N_EDGES = 5278


def _load(data_dir: Path, name: str):
    with open(data_dir / f"ind.cora.{name}", "rb") as f:
        return pickle.load(f, encoding="latin1")  # Python2 序列化


def load_cora(data_dir: str | Path = "data/cora") -> Dict:
    data_dir = Path(data_dir)
    if not (data_dir / "ind.cora.x").exists():
        raise FileNotFoundError(
            f"{data_dir} 下没有 Cora 数据，先跑 python exp/fetch_cora.py")

    x, y = _load(data_dir, "x"), _load(data_dir, "y")
    tx, ty = _load(data_dir, "tx"), _load(data_dir, "ty")
    allx, ally = _load(data_dir, "allx"), _load(data_dir, "ally")
    graph = _load(data_dir, "graph")
    test_idx_orig = np.loadtxt(data_dir / "ind.cora.test.index", dtype=np.int64)
    # 关键：tx / ty 的第 j 行属于**原始顺序**里的第 j 个测试节点，不是排序后的第 j 个。
    # 按排序编号赋值会把标签整体洗牌（实测同配率从 0.82 掉到 0.28）。
    test_idx = np.sort(test_idx_orig)

    assert test_idx.shape[0] == 1000
    assert set(test_idx.tolist()) == set(range(1708, 2708)), "测试节点应恰好是 1708..2707"

    feats = np.zeros((N_NODES, N_FEATURES), dtype=np.float32)
    feats[:allx.shape[0]] = np.asarray(allx.todense(), dtype=np.float32)
    feats[test_idx_orig] = np.asarray(tx.todense(), dtype=np.float32)

    labels = np.zeros(N_NODES, dtype=np.int64)
    labels[:ally.shape[0]] = ally.argmax(1)
    labels[test_idx_orig] = ty.argmax(1)

    # graph 是无向图的邻接字典（每方向各一条），对称化并去自环
    rows, cols = [], []
    for u, nbrs in graph.items():
        for v in nbrs:
            rows.append(int(u))
            cols.append(int(v))
    A = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(N_NODES, N_NODES)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()

    assert len(rows) == N_EDGES_DIRECTED, f"有向记录应为 {N_EDGES_DIRECTED}，实际 {len(rows)}"
    edges = np.asarray(A.nonzero()).T                      # 有向（对称化后两方向都在）
    edges = np.unique(np.sort(edges, axis=1), axis=0)      # 无向、去重
    assert edges.shape[0] == N_EDGES, (
        f"去重后边数应为 {N_EDGES}，实际 {edges.shape[0]}"
        "（文献的 5429 = 10858/2，未去重）")

    splits = make_splits(labels, test_idx, seed=0)

    return {
        "features": feats,
        "labels": labels,
        "adj": A,
        "edges": edges,
        "test_idx": test_idx,
        **splits,
        "n": N_NODES,
        "d": N_FEATURES,
        "classes": N_CLASSES,
    }


def make_splits(labels: np.ndarray, test_idx: np.ndarray, seed: int = 0):
    """**公开划分**（Planetoid 原样）：train = 0..139，val = 140..639，test = 1708..2707。

    上一版按类分层重采样，只有 10/140 与公开训练集重合——那不是公开划分，
    与论文数字不可比（检测实测：公开划分 3 seed 均值 0.8170，重采样 0.8103）。
    `ind.cora.x` 恰好就是这 140 个训练节点的特征，`x == allx[:140]` 可作断言。
    """
    train = np.arange(0, 140)
    val = np.arange(140, 640)
    return {"train_idx": train, "val_idx": val, "test_idx": test_idx}


def normalize_features(feats: np.ndarray) -> np.ndarray:
    """按行 L1 归一化（GCN 的标准预处理）。"""
    rowsum = feats.sum(1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return feats / rowsum


def normalized_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    """对称归一化 Â = D^{-1/2}(A+I)D^{-1/2}。"""
    A = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = sp.diags(dinv, format="csr")
    return (D @ A @ D).astype(np.float32)


def homophily(edges: np.ndarray, labels: np.ndarray, labeled: np.ndarray) -> float:
    """同配率：两端都已知标签的边中，标签相同的比例。"""
    mask = labeled[edges[:, 0]] & labeled[edges[:, 1]]
    e = edges[mask]
    return float((labels[e[:, 0]] == labels[e[:, 1]]).mean())
