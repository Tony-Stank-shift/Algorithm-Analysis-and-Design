"""对节点分类的攻击：随机加边基线 + FGA（Fast Gradient Attack）。

FGA 的思路：把邻接矩阵当连续变量，算损失对所有可能边的梯度，
挑"加了之后损失涨得最多"的那些边（即梯度最大的非边），加进图里。

关键实现点：
  · 梯度必须在 **Â 的输入 A** 上算，不能在 Â 上算——因为 Â 是 A 的函数，
    真正能改的是 A。
  · 已有的边与自环要掩掉，否则会在原地加权重（那不是"加边"）。
  · 每轮加完边都要重新归一化，所以攻击是迭代的：每轮加 n_add/rounds 条。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cora import normalized_adj, N_NODES
import scipy.sparse as sp


def dense_from_edges(n: int, edges: np.ndarray) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    A[edges[:, 0], edges[:, 1]] = 1.0
    A[edges[:, 1], edges[:, 0]] = 1.0
    return A


def edges_from_dense(A: np.ndarray) -> np.ndarray:
    """从稠密对称邻接恢复无向边表（上三角）。"""
    iu = np.triu_indices(A.shape[0], k=1)
    m = A[iu] > 0
    return np.stack([iu[0][m], iu[1][m]], axis=1).astype(np.int64)


def norm_np(A: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normalized_adj(sp.csr_matrix(A)).toarray())


def accuracy_under(model, A_np: np.ndarray, X: torch.Tensor, y: torch.Tensor,
                   idx: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        out = model(norm_np(A_np), X)
    pred = out[torch.from_numpy(idx)].argmax(1)
    return float((pred == y[torch.from_numpy(idx)]).float().mean())


def random_attack(A: np.ndarray, n_add: int, seed: int = 0) -> np.ndarray:
    """随机挑 n_add 条**当前不存在**的边加上（无向）。"""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    A = A.copy()
    added, tries = 0, 0
    while added < n_add and tries < 20 * n_add:
        tries += 1
        i, j = rng.integers(0, n, 2)
        if i == j or A[i, j] > 0:
            continue
        A[i, j] = A[j, i] = 1.0
        added += 1
    return A


def fga_attack(model, A_np: np.ndarray, X: torch.Tensor, y: torch.Tensor,
               idx: np.ndarray, n_add: int, rounds: int = 5,
               seed: int = 0) -> np.ndarray:
    """迭代 FGA：每轮对目标节点算梯度，加 n_add/rounds 条边。"""
    rng = np.random.default_rng(seed)
    n = A_np.shape[0]
    A = A_np.copy()
    per_round = max(1, n_add // rounds)
    tgt = torch.from_numpy(idx)

    for _ in range(rounds):
        A_var = torch.tensor(A, requires_grad=True)
        An = _norm_torch(A_var)
        out = model(An, X)
        loss = F.cross_entropy(out[tgt], y[tgt])
        model.zero_grad()
        loss.backward()
        # 必须用**带符号**梯度：加一条边让损失变化的近似量就是 ∂L/∂A_ij。
        # 取绝对值会把"最能降低损失"的边也挑进来，净效果反而提升准确率
        # （实测 abs 版本把 acc 从 0.808 抬到 0.875）。
        g = A_var.grad.detach()

        # 掩掉已有边、自环；只在上三角挑
        mask = torch.triu(torch.ones(n, n), diagonal=1) * (torch.tensor(A) == 0)
        score = (g * mask).flatten()
        top = torch.topk(score, k=per_round).indices
        rows = (top // n).numpy()
        cols = (top % n).numpy()
        A[rows, cols] = 1.0
        A[cols, rows] = 1.0

    return A


def _norm_torch(A: torch.Tensor) -> torch.Tensor:
    """可微的对称归一化 Â = D^{-1/2}(A+I)D^{-1/2}（A 是 dense，含自环）。"""
    n = A.shape[0]
    A = A + torch.eye(n)
    deg = A.sum(1)
    d = deg.clamp_min(1e-8).pow(-0.5)
    return A * d.unsqueeze(0) * d.unsqueeze(1)
