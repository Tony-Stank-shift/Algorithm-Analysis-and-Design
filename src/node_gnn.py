"""节点分类模型：2 层 GCN（对称归一化）与 MLP 对照。

GCN 的一层：H' = σ(Â H W)，其中 Â = D^{-1/2}(A+I)D^{-1/2}。
MLP 用同一个分类头，但每层把 Â 换成单位阵——它看不到邻居，是"无结构"基线。
两者的差异只在这一步，因此"GCN 比 MLP 好多少"能干净地归因于图结构。

全部用 dense 张量：Cora 只有 2708 个节点，Â 稠密化后 29 MB，换来实现简单、
且后面做边掩码解释时可以直接改 Â。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.w1 = nn.Linear(d_in, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_classes, bias=False)
        self.dropout = dropout

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        h = F.dropout(X, p=self.dropout, training=self.training)
        h = torch.spmm(A, self.w1(h))
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return torch.spmm(A, self.w2(h))


class MLP(nn.Module):
    """同样的两层，但没有 Â —— 每个节点只看自己的特征。"""

    def __init__(self, d_in: int, hidden: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.w1 = nn.Linear(d_in, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_classes, bias=False)
        self.dropout = dropout

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        h = F.dropout(X, p=self.dropout, training=self.training)
        h = self.w1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.w2(h)


def to_torch(A_norm, feats) -> tuple[torch.Tensor, torch.Tensor]:
    """scipy 稀疏 Â / numpy 特征 → dense torch 张量。"""
    A = A_norm.toarray() if hasattr(A_norm, "toarray") else np.asarray(A_norm)
    return (torch.from_numpy(np.asarray(A, dtype=np.float32)),
            torch.from_numpy(np.asarray(feats, dtype=np.float32)))


def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: np.ndarray) -> float:
    pred = logits[torch.from_numpy(idx)].argmax(1)
    return float((pred == y[torch.from_numpy(idx)]).float().mean())


def train(model, A, X, y, splits, epochs: int = 200, lr: float = 0.01,
          weight_decay: float = 5e-4, patience: int = 100, seed: int = 0):
    """全监督 loss，但只用 train_idx 的标签；val 用于早停与选模型。"""
    torch.set_num_threads(1)   # 单线程才保证同种子跨进程可复现（多线程归约顺序不定）
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr = torch.from_numpy(splits["train_idx"])
    va = torch.from_numpy(splits["val_idx"])
    best = {"val_loss": float("inf"), "state": None, "epoch": -1}
    hist = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(A, X)
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(A, X)
            vl = float(F.cross_entropy(out[va], y[va]))
            va_acc = accuracy(out, y, splits["val_idx"])
        hist.append({"epoch": ep, "train_loss": float(loss.detach()),
                     "val_loss": vl, "val_acc": va_acc})
        if vl < best["val_loss"] - 1e-6:
            best = {"val_loss": vl, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": ep}
        elif ep - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    return model, hist, best
