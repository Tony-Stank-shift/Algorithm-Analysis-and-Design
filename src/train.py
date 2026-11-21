"""训练与评测管线：监督学「源点到各点的距离」。

约定
----
- 网络在**归一化空间**工作（距离与边权都除以 W_SCALE=10），对外报告时乘回来。
- min-sum 臂的初始状态是 LARGE=1e4（表示"还没连上"），目标恒有限（图都连通），
  所以损失里要把预测 clamp 到 [0, CLAMP]，否则前几轮会被 1e4 的量级炸掉。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

import algorithms
import gnn

CLAMP = 20.0  # 归一化空间的上限：真实直径最大约 8，20 足够宽松


def retarget(rec: dict, s: int) -> dict:
    """把记录改成「以 s 为源点」的版本（重算 d_true）。"""
    adj = algorithms.build_adj(rec["n"], rec["ei"], rec["w"])
    d, _, _ = algorithms.dijkstra(rec["n"], adj, s)
    out = dict(rec)
    out["s"] = s
    out["d_true"] = d
    return out


def build_samples(gen: Callable[[int], dict], seeds, k_sources: int,
                  seed0: int = 0) -> List[Tuple[dict, int]]:
    samples: List[Tuple[dict, int]] = []
    for sd in seeds:
        rec = gen(sd)
        rng = np.random.default_rng(seed0 * 1_000_003 + sd)
        k = min(k_sources, rec["n"])
        srcs = rng.choice(rec["n"], size=k, replace=False)
        for s in srcs:
            samples.append((retarget(rec, int(s)), int(s)))
    return samples


def train_model(model: torch.nn.Module, samples, epochs: int = 30, lr: float = 1e-2,
                batch: int = 8, seed: int = 0, verbose: bool = False):
    """注意：**模型的构造必须发生在 torch.manual_seed 之后**。

    早先版本在 train_model 内部才设种子，而调用方先 make_model()，
    导致第 k 次运行的初始化取决于前 k-1 次消耗的 RNG——per-seed 结果无法复现
    （第三方检测实测两个进程的 initial_seed 不同）。现在种子由调用方设。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(samples))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            b = gnn.collate([samples[j] for j in idx])
            raw = model(b)
            pred = raw.clamp(0.0, CLAMP)
            # 只监督「在 K 层内已经连上」的点：
            # 结构臂在 K 层内到不了的远点恒为 LARGE，对它们算损失既没梯度也没信息，
            # 反而把有用的近点信号淹没。黑盒臂输出有界，掩码恒为 1，所以两臂规则一致。
            mask = (raw < 1e3).float()
            loss = (((pred - b["y"]) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(idx)
        losses.append(tot / len(samples))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  mse={losses[-1]:.5f}")
    return losses


def save(model: torch.nn.Module, path) -> None:
    torch.save(model.state_dict(), path)


def load(model: torch.nn.Module, path) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, samples, batch: int = 16,
             layers: int | None = None) -> Dict[str, float]:
    """返回 MAE / 中位误差 / 绝对命中率(±0.5) / 相对命中率(5%) / 最坏误差。

    layers 只对 MinSumMPNN 生效：给了比训练深度更大的值，就是「推理时多迭代」。
    """
    model.eval()
    extrap = isinstance(model, gnn.MinSumMPNN) and layers is not None
    errs, trues = [], []
    for i in range(0, len(samples), batch):
        b = gnn.collate(samples[i:i + batch])
        out = model(b, layers=layers) if extrap else model(b)
        pred = out.clamp(0.0, CLAMP) * gnn.W_SCALE
        true = b["y"] * gnn.W_SCALE
        errs.append((pred - true).abs())
        trues.append(true)
    e = torch.cat(errs).numpy()
    t = torch.cat(trues).numpy()
    rel = e / np.maximum(np.abs(t), 1e-9)
    return {
        "mae": float(e.mean()),
        "med_err": float(np.median(e)),
        "hit_abs05": float((e < 0.5).mean()),
        "hit_rel5": float((rel < 0.05).mean()),
        "max_err": float(e.max()),
        "n": int(e.size),
    }
