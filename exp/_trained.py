"""按需训练并缓存阶段 3 的模型（默认 d=8, seed=0），给阶段 4 复用。

缓存到 results/models/，避免每次重训。种子在建模**之前**设（D14）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))

import torch  # noqa: E402

import gnn  # noqa: E402
import run_black as RB  # noqa: E402
import train as T  # noqa: E402

CACHE = ROOT / "results" / "models"
TRAIN_SEEDS = range(2000, 2200)


def get(arm: str = "minsum", depth: int = 8, seed: int = 0,
        epochs: int = 40, sources: int = 2):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{arm}_d{depth}_s{seed}.pt"
    # 与 run_black 一致：单线程训练，否则同种子跨进程不可复现
    torch.set_num_threads(1)
    torch.manual_seed(seed)                    # 必须早于 make_model
    model = RB.make_model(arm, depth)
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        import os
        if os.environ.get("JARVIS_NO_TRAIN"):
            raise RuntimeError(
                f"缺少缓存 {path.name}，而当前是 --skip-train 模式。"
                "请先跑一次完整的 python exp/run_black.py 生成 results/models/。")
        tr = T.build_samples(RB.gen_er, TRAIN_SEEDS, sources, seed0=7)
        T.train_model(model, tr, epochs=epochs, seed=seed)
        torch.save(model.state_dict(), path)
    model.eval()
    return model


def identity() -> gnn.MinSumMPNN:
    m = gnn.MinSumMPNN(layers=128, identity=True)
    m.eval()
    return m
