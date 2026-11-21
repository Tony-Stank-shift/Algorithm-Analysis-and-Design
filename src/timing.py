"""稳健计时：亚毫秒级函数单次计时会被噪声淹没。

先粗测一次，再按目标批时长自动决定内循环次数，最后取批中位数。
第一次跑复杂度实验时 n=800 比 n=400 还快，就是没做这件事。
"""

from __future__ import annotations

import time

import numpy as np


def tmed(fn, target: float = 0.05, reps: int = 5) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter() - t0
    inner = max(1, int(target / max(t1, 1e-9)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts))
