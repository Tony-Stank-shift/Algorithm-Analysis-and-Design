"""1.4 验收：Cora 数据体检。

判据：
  · 形状 2708×1433、7 类、5429 条无向边
  · 标准划分 140 / 500 / 1000，且三者互不相交
  · 同配率 ≈ 0.81（Cora 的已知性质，用来确认标签与图结构对得上）
  · 特征为 0/1、无孤立点被错当成测试点
  · 对称归一化后对角线/行和性质正确

用法: python exp/check_cora.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import cora  # noqa: E402


def main() -> int:
    d = cora.load_cora(ROOT / "data" / "cora")
    F, y, A = d["features"], d["labels"], d["adj"]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    labeled = np.zeros(d["n"], dtype=bool)
    labeled[np.concatenate([tr, va, te])] = True

    # 1) 形状与规模
    shape_ok = (F.shape == (2708, 1433) and d["edges"].shape[0] == cora.N_EDGES
                and len(np.unique(y)) == 7)
    print(f"[{'OK' if shape_ok else 'X'}] 形状：特征 {F.shape}、去重后无向边 {d['edges'].shape[0]}"
          f"（10858 条有向记录 / 2 = 5429，但其中 151 对重复，去重后 {cora.N_EDGES}）、类 {len(np.unique(y))}")

    # 2) 划分大小与互斥
    split_ok = (len(tr) == 140 and len(va) == 500 and len(te) == 1000)
    disjoint = (len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0
                and len(set(va) & set(te)) == 0)
    print(f"[{'OK' if split_ok and disjoint else 'X'}] 划分：train {len(tr)} / val {len(va)} / test {len(te)}，互斥={disjoint}")

    # 3) 每类训练样本数
    per_class = [int((y[tr] == c).sum()) for c in range(7)]
    per_ok = all(v == 20 for v in per_class)
    print(f"[{'OK' if per_ok else 'X'}] 每类训练样本：{per_class}")

    # 4) 同配率
    h = cora.homophily(d["edges"], y, labeled)
    h_ok = 0.75 <= h <= 0.86
    print(f"[{'OK' if h_ok else 'X'}] 同配率：{h:.3f}（Cora 已知约 0.81）")

    # 5) 特征取值
    vals = np.unique(F)
    f_ok = set(vals.tolist()) <= {0.0, 1.0}
    print(f"[{'OK' if f_ok else 'X'}] 特征取值：{vals.tolist()[:5]}，行和均值 {F.sum(1).mean():.1f}")

    # 6) 邻接性质
    sym = abs(A - A.T).max() == 0
    diag = A.diagonal().sum() == 0
    deg = np.asarray(A.sum(1)).ravel()
    print(f"[{'OK' if sym and diag else 'X'}] 邻接：对称={sym}，无自环={diag}，"
          f"度 {deg.min():.0f}–{deg.max():.0f}（中位 {np.median(deg):.0f}）")

    # 7) 归一化邻接：对角线应等于 1/(deg+1)，行和为正
    An = cora.normalized_adj(A)
    rowsum = np.asarray(An.sum(1)).ravel()
    deg1 = deg + 1.0
    diag_ok = np.allclose(np.asarray(An.diagonal()), 1.0 / deg1, atol=1e-5)
    n_ok = diag_ok and np.all(rowsum > 0)
    print(f"[{'OK' if n_ok else 'X'}] 归一化邻接：对角=1/(deg+1) → {diag_ok}，"
          f"行和均为正 → {np.all(rowsum > 0)}")

    ok = shape_ok and split_ok and disjoint and per_ok and h_ok and f_ok and sym and diag and n_ok
    print("\n[PASS] 1.4" if ok else "\n[FAIL] 1.4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
