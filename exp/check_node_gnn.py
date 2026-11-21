"""2.1 验收：GCN 的一层是否算对（与手算小例子逐元素对齐）。

构造一个 4 节点的路径图 0-1-2-3，特征取成一个可分辨的矩阵，
手算 Â、再手算一层 H'=σ(Â X W)（W 取单位阵、σ 取恒等），与实现比对。

用法: python exp/check_node_gnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import cora  # noqa: E402
import node_gnn  # noqa: E402


def main() -> int:
    # ---- 手算例：路径图 0-1-2-3 ----
    edges = [(0, 1), (1, 2), (2, 3)]
    A = sp.coo_matrix((np.ones(6, np.float32),
                       ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])),
                      shape=(4, 4)).tocsr()
    An = cora.normalized_adj(A).toarray()
    # A+I 的度数：2,3,3,2（含自环）→ d^-1/2 = 1/√2, 1/√3, 1/√3, 1/√2
    s6 = 1 / np.sqrt(6.0)
    expect = np.array([
        [0.5, s6, 0.0, 0.0],
        [s6, 1 / 3, 1 / 3, 0.0],
        [0.0, 1 / 3, 1 / 3, s6],
        [0.0, 0.0, s6, 0.5],
    ], dtype=np.float32)
    norm_ok = np.allclose(An, expect, atol=1e-6)
    print(f"[{'OK' if norm_ok else 'X'}] 对称归一化与手算一致")

    # ---- 两层传播：W1=W2=I 且输入非负时 relu 为恒等，故输出应为 Â²X ----
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    model = node_gnn.GCN(d_in=2, hidden=2, n_classes=2, dropout=0.0)
    with torch.no_grad():
        model.w1.weight.copy_(torch.eye(2))
        model.w2.weight.copy_(torch.eye(2))
        model.eval()
        out = model(torch.from_numpy(An), X)
    hand = torch.from_numpy(expect @ expect @ X.numpy())
    prop_ok = torch.allclose(out, hand, atol=1e-6)
    print(f"[{'OK' if prop_ok else 'X'}] 两层传播 Â²X 与手算一致")

    # ---- MLP 必须忽略 Â：换一个 Â，输出不该变 ----
    mlp = node_gnn.MLP(d_in=2, hidden=2, n_classes=2, dropout=0.0)
    with torch.no_grad():
        mlp.w1.weight.copy_(torch.eye(2))
        mlp.w2.weight.copy_(torch.eye(2))
        mlp.eval()
        o1 = mlp(torch.from_numpy(An), X)
        o2 = mlp(torch.eye(4), X)
    mlp_ok = torch.allclose(o1, o2, atol=1e-6) and not torch.allclose(o1, out, atol=1e-4)
    print(f"[{'OK' if mlp_ok else 'X'}] MLP 对 Â 不敏感、且与 GCN 输出不同")

    # ---- 真实数据上前向形状 ----
    d = cora.load_cora(ROOT / "data" / "cora")
    An_full = cora.normalized_adj(d["adj"])
    A_t, X_t = node_gnn.to_torch(An_full, cora.normalize_features(d["features"]))
    g = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.0)
    g.eval()
    with torch.no_grad():
        logits = g(A_t, X_t)
    shape_ok = tuple(logits.shape) == (2708, 7) and torch.isfinite(logits).all()
    print(f"[{'OK' if shape_ok else 'X'}] 真实数据前向：{tuple(logits.shape)}，全有限={bool(torch.isfinite(logits).all())}")

    ok = norm_ok and prop_ok and mlp_ok and shape_ok
    print("\n[PASS] 2.1" if ok else "\n[FAIL] 2.1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
