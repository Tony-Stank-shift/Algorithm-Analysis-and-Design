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

    # ---- 回归断言（检测指出上一版的 W=I 对照覆盖不到这些）----
    # (a) 随机权重 + 非负 X，与 numpy 参考实现比对
    torch.manual_seed(3)
    g2 = node_gnn.GCN(5, 4, 3, dropout=0.0)
    g2.eval()
    Xr = torch.rand(6, 5)
    Ar = torch.rand(6, 6)
    Ar = (Ar + Ar.T) / 2
    with torch.no_grad():
        out = g2(Ar, Xr)
    w1 = g2.w1.weight.detach().numpy()
    w2 = g2.w2.weight.detach().numpy()
    An = Ar.numpy()
    # 两层：h1 = relu(Â X W1ᵀ)，out = Â h1 W2ᵀ —— 别漏第二个 Â
    h1 = np.maximum(An @ Xr.numpy() @ w1.T, 0.0)
    ref = An @ h1 @ w2.T
    rand_ok = np.allclose(out.numpy(), ref, atol=1e-5)
    print(f"[{'OK' if rand_ok else 'X'}] 随机权重下与 NumPy 参考 (Â·relu(ÂXW1)·W2) 一致")

    # (b) dropout 只在 train 生效：eval 下两次前向必须相同
    g3 = node_gnn.GCN(5, 4, 3, dropout=0.5)
    g3.eval()
    with torch.no_grad():
        e1, e2 = g3(Ar, Xr), g3(Ar, Xr)
    g3.train()
    with torch.no_grad():
        t1, t2 = g3(Ar, Xr), g3(Ar, Xr)
    drop_ok = torch.allclose(e1, e2) and not torch.allclose(t1, t2)
    print(f"[{'OK' if drop_ok else 'X'}] dropout：eval 两次一致、train 两次不同")

    # (c) 损失只依赖 train_idx：打乱 test 标签后损失必须不变
    y = torch.from_numpy(d["labels"])
    tr = torch.from_numpy(d["train_idx"])
    with torch.no_grad():
        l1 = torch.nn.functional.cross_entropy(g(A_t, X_t)[tr], y[tr])
    y2 = y.clone()
    perm = torch.randperm(1000) + 1708
    y2[perm] = y2[perm][torch.randperm(1000)]
    with torch.no_grad():
        l2 = torch.nn.functional.cross_entropy(g(A_t, X_t)[tr], y2[tr])
    leak_ok = torch.allclose(l1, l2)
    print(f"[{'OK' if leak_ok else 'X'}] 打乱 test 标签后训练损失不变（只算 train_idx）")

    shape_ok = shape_ok and rand_ok and drop_ok and leak_ok
    ok = norm_ok and prop_ok and mlp_ok and shape_ok
    print("\n[PASS] 2.1" if ok else "\n[FAIL] 2.1")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
