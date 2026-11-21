"""5.1 验收：三条负控制——桩实现必须让判据失败。

1. 把 GCN 换成 MLP（去掉图）：Cora 上准确率必须掉到 ≤0.65
2. 把解释器换成**均匀掩码**（等价于随便挑 k 条边）：SBM 上的"解释重要性倍率"必须塌到 ~1
3. 把标签打乱：GCN 准确率必须掉到 1/7 附近

三条都不通过才说明判据有效；任何一条"仍然通过"都说明对应判据没有约束力。

用法: python exp/check_b_negative_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cora  # noqa: E402
import explain_node  # noqa: E402
import node_gnn  # noqa: E402
import sbm  # noqa: E402


def control_mlp(seed: int = 0, epochs: int = 100) -> float:
    d = cora.load_cora(ROOT / "data" / "cora")
    A, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]),
                             cora.normalize_features(d["features"]))
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}
    torch.manual_seed(seed)
    m = node_gnn.MLP(d["d"], 16, d["classes"])
    m, _, _ = node_gnn.train(m, A, X, y, splits, epochs=epochs, seed=seed)
    m.eval()
    with torch.no_grad():
        return node_gnn.accuracy(m(A, X), y, splits["test_idx"])


def control_shuffled_labels(seed: int = 0, epochs: int = 100):
    """打乱标签后，模型学不到东西，应该塌到**多数类**水平（不是 1/7）。

    Cora 的类分布不均衡（最大类占 30%），随机猜的"最优策略"就是全猜多数类，
    所以正确的地板是多数类比例，不是 1/7。
    """
    d = cora.load_cora(ROOT / "data" / "cora")
    A, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]),
                             cora.normalize_features(d["features"]))
    rng = np.random.default_rng(seed)
    y_shuf = torch.from_numpy(rng.permutation(d["labels"]))
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}
    torch.manual_seed(seed)
    m = node_gnn.GCN(d["d"], 16, d["classes"])
    m, _, _ = node_gnn.train(m, A, X, y_shuf, splits, epochs=epochs, seed=seed)
    m.eval()
    with torch.no_grad():
        acc = node_gnn.accuracy(m(A, X), y_shuf, splits["test_idx"])
    major = float(np.bincount(d["labels"]).max() / len(d["labels"]))
    return acc, major


def control_uniform_mask(seed: int = 0, n_nodes: int = 8, k: int = 6) -> float:
    """桩实现：把解释边换成**掩码最低的 k 条**（bottom-k），即与真值反着来。

    上一版用"随机 k 条"是自证的（随机 vs 随机，比值必然 ≈1）；bottom-k 才是
    系统性错误排序，如果主判据还能通过，说明主判据没有牙。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = sbm.make_sbm(seed=seed)
    splits = sbm.split(g["n"], seed=seed)
    feats = cora.normalize_features(g["features"])
    A, X = node_gnn.to_torch(cora.normalized_adj(g["adj"]), feats)
    y = torch.from_numpy(g["labels"])
    model = node_gnn.GCN(g["d"], 16, g["classes"], dropout=0.3)
    model, _, _ = node_gnn.train(model, A, X, y, splits, epochs=200, seed=seed)
    model.eval()
    with torch.no_grad():
        logits = model(A, X)

    rng = np.random.default_rng(seed + 7)
    dt, dr = [], []
    for node in splits["test_idx"][:n_nodes]:
        node = int(node)
        cls = int(torch.argmax(logits[node]))
        ex = explain_node.explain_node(model, feats, g["edges"], node, cls, k, seed=seed)
        cand, mask = ex["cand"], ex["mask"]
        if len(cand) < 2 * k:
            continue
        order = np.argsort(mask)[:k]          # bottom-k：最不重要的边
        p_full = float(explain_node._forward_probs(model, feats, g["edges"], node, None, 2)[cls])
        delta = np.array([p_full - float(explain_node._forward_probs(
            model, feats, g["edges"], node, [e for e in cand if e != c], 2)[cls])
            for c in cand])
        dt.append(float(delta[order].mean()))
        dr.append(float(np.mean([
            delta[rng.choice(len(cand), size=k, replace=False)].mean() for _ in range(20)])))
    return float(np.mean(dt) / max(np.mean(dr), 1e-12)) if np.mean(dr) > 1e-9 else 0.0


def main() -> int:
    acc_mlp = control_mlp()
    acc_shuf, major = control_shuffled_labels()
    ratio_bottom = control_uniform_mask()

    checks = [
        ("MLP 顶替 GCN：准确率 ≤0.65", acc_mlp <= 0.65, f"{acc_mlp:.4f}"),
        ("打乱标签：准确率 ≤ 多数类+0.05", acc_shuf <= major + 0.05,
         f"{acc_shuf:.4f} vs 多数类 {major:.3f}"),
        ("bottom-k 桩实现：解释重要性倍率 <2（不能复现主结论）", ratio_bottom < 2.0,
         f"{ratio_bottom:.2f}×"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 5.1 负控制全部生效" if ok else "\n[FAIL] 5.1 有判据没被桩实现打穿")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

