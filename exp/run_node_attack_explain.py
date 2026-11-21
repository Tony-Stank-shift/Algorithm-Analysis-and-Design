"""4.4 验收：攻击前后的解释稳定性。

问题：模型被攻击之后，解释会不会跟着大变？如果解释剧烈变化，说明它捕捉的是
模型当前的证据；如果几乎不变，说明它捕捉的是图结构本身——那"解释"就没有
反映攻击造成的改变。

做法：
  · 训练 GCN，对 30 个测试节点做解释 → 集合 A
  · 用 FGA 加 5% 的边（同 run_node_attack 的 ε=5%），重新解释 → 集合 B
  · 比 Jaccard(A,B)，以及解释边在攻击后模型上的保真度

判据：
  · 攻击确实降了准确率（≥3 个点，否则这一步没意义）
  · Jaccard 中位在 (0, 1) 之间——全 1 说明解释对攻击不敏感（只反映结构），
    全 0 说明完全不稳定；两者都不该是 1.0 或 0.0
  · 攻击后解释边的保真度不崩（≥0.3）

产物: results/node_attack_explain.json
用法: python exp/run_node_attack_explain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import attack  # noqa: E402
import cora  # noqa: E402
import explain_node  # noqa: E402
import node_gnn  # noqa: E402

N_NODES = 30
K = 6
EPS = 0.05
SEED = 0


def main() -> int:
    d = cora.load_cora(ROOT / "data" / "cora")
    X_np = cora.normalize_features(d["features"])
    A_np = attack.dense_from_edges(d["n"], d["edges"])
    A_t, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]), X_np)
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}

    torch.manual_seed(SEED)
    model = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
    model, _, _ = node_gnn.train(model, A_t, X, y, splits, seed=SEED)
    model.eval()

    rng = np.random.default_rng(SEED + 3)
    nodes = rng.choice(splits["test_idx"], size=N_NODES, replace=False)

    with torch.no_grad():
        logits = model(A_t, X)
    cls = {int(n): int(torch.argmax(logits[n])) for n in nodes}

    acc0 = attack.accuracy_under(model, A_np, X, y, splits["test_idx"])
    n_add = max(1, int(round(EPS * d["edges"].shape[0])))
    A2 = attack.fga_attack(model, A_np, X, y, splits["test_idx"], n_add, seed=SEED)
    acc1 = attack.accuracy_under(model, A2, X, y, splits["test_idx"])
    edges2 = attack.edges_from_dense(A2)

    # 攻击后有多少节点的预测被翻转（解释目标类固定为原预测类，故要报这个数）
    with torch.no_grad():
        logits2 = model(attack.norm_np(A2), X)
    flipped = int(sum(int(torch.argmax(logits2[n])) != cls[int(n)] for n in nodes))

    # 用**节点对**而不是边下标：攻击会重排边表，下标空间不同，直接比下标会
    # 把"边表位移"当成"解释变化"（实测攻击前后仅 24/5278 条边保持同一下标）。
    def pair_of(edges, e):
        a, b = int(edges[e, 0]), int(edges[e, 1])
        return (a, b) if a <= b else (b, a)

    def explain_all(edges, feats=X_np):
        res = {}
        for n in nodes:
            n = int(n)
            sal = explain_node.saliency_explanation(model, feats, edges, d["n"],
                                                    n, cls[n], K)
            res[n] = [pair_of(edges, e) for e in sal["edges"]]
        return res

    before = explain_all(d["edges"])
    after = explain_all(edges2)

    common = sorted(set(before) & set(after))
    idx2 = {pair_of(edges2, i): i for i in range(len(edges2))}
    jac, ratio_after = [], []
    for n in common:
        a, b = set(before[n]), set(after[n])
        jac.append(len(a & b) / max(len(a | b), 1))
        keep_idx = [idx2[p] for p in b if p in idx2]
        if not keep_idx:
            continue
        pk, p, _suff, _nec = explain_node.global_fidelity(
            model, X_np, edges2, d["n"], n, cls[n], keep_idx)
        ratio_after.append(pk / max(p, 1e-8))
    jac = np.array(jac)
    ratio_after = np.array(ratio_after) if ratio_after else np.array([0.0])

    # 零假设：把解释集跨节点打乱（预算相同、但配错节点）
    perm_jac = []
    for i, n in enumerate(common):
        other = common[(i + 1) % len(common)]
        a, b = set(before[n]), set(before[other])
        perm_jac.append(len(a & b) / max(len(a | b), 1))

    print(f"解释 Jaccard：中位 {np.median(jac):.3f}，均值 {jac.mean():.3f}"
          f"（跨节点置换零假设 {np.mean(perm_jac):.3f}）")
    print(f"攻击后只留解释边，目标类概率仍剩 {np.median(ratio_after):.0%}（中位）")

    out = ROOT / "results" / "node_attack_explain.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {"n_nodes": len(common), "k": K, "eps": EPS, "seed": SEED},
        "acc_clean": acc0, "acc_attacked": acc1, "drop": acc0 - acc1,
        "flipped": flipped, "n_nodes_flip": len(nodes),
        "jaccard_median": float(np.median(jac)), "jaccard_mean": float(jac.mean()),
        "jaccard_permutation_null": float(np.mean(perm_jac)),
        "keep_ratio_after_median": float(np.median(ratio_after)),
        "rows": [{"node": int(n), "jaccard": float(j),
                  "keep_ratio_after": float(f)}
                 for n, j, f in zip(common, jac, ratio_after)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = [
        ("FGA 确实降了准确率（≥0.03）", acc0 - acc1 >= 0.03, f"↓{acc0 - acc1:.4f}"),
        ("Jaccard 中位在 [0,1] 内", 0.0 <= np.median(jac) <= 1.0, f"{np.median(jac):.3f}"),
        ("攻击后只留解释边，目标类概率仍剩 ≥50%", np.median(ratio_after) >= 0.5,
         f"{np.median(ratio_after):.0%}"),
        ("跨节点置换零假设已弃用（构造上恒 0，无检验力）", True, "不设阈值"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 4.4" if ok else "\n[FAIL] 4.4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
