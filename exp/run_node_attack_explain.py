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

    # 攻击前解释（用原始边表）
    def explain_all(edges, feats=X_np):
        res = {}
        for n in nodes:
            n = int(n)
            ex = explain_node.explain_node(model, feats, edges, n, cls[n], K, seed=SEED)
            if len(ex["cand"]) < 2 * K:
                continue
            res[n] = [int(e) for e in ex["cand"][np.argsort(-ex["mask"])[:K]]]
        return res

    before = explain_all(d["edges"])
    after = explain_all(edges2)

    common = sorted(set(before) & set(after))
    jac, fid_after = [], []
    for n in common:
        a, b = set(before[n]), set(after[n])
        jac.append(len(a & b) / max(len(a | b), 1))
        _, _, fp, _ = explain_node.global_fidelity(model, X_np, edges2, d["n"], n, cls[n],
                                                   sorted(b))
        fid_after.append(fp)
    jac = np.array(jac)
    fid_after = np.array(fid_after)

    # 对照：随机边集之间的 Jaccard（解释完全无关时的水平）
    rand_jac = []
    for n in common[:20]:
        a = set(rng.choice(len(d["edges"]), size=K, replace=False).tolist())
        b = set(rng.choice(len(d["edges"]), size=K, replace=False).tolist())
        rand_jac.append(len(a & b) / max(len(a | b), 1))

    print(f"攻击：ε=5%（+{n_add} 边），acc {acc0:.4f} → {acc1:.4f}（↓{acc0 - acc1:.4f}）；"
          f"{flipped}/{len(nodes)} 个解释节点的预测被翻转")
    print(f"解释 Jaccard：中位 {np.median(jac):.3f}，均值 {jac.mean():.3f}"
          f"（随机边集对照 {np.mean(rand_jac):.3f}）")
    print(f"攻击后解释边的全局 Fid+ 中位 {np.median(fid_after):.3f}")

    out = ROOT / "results" / "node_attack_explain.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {"n_nodes": len(common), "k": K, "eps": EPS, "seed": SEED},
        "acc_clean": acc0, "acc_attacked": acc1, "drop": acc0 - acc1,
        "flipped": flipped, "n_nodes_flip": len(nodes),
        "jaccard_median": float(np.median(jac)), "jaccard_mean": float(jac.mean()),
        "jaccard_random": float(np.mean(rand_jac)),
        "fid_plus_after_median": float(np.median(fid_after)),
        "rows": [{"node": int(n), "jaccard": float(j),
                  "fid_after": float(f)} for n, j, f in zip(common, jac, fid_after)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = [
        ("FGA 确实降了准确率（≥0.03）", acc0 - acc1 >= 0.03, f"↓{acc0 - acc1:.4f}"),
        ("Jaccard 中位在 (0,1) 内（既不僵化也不乱）",
         0.0 < np.median(jac) < 1.0, f"{np.median(jac):.3f}"),
        ("攻击后解释仍高于随机边集水平",
         np.median(jac) > np.mean(rand_jac) * 2, f"{np.median(jac):.3f} vs {np.mean(rand_jac):.3f}"),
        ("攻击后解释边保真度 ≥0.3", np.median(fid_after) >= 0.3,
         f"{np.median(fid_after):.3f}"),
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
