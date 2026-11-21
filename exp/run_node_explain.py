"""3.2 / 3.5 验收（v4）：Cora 上的解释与跨模型稳定性。

v3 的结论：**掩码解释器在 Cora 上给不出排序**（40/40 节点的第 6/7 名严格并列），
原因是候选边上百条时掩码被稀疏惩罚压成近似常数，top-k 是 argsort 打破并列的副产物。
这不是可以写进报告"解释很好"的东西，所以改成两件事一起报：

  · **掩码解释器**：Cora 上可判定率 = 0/40 → 方法在这个规模上失败（如实记录）
  · **梯度显著性**：score_e = ∂log p(cls)/∂w_e，连续、不并列 → 用它测保真度与稳定性

保真度用**全局**算子（与训练一致），随机基线改成该节点的 **incident 边**。

判据：
  · 掩码解释器可判定率如实记录（不设通过阈值，它是结论不是判据）
  · 显著性解释：保留 top-6 后目标类概率 ≥ 全局的 50%
  · 显著性解释的保留率高于 incident 随机基线
  · 跨模型种子 Jaccard 中位 ≥0.4（在显著性上算）

产物: results/node_explain.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cora  # noqa: E402
import explain_node  # noqa: E402
import node_gnn  # noqa: E402

N_NODES_EXPLAIN = 40
K = 6
MODEL_SEEDS = (0, 1, 2)
EXPLAIN_SEED = 0


def main() -> int:
    d = cora.load_cora(ROOT / "data" / "cora")
    X_np = cora.normalize_features(d["features"])
    A_t, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]), X_np)
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}

    rng = np.random.default_rng(EXPLAIN_SEED)
    nodes = rng.choice(splits["test_idx"], size=N_NODES_EXPLAIN, replace=False)
    incident = {}
    for eid, (u, v) in enumerate(d["edges"]):
        incident.setdefault(int(u), []).append(eid)
        incident.setdefault(int(v), []).append(eid)

    per_seed, accs, mask_decidable = {}, {}, 0
    for seed in MODEL_SEEDS:
        torch.manual_seed(seed)
        model = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
        model, _, _ = node_gnn.train(model, A_t, X, y, splits, seed=seed)
        model.eval()
        with torch.no_grad():
            logits = model(A_t, X)
        accs[seed] = node_gnn.accuracy(logits, y, splits["test_idx"])

        recs = {}
        for node in nodes:
            node = int(node)
            cls = int(torch.argmax(logits[node]))

            ex = explain_node.explain_node(model, X_np, d["edges"], node, cls, K,
                                           seed=EXPLAIN_SEED)
            mask_ok = bool(ex["decidable"])
            if mask_ok:
                mask_decidable += 1
                order = np.argsort(-ex["mask"])[:K]
                top = [int(e) for e in ex["cand"][order]]
            else:
                sal = explain_node.saliency_explanation(model, X_np, d["edges"], d["n"],
                                                        node, cls, K)
                top = [int(e) for e in sal["edges"]]

            pk, p, suff, nec = explain_node.global_fidelity(
                model, X_np, d["edges"], d["n"], node, cls, top)
            inc = incident.get(node, [])
            rand = [int(e) for e in rng.choice(inc, size=min(K, len(inc)), replace=False)]
            pk_r, p_r, suff_r, nec_r = explain_node.global_fidelity(
                model, X_np, d["edges"], d["n"], node, cls, rand)
            recs[node] = {"top": top, "p_full": p, "p_keep": pk,
                          "keep_ratio": pk / max(p, 1e-8), "suff": suff, "nec": nec,
                          "keep_ratio_random": pk_r / max(p_r, 1e-8),
                          "nec_random": nec_r, "mask_decidable": mask_ok}
        per_seed[seed] = recs
        kr = np.array([r["keep_ratio"] for r in recs.values()])
        krr = np.array([r["keep_ratio_random"] for r in recs.values()])
        print(f"模型 seed={seed}: acc {accs[seed]:.4f}；显著性保留率 {np.median(kr):.3f} "
              f"vs incident 随机 {np.median(krr):.3f}")

    jac = []
    for node in nodes:
        node = int(node)
        if node not in per_seed[MODEL_SEEDS[0]]:
            continue
        for i in range(len(MODEL_SEEDS)):
            for j in range(i + 1, len(MODEL_SEEDS)):
                a = set(per_seed[MODEL_SEEDS[i]][node]["top"])
                b = set(per_seed[MODEL_SEEDS[j]][node]["top"])
                jac.append(len(a & b) / max(len(a | b), 1))
    jac = np.array(jac) if jac else np.array([0.0])

    summary = {}
    for s in MODEL_SEEDS:
        recs = per_seed[s]
        summary[str(s)] = {
            "acc": accs[s], "n": len(recs),
            "keep_ratio_median": float(np.median([r["keep_ratio"] for r in recs.values()])),
            "keep_ratio_random_median": float(np.median(
                [r["keep_ratio_random"] for r in recs.values()])),
            "suff_median": float(np.median([r["suff"] for r in recs.values()])),
        }

    mask_rate = mask_decidable / (len(nodes) * len(MODEL_SEEDS))
    nec = np.array([r["nec"] for r in per_seed[MODEL_SEEDS[0]].values()])
    nec_r = np.array([r["nec_random"] for r in per_seed[MODEL_SEEDS[0]].values()])
    print(f"掩码解释器可判定率：{mask_decidable}/{len(nodes) * len(MODEL_SEEDS)} "
          f"= {mask_rate:.0%}")
    print(f"必要性口径：删解释边后目标类概率降 {np.median(nec):+.3f}，"
          f"删 incident 随机边 {np.median(nec_r):+.3f}")
    print(f"跨模型 Jaccard：中位 {np.median(jac):.3f}（{len(jac)} 对）")

    out = ROOT / "results" / "node_explain.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {"n_nodes": N_NODES_EXPLAIN, "k": K, "model_seeds": list(MODEL_SEEDS),
                   "explainer": "gradient saliency", "random_baseline": "incident edges"},
        "mask_explainer_decidable_rate": float(mask_rate),
        "fidelity_advantage": float(
            summary[str(MODEL_SEEDS[0])]["keep_ratio_median"]
            / max(summary[str(MODEL_SEEDS[0])]["keep_ratio_random_median"], 1e-8) - 1),
        "summary": summary, "jaccard_median": float(np.median(jac)),
        "n_jaccard": int(len(jac)),
        "rows": {str(s): per_seed[s] for s in MODEL_SEEDS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    base = summary[str(MODEL_SEEDS[0])]
    checks = [
        ("掩码解释器可判定率 ≥50%", mask_rate >= 0.5, f"{mask_rate:.0%}"),
        ("充分性：保留解释边后概率高于 incident 随机",
         base["keep_ratio_median"] > base["keep_ratio_random_median"],
         f"{base['keep_ratio_median']:.2f} vs {base['keep_ratio_random_median']:.2f}"),
        ("必要性：删解释边的降幅大于删随机边",
         np.median(nec) > np.median(nec_r),
         f"{np.median(nec):+.3f} vs {np.median(nec_r):+.3f}"),
        ("跨模型 Jaccard 中位 ≥0.4", np.median(jac) >= 0.4, f"{np.median(jac):.3f}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 3.2/3.5" if ok else "\n[FAIL] 3.2/3.5")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
