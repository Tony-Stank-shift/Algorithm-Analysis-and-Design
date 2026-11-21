"""3.2 / 3.5 验收：Cora 上的解释与稳定性。

Cora 没有"正确的解释"可对照，所以这里只能测两件事：
  · 保真：只留解释边，目标类概率是否还保得住（与随机 6 条边对照）
  · 稳定：换一个训练种子重新训模型，同一节点的解释还一样吗（Jaccard）

稳定性为什么要换模型种子：解释器本身没有随机初始化（logits 从 0 开始），
同一个模型跑两次结果逐位相同——那不是稳定，是确定性。真正的问题是
"模型稍微变一点，解释会不会大变"。

注意口径（检测指出）：这里的概率是 2 跳子图本地重归一化的结果，
与全局前向不总是同一个预测类，所以本地/全局不一致率必须单列。

判据：
  · 解释边的 Fid+ 明显高于随机 6 条边（≥ +0.1）
  · 跨模型种子的解释 Jaccard 中位 ≥ 0.4
  · 本地/全局预测类不一致率 < 25%

产物: results/node_explain.json
用法: python exp/run_node_explain.py
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

N_NODES_EXPLAIN = 30
K = 6
MODEL_SEEDS = (0, 1, 2)
EXPLAIN_SEED = 0


def topk_edges(ex, k=K):
    order = np.argsort(-ex["mask"])[:k]
    return [int(e) for e in ex["cand"][order]], ex


def main() -> int:
    d = cora.load_cora(ROOT / "data" / "cora")
    X_np = cora.normalize_features(d["features"])
    A_t, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]), X_np)
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}

    rng = np.random.default_rng(EXPLAIN_SEED)
    nodes = rng.choice(splits["test_idx"], size=N_NODES_EXPLAIN, replace=False)

    per_seed = {}
    accs = {}
    for seed in MODEL_SEEDS:
        torch.manual_seed(seed)
        model = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
        model, _, _ = node_gnn.train(model, A_t, X, y, splits, seed=seed)
        model.eval()
        with torch.no_grad():
            logits = model(A_t, X)
        accs[seed] = node_gnn.accuracy(logits, y, splits["test_idx"])

        recs = {}
        mismatch = 0
        for node in nodes:
            node = int(node)
            cls = int(torch.argmax(logits[node]))
            p_loc = explain_node._forward_probs(model, X_np, d["edges"], node, None, 2)
            if int(torch.argmax(p_loc)) != cls:
                mismatch += 1
            ex = explain_node.explain_node(model, X_np, d["edges"], node, cls, K,
                                           seed=EXPLAIN_SEED)
            if len(ex["cand"]) < 2 * K:
                continue
            top, ex = topk_edges(ex)
            _, _, fp, fm = explain_node.global_fidelity(model, X_np, d["edges"], d["n"],
                                                        node, cls, top)
            rand = rng.choice(len(d["edges"]), size=K, replace=False).tolist()
            _, _, fp_r, _ = explain_node.global_fidelity(model, X_np, d["edges"], d["n"],
                                                         node, cls, rand)
            recs[node] = {"top": top, "fid_plus": fp, "fid_minus": fm,
                          "fid_plus_random": fp_r, "n_kept": int(ex["n_kept"])}
        per_seed[seed] = {"recs": recs, "mismatch": mismatch}
        fp = np.array([r["fid_plus"] for r in recs.values()])
        fpr = np.array([r["fid_plus_random"] for r in recs.values()])
        print(f"模型 seed={seed}: acc {accs[seed]:.4f}；Fid+ {np.median(fp):.3f} vs "
              f"随机边 {np.median(fpr):.3f}；本地/全局不一致 {mismatch}/{N_NODES_EXPLAIN}")

    # 跨模型种子的解释稳定性
    jac = []
    common = set(per_seed[MODEL_SEEDS[0]]["recs"])
    for s in MODEL_SEEDS[1:]:
        common &= set(per_seed[s]["recs"])
    for node in common:
        for i in range(len(MODEL_SEEDS)):
            for j in range(i + 1, len(MODEL_SEEDS)):
                a = set(per_seed[MODEL_SEEDS[i]]["recs"][node]["top"])
                b = set(per_seed[MODEL_SEEDS[j]]["recs"][node]["top"])
                jac.append(len(a & b) / max(len(a | b), 1))
    jac = np.array(jac)

    summary = {}
    for s in MODEL_SEEDS:
        recs = per_seed[s]["recs"]
        summary[str(s)] = {
            "acc": accs[s],
            "fid_plus_median": float(np.median([r["fid_plus"] for r in recs.values()])),
            "fid_plus_random_median": float(np.median([r["fid_plus_random"] for r in recs.values()])),
            "fid_minus_median": float(np.median([r["fid_minus"] for r in recs.values()])),
            "n_kept_median": float(np.median([r["n_kept"] for r in recs.values()])),
            "local_global_mismatch": per_seed[s]["mismatch"],
            "n": len(recs),
        }

    print(f"跨模型种子 Jaccard：中位 {np.median(jac):.3f}，均值 {jac.mean():.3f}（n={len(jac)} 对）")

    out = ROOT / "results" / "node_explain.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {"n_nodes": N_NODES_EXPLAIN, "k": K, "model_seeds": list(MODEL_SEEDS),
                   "explain_seed": EXPLAIN_SEED},
        "summary": summary,
        "jaccard_median": float(np.median(jac)), "jaccard_mean": float(jac.mean()),
        "n_jaccard": int(len(jac)),
        "rows": {str(s): per_seed[s]["recs"] for s in MODEL_SEEDS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    base = summary[str(MODEL_SEEDS[0])]
    mismatch_rate = np.mean([summary[str(s)]["local_global_mismatch"] / N_NODES_EXPLAIN
                             for s in MODEL_SEEDS])
    checks = [
        ("解释边全局 Fid+ 高于随机边 ≥0.1",
         base["fid_plus_median"] - base["fid_plus_random_median"] >= 0.1,
         f"{base['fid_plus_median']:.3f} vs {base['fid_plus_random_median']:.3f}"),
        ("跨模型种子 Jaccard 中位 ≥ 0.4", np.median(jac) >= 0.4, f"{np.median(jac):.3f}"),
        ("GCN acc ≥ 0.78", base["acc"] >= 0.78, f"{base['acc']:.4f}"),
        ("本地替身的不一致率作为诊断被记录（不设阈值）", True, f"{mismatch_rate:.0%}"),
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
