"""3.3–3.4 验收：合成 SBM 上的**真值解释检验**（B 线的核心）。

真值怎么来：对每条候选边做消融——删掉它重算目标类概率，下降量 Δ 就是它的真实
重要性。这是直接测出来的，不依赖任何假设。

结果要点（详见 results/node_planted.json 与 DECISIONS.md D23）：
  · 解释器 top-k 的平均消融重要性约是随机边的 10 倍以上，p < 0.05
  · 但约四成节点的掩码退化成常数（说明这些节点上模型没用结构），
    主统计只算非退化节点，退化率单独报告
  · 过程中踩过两个坑，都记进了 DECISIONS：① 用"同社区边"当真值是基数率陷阱
    （SBM 里 93% 的边同社区，随机精度 0.94）；② 为提速把候选边抽样到 40 条，
    导致解释器选的边不在候选集里，把 11.8× 稀释成 1.45×

判据：
  · GCN test acc ≥ 0.88
  · 解释边的消融重要性显著高于随机（p < 0.05）且 ≥ 2 倍
  · Fid+ 中位 ≥ 0.5（解释至少保真）
  · 退化率 < 50%

产物: results/node_planted.json
用法: python exp/run_node_planted.py
"""

from __future__ import annotations

import json
import sys
from math import erfc, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cora  # noqa: E402
import explain_node  # noqa: E402
import node_gnn  # noqa: E402
import sbm  # noqa: E402

N_EXPLAIN = 40
K = 6
N_RAND = 20
SEED = 0


def mwu_p(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    ranks = np.argsort(np.argsort(np.concatenate([a, b]))) + 1.0
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    z = (u - n1 * n2 / 2) / max(np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12), 1e-9)
    return float(erfc(abs(z) / sqrt(2)))


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED + 7)
    g = sbm.make_sbm(seed=SEED)
    splits = sbm.split(g["n"], seed=SEED)
    feats = cora.normalize_features(g["features"])
    A, X = node_gnn.to_torch(cora.normalized_adj(g["adj"]), feats)
    y = torch.from_numpy(g["labels"])

    model = node_gnn.GCN(g["d"], 16, g["classes"], dropout=0.3)
    model, _, _ = node_gnn.train(model, A, X, y, splits, epochs=300, seed=SEED)
    model.eval()
    with torch.no_grad():
        logits = model(A, X)
    acc = {k: node_gnn.accuracy(logits, y, splits[f"{k}_idx"]) for k in ("train", "val", "test")}
    print(f"GCN on SBM: train {acc['train']:.3f} val {acc['val']:.3f} test {acc['test']:.3f}")

    rows, n_degen, n_try = [], 0, 0
    for node in splits["test_idx"][:N_EXPLAIN]:
        node = int(node)
        n_try += 1
        ex = explain_node.explain_node(model, feats, g["edges"], g["labels"],
                                       node, K, n_hops=2, seed=SEED)
        if not ex["edges"]:
            continue
        mask_std = float(np.std(ex["mask"]))
        cand = _candidate_edges(g["edges"], node, hops=2)
        idx = {int(c): i for i, c in enumerate(cand)}

        # 消融真值：删掉每条候选边后目标类概率下降量
        p_full = _prob(model, feats, g["edges"], node, None)
        cls = int(np.argmax(p_full))
        delta = np.array([p_full[cls] - _prob(model, feats, g["edges"], node,
                                              [e for e in cand if e != c])[cls]
                          for c in cand])

        top_edges = [e for e in ex["edges"] if int(e) in idx]
        if not top_edges:
            continue
        d_top = float(np.mean([delta[idx[int(e)]] for e in top_edges]))
        d_rand = float(np.mean([
            np.mean([delta[idx[int(c)]] for c in rng.choice(cand, size=K, replace=False)])
            for _ in range(N_RAND)]))
        _, _, fp, fm = explain_node.fidelity(model, feats, g["edges"], g["labels"],
                                             node, ex["edges"])
        if mask_std < 0.05:
            n_degen += 1
        rows.append({"node": node, "delta_top": d_top, "delta_random": d_rand,
                     "ratio": d_top / max(d_rand, 1e-12), "mask_std": mask_std,
                     "fid_plus": fp, "fid_minus": fm, "n_cand": int(len(cand))})

    allr = rows
    rows = [r for r in allr if r["mask_std"] >= 0.05]   # 退化的掩码排不出序，主统计只算非退化
    dt = np.array([r["delta_top"] for r in rows])
    dr = np.array([r["delta_random"] for r in rows])
    fid = np.array([r["fid_plus"] for r in rows])
    p = mwu_p(dt, dr)
    degen_rate = 1 - len(rows) / max(len(allr), 1)

    print(f"[非退化 {len(rows)} 个] 解释边平均消融Δ {dt.mean():.4f} vs 随机 {dr.mean():.4f}"
          f"（{dt.mean() / max(dr.mean(), 1e-12):.2f}×），MWU p={p:.4f}")
    print(f"逐节点比值中位 {np.median(dt / np.maximum(dr, 1e-12)):.2f}；"
          f"Fid+ 中位 {np.median(fid):.3f}")
    print(f"退化（mask std<0.05）{len(allr) - len(rows)}/{len(allr)} = {degen_rate:.0%}；"
          f"平均候选边 {np.mean([r['n_cand'] for r in allr]):.0f}")

    out = ROOT / "results" / "node_planted.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {"graph": "sbm", "n": g["n"], "k": K, "n_explain": len(rows),
                   "n_try": n_try, "seed": SEED, "n_rand": N_RAND,
                   "truth": "ablation delta（删掉该边后目标类概率下降量）"},
        "gcn": acc, "delta_top_mean": float(dt.mean()), "delta_random_mean": float(dr.mean()),
        "ratio": float(dt.mean() / max(dr.mean(), 1e-12)),
        "ratio_median": float(np.median(dt / np.maximum(dr, 1e-12))),
        "mwu_p": p, "fid_plus_median": float(np.median(fid)),
        "degenerate_rate": float(degen_rate), "n_clean": len(rows),
        "rows": allr}, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = [
        ("GCN test acc ≥ 0.88", acc["test"] >= 0.88, f"{acc['test']:.3f}"),
        ("解释边的消融重要性显著高于随机（p < 0.05）", p < 0.05, f"p={p:.4f}"),
        ("重要性至少是随机的 2 倍", dt.mean() >= 2 * dr.mean(),
         f"{dt.mean() / max(dr.mean(), 1e-12):.2f}×"),
        ("Fid+ 中位 ≥ 0.5", np.median(fid) >= 0.5, f"{np.median(fid):.3f}"),
        ("退化率 < 50%", degen_rate < 0.5, f"{degen_rate:.0%}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 3.3/3.4" if ok else "\n[FAIL] 3.3/3.4")
    return 0 if ok else 1


def _prob(model, feats, edges, node, keep):
    return explain_node._forward_probs(model, feats, edges, node, keep, 2).numpy()


def _candidate_edges(edges, node, hops=2):
    adj = {}
    for eid, (u, v) in enumerate(edges):
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    cur, seen = {node}, {node}
    for _ in range(hops):
        nxt = set()
        for u in cur:
            nxt |= set(adj.get(u, []))
        nxt -= seen
        seen |= nxt
        cur = nxt
    return np.array([e for e in range(len(edges))
                     if int(edges[e, 0]) in seen and int(edges[e, 1]) in seen])


if __name__ == "__main__":
    sys.exit(main())
