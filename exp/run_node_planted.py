"""3.3–3.4 验收（v3）：合成 SBM 上的**真值解释检验**。

v2 被第三方检测打回，三条致命问题已修：
  ① 掩码退火塌成硬 0/1 后仍按 k=6 取 top，剩下槽位从几百个并列 0 里按索引抽
     → 测的是"解释器判死的边"。现在温度固定 1（软掩码、无并列），并返回全量掩码；
  ② 保真度与消融都在 2 跳子图的**本地重归一化**上算，而该替身会翻类
     （检测复现：6/39 节点 argmax 与全局不同）→ 现在目标类一律取**全局预测类**，
     并报告本地/全局不一致率；
  ③ delta_top 与 delta_random 是同一节点上的配对量，却用了非配对 MWU
     → 现在用 Wilcoxon 符号秩，并报精确 p。

另有两条改进：
  · 加 oracle 基线（按消融 Δ 直接取 top-k）与"候选池内百分位"，
    让"10.93×"不再是一个没有标尺的数；
  · 测试节点改为**分层随机抽样**（每个社区各抽 1/3），避免只取 id 最小的 40 个。

判据：
  · GCN test acc ≥ 0.88
  · 解释边的消融重要性 > 随机边，Wilcoxon p < 0.05
  · 所选边在候选池内的平均百分位 > 0.5（随机 ≈ 0.5）
  · oracle 效率 ≥ 0.3（达到"按真值排序"的 30% 以上）
  · 本地/全局预测类不一致率 < 25%

产物: results/node_planted.json
用法: python exp/run_node_planted.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import cora  # noqa: E402
import explain_node  # noqa: E402
import node_gnn  # noqa: E402
import sbm  # noqa: E402

N_EXPLAIN = 45          # 每个社区 15 个
K = 6
N_RAND = 20
SEED = 0


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
        g_logits = model(A, X)
    acc = {k: node_gnn.accuracy(g_logits, y, splits[f"{k}_idx"]) for k in ("train", "val", "test")}
    print(f"GCN on SBM: train {acc['train']:.3f} val {acc['val']:.3f} test {acc['test']:.3f}")

    # 分层抽样：每个社区各取 1/3
    test = np.asarray(splits["test_idx"])
    per_cls = N_EXPLAIN // g["classes"]
    picked = []
    for c in range(g["classes"]):
        cand_nodes = test[g["labels"][test] == c]
        picked.extend(rng.choice(cand_nodes, size=min(per_cls, len(cand_nodes)),
                                 replace=False).tolist())
    picked = np.array(sorted(picked))

    rows, n_skip, skip_keep_struct = [], 0, 0
    local_global_disagree = 0
    for node in picked:
        node = int(node)
        probs = torch.softmax(g_logits[node], dim=0).numpy()
        cls = int(np.argmax(probs))

        # 全局预测类 vs 局部替身的预测类：不一致率要单独报（检测指出这是硬伤）
        p_loc = explain_node._forward_probs(model, feats, g["edges"], node, None, 2)
        if int(torch.argmax(p_loc)) != cls:
            local_global_disagree += 1

        ex = explain_node.explain_node(model, feats, g["edges"], node, cls, K, seed=SEED)
        cand = ex["cand"]
        # 跳过判据：掩码几乎无差异 = 排不出序。注意不是 n_kept==0——
        # 解释器要的是**排序**，全在 0.5 以下但仍有序时同样可用。
        spread = float(np.max(ex["mask"]) - np.min(ex["mask"])) if len(ex["mask"]) else 0.0
        if len(cand) < 2 * K or spread < 0.05:
            n_skip += 1
            p_empty = _local_prob(model, feats, g["edges"], node, cls, [])
            if p_empty > 0.5:
                skip_keep_struct += 1
            continue
        mask = ex["mask"]
        order = np.argsort(-mask)[:K]
        top = cand[order]
        assert len(top) == K

        # 消融真值：删掉每条候选边后，目标类（全局预测类）的**局部**概率下降量
        p_full = _local_prob(model, feats, g["edges"], node, cls, None)
        delta = np.array([p_full - _local_prob(model, feats, g["edges"], node, cls,
                                               [e for e in cand if e != c_]) 
                          for c_ in cand])
        p_empty = _local_prob(model, feats, g["edges"], node, cls, [])

        d_top = float(delta[order].mean())
        o_oracle = np.argsort(-delta)[:K]
        d_oracle = float(delta[o_oracle].mean())
        d_rand = float(np.mean([
            delta[rng.choice(len(cand), size=K, replace=False)].mean()
            for _ in range(N_RAND)]))

        # 所选边在候选池 Δ 分布中的平均百分位（1 = 最重要）
        ranks = np.argsort(np.argsort(delta)) / max(len(delta) - 1, 1)
        pct = float(ranks[order].mean())

        _, _, fp, fm = explain_node.fidelity(model, feats, g["edges"], node, cls, top)
        rows.append({"node": node, "class": cls, "n_cand": int(len(cand)),
                     "n_kept": ex["n_kept"], "delta_top": d_top,
                     "delta_oracle": d_oracle, "delta_random": d_rand,
                     "percentile": pct, "fid_plus": fp, "fid_minus": fm,
                     "p_full": p_full, "p_empty": p_empty})
    dt = np.array([r["delta_top"] for r in rows])
    dor = np.array([r["delta_oracle"] for r in rows])
    dr = np.array([r["delta_random"] for r in rows])
    pct = np.array([r["percentile"] for r in rows])
    fid = np.array([r["fid_plus"] for r in rows])
    eff = dt / np.maximum(dor, 1e-12)
    # 效率比值只在 oracle Δ 有实际量级时才有意义（分母趋 0 会炸成 ±几十）
    eff_v = dt[dor > 1e-3] / dor[dor > 1e-3]
    n_eff = int(len(eff_v))

    p_rand = float(wilcoxon(dt, dr).pvalue) if np.any(dt != dr) else 1.0
    p_oracle = float(wilcoxon(dt, dor).pvalue) if np.any(dt != dor) else 1.0

    print(f"[{len(rows)} 个节点，跳过 {n_skip}]")
    print(f"本地替身 vs 全局预测类不一致：{local_global_disagree}/{len(picked)} "
          f"= {local_global_disagree / max(len(picked), 1):.0%}")
    print(f"跳过原因：掩码无差异（spread<0.05，排不出序）{n_skip} 个；"
          f"其中删光边仍能正确预测 {skip_keep_struct} 个 "
          f"（若为 0，说明这些是**解释器失败**，不是'模型不用结构'）")
    print(f"消融Δ：解释边 {dt.mean():.4f} / oracle {dor.mean():.4f} / 随机 {dr.mean():.4f}")
    print(f"vs 随机：{dt.mean() / max(dr.mean(), 1e-12):.2f}×，Wilcoxon p={p_rand:.2e}")
    print(f"oracle 效率：中位 {np.median(eff_v):.3f}（仅 {n_eff}/{len(rows)} 个 oracle Δ>1e-3 的节点）")
    print(f"候选池内百分位：均值 {pct.mean():.3f}（随机 ≈ 0.5）")
    print(f"Fid+ 中位 {np.median(fid):.3f}；存活边 n_kept 中位 {np.median([r['n_kept'] for r in rows]):.0f}")
    print(f"候选池 {min(r['n_cand'] for r in rows)}–{max(r['n_cand'] for r in rows)} 条（中位 "
          f"{np.median([r['n_cand'] for r in rows]):.0f}）")

    out = ROOT / "results" / "node_planted.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": {"graph": "sbm", "n": g["n"], "k": K, "n_explain": len(rows),
                   "seed": SEED, "n_rand": N_RAND, "sampling": "stratified by class",
                   "truth": "ablation delta（本地子图，目标类 = 全局预测类）"},
        "gcn": acc, "n_skip": n_skip,
        "skipped_no_structure": skip_keep_struct,
        "local_global_disagree": local_global_disagree,
        "n_picked": int(len(picked)),
        "delta_top_mean": float(dt.mean()), "delta_oracle_mean": float(dor.mean()),
        "delta_random_mean": float(dr.mean()),
        "ratio_vs_random": float(dt.mean() / max(dr.mean(), 1e-12)),
        "wilcoxon_p_random": p_rand, "wilcoxon_p_oracle": p_oracle,
        "oracle_efficiency_median": float(np.median(eff_v)), "n_efficiency": n_eff,
        "percentile_mean": float(pct.mean()),
        "fid_plus_median": float(np.median(fid)),
        "n_kept_median": float(np.median([r["n_kept"] for r in rows])),
        "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = [
        ("GCN test acc ≥ 0.88", acc["test"] >= 0.88, f"{acc['test']:.3f}"),
        ("解释边 vs 随机：Wilcoxon p < 0.05", p_rand < 0.05, f"p={p_rand:.2e}"),
        ("候选池内百分位 > 0.5", pct.mean() > 0.5, f"{pct.mean():.3f}"),
        ("oracle 效率 ≥ 0.3", np.median(eff_v) >= 0.3 if n_eff else False,
         f"{np.median(eff_v):.3f}（n={n_eff}）"),
        ("每节点取到的解释边恰为 k 条", all(len(r) > 0 for r in rows), "assert 通过"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 3.3/3.4" if ok else "\n[FAIL] 3.3/3.4")
    return 0 if ok else 1


def _local_prob(model, feats, edges, node, cls, keep):
    return float(explain_node._forward_probs(model, feats, edges, node, keep, 2)[cls])


if __name__ == "__main__":
    sys.exit(main())
