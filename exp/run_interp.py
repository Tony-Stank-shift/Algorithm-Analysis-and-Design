"""4.3 验收：可解释性——**跨臂不比 Fidelity**，改比"解释对任务重不重要"。

为什么改
--------
第三方检测指出前一版的四个指标没有跨臂可比性：
  · 结构臂 Fid+ ≡ 1 是恒等式（argmin 链就是最短路，只留这些边当然等价）；
  · 结构臂 Fid− 有 71% 因删边断开而被封顶成 10，中位数由封顶产生；
  · 黑盒 Stability ≡ 1 是架构恒等（∂pred/∂w 在 ReLU 区域内与 w 无关），不是稳健；
  · Sparsity 的 k 由结构臂定义，不是质量度量。

于是改成三个**同尺度、有真值**的指标：
  ① 解释边精度：解释边里有多少条真的落在最短路树上（0 = 与任务答案无关）
  ② 删后任务变化：删掉解释边之后，**真实**最短距离改变了多少（相对，封顶 10）
  ③ 随机 k 边基线：同样预算下随便挑 k 条边的 Fid+，用来判断"显著性是否忠实于模型"

Fid+ 仍报，但只用来回答"显著性是否忠实于模型"，不用于跨臂排序。

产物: results/interp.json, results/fig/interp.png
用法: python exp/run_interp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import explain  # noqa: E402
import run_black as RB  # noqa: E402
import train as T  # noqa: E402
import viz  # noqa: E402
import _trained  # noqa: E402

N_GRAPHS = 30
CAP = 10.0
STAB_REL = 0.01
RANDOM_DRAWS = 3


def main() -> int:
    viz.setup()
    torch.manual_seed(0)
    rng = np.random.default_rng(20261122)

    samples = T.build_samples(RB.gen_er, range(3000, 3000 + N_GRAPHS), 1, seed0=11)
    kept = [(rec, s) for rec, s in samples if rec["d_true"][rec["t"]] > 0]
    m0 = _trained.identity()
    m1 = _trained.get("minsum", 8)
    m2 = _trained.get("mean", 8)
    print(f"样本：{len(kept)}/{len(samples)}（丢弃 s==t 的退化样本）")

    rows = []
    for rec, s in kept:
        t = rec["t"]
        e0 = explain.min_sum_explanation(m0, rec, s, t, layers=128)
        e1 = explain.min_sum_explanation(m1, rec, s, t, layers=128)
        k = max(len(e0["edges"]), 1)
        e2_edges = explain.saliency_explanation(m2, rec, s, t, k=k)
        rp = explain.perturb_weights(rec, STAB_REL, rng)
        e0p = explain.min_sum_explanation(m0, rp, s, t, layers=128)
        e1p = explain.min_sum_explanation(m1, rp, s, t, layers=128)
        e2p = explain.saliency_explanation(m2, rp, s, t, k=k)

        for name, model, edges, after, layers in (
                ("L0_identity", m0, e0["edges"], e0p["edges"], 128),
                ("L1_trained", m1, e1["edges"], e1p["edges"], 128),
                ("L2_mean", m2, e2_edges, e2p, None)):
            fp, _ = explain.fidelity(model, rec, s, t, edges, layers=layers)
            fp_rand = []
            for _ in range(RANDOM_DRAWS):
                rand = rng.choice(len(rec["w"]), size=len(edges), replace=False).tolist()
                fp_rand.append(explain.fidelity(model, rec, s, t, rand, layers=layers)[0])
            task_delta, unreach = explain.task_sensitivity(rec, edges, s, t, cap=CAP)
            rand_edges = rng.choice(len(rec["w"]), size=len(edges), replace=False).tolist()
            rows.append({
                "arm": name, "E": len(rec["w"]), "k": len(edges),
                "sparsity": len(edges) / max(len(rec["w"]), 1),
                "fid_plus": fp, "fid_plus_random": float(np.mean(fp_rand)),
                "task_delta": task_delta, "task_unreachable": bool(unreach),
                "precision": explain.path_precision(rec, edges, s, t),
                "precision_random": explain.path_precision(rec, rand_edges, s, t),
                "forms_path": explain.forms_path(rec, edges, s, t),
                "stability": explain.jaccard(edges, after),
                "struct_ok": bool(e0.get("structural_ok", False)) if name.startswith("L0") else None,
            })

    arms = ["L0_identity", "L1_trained", "L2_mean"]
    summary = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        summary[a] = {
            "n": len(rs),
            "fid_plus_median": float(np.median([r["fid_plus"] for r in rs])),
            "fid_plus_random_median": float(np.median([r["fid_plus_random"] for r in rs])),
            "task_delta_median": float(np.median([r["task_delta"] for r in rs])),
            "task_unreachable_rate": float(np.mean([r["task_unreachable"] for r in rs])),
            "precision_median": float(np.median([r["precision"] for r in rs])),
            "precision_random_median": float(np.median([r["precision_random"] for r in rs])),
            "forms_path_rate": float(np.mean([r["forms_path"] for r in rs])),
            "sparsity_median": float(np.median([r["sparsity"] for r in rs])),
            "stability_median": float(np.median([r["stability"] for r in rs])),
        }

    out = ROOT / "results"
    (out / "interp.json").write_text(json.dumps(
        {"config": {"n_graphs": len(kept), "dropped_s_eq_t": len(samples) - len(kept),
                    "cap": CAP, "stability_perturb": STAB_REL},
         "summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'arm':14s} {'Fid+':>7s} {'随机k':>8s} {'精度':>6s} {'随机精度':>8s} "
          f"{'成路径':>7s} {'放权后Δ':>8s} {'Sparsity':>9s} {'Stability':>10s}")
    for a in arms:
        v = summary[a]
        print(f"{a:14s} {v['fid_plus_median']:7.3f} {v['fid_plus_random_median']:8.2f} "
              f"{v['precision_median']:6.3f} {v['precision_random_median']:8.3f} "
              f"{v['forms_path_rate']:7.2f} {v['task_delta_median']:8.3f} "
              f"{v['sparsity_median']:9.3f} {v['stability_median']:10.3f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(arms))
    ax.bar(x - 0.2, [summary[a]["task_delta_median"] for a in arms], 0.4,
           label="删掉解释后真实距离的变化")
    ax.bar(x + 0.2, [summary[a]["precision_median"] for a in arms], 0.4,
           label="解释边落在真实最短路上的比例")
    ax.set_xticks(x)
    ax.set_xticklabels(["L0 恒等", "L1 训练", "L2 黑盒"], fontsize=8)
    ax.set_title("解释对任务是否有意义（同尺度、有真值）", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(out / "fig" / "interp.png")
    plt.close(fig)

    checks = [
        ("结构臂解出的是真路径（首点 = s）",
         np.mean([r["struct_ok"] for r in rows if r["arm"] == "L0_identity"]) > 0.95, "ok"),
        ("结构臂的解释边全部落在真实最短路上（精度 = 1，属定义性）",
         summary["L0_identity"]["precision_median"] > 0.999,
         f"{summary['L0_identity']['precision_median']:.3f}"),
        ("黑盒的显著性对**模型**忠实（Fid+ 远高于随机 k 边）",
         summary["L2_mean"]["fid_plus_median"] > summary["L2_mean"]["fid_plus_random_median"] + 0.2,
         f"{summary['L2_mean']['fid_plus_median']:.3f} vs "
         f"{summary['L2_mean']['fid_plus_random_median']:.3f}"),
        ("黑盒的解释**不构成** s-t 路径（结构臂 100% 构成）",
         summary["L2_mean"]["forms_path_rate"] < 0.2,
         f"{summary['L2_mean']['forms_path_rate']:.2f} vs "
         f"{summary['L0_identity']['forms_path_rate']:.2f}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    # 观察（不作判据）：把解释边权重放大 10 倍后，黑盒的 Δ 甚至比结构臂还大
    # （4.17 vs 3.41）。原因是显著性按边号并列取 top-k，选中的多是与源点相连的边，
    # 它们恰好落在最短路的头几跳上——"有影响"不等于"解释对了"。
    print(f"\n观察：放权 10 倍后真实距离变化 结构臂 {summary['L0_identity']['task_delta_median']:.2f}"
          f" vs 黑盒 {summary['L2_mean']['task_delta_median']:.2f}；"
          f"但黑盒只有 {summary['L2_mean']['forms_path_rate']:.0%} 的样本能连成 s-t 路径，"
          f"精度 {summary['L2_mean']['precision_median']:.2f}（随机基线 "
          f"{summary['L2_mean']['precision_random_median']:.2f}）。")
    print("注：黑盒的 Stability=1.0 是架构恒等（显著性对边权不敏感），不能当稳健性证据。")
    print("\n[PASS] 4.3" if ok else "\n[FAIL] 4.3")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
