"""4.2 验收：鲁棒性——权重噪声 / 删边 / 规模外推 下三个臂的退化。

口径（三个臂都在**同一批扰动图**上评测）：
  L0 未训练恒等 min-sum（= Bellman-Ford）
  L1 训练后的 min-sum（2 参数）
  L2 mean-GNN（黑盒）
白盒的"误差"永远为 0（它就是在扰动图上重新算），所以真正要看的是：
  ① 任务本身变了多少：S = mean |d_new - d_old| / d_old（三个臂共享）
  ② 模型在扰动后的误差：MAE vs 新的真值
  ③ 相对退化：(MAE_pert - MAE_clean) / MAE_clean

产物: results/robust.json, results/fig/robust.png
用法: python exp/run_robust.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import algorithms  # noqa: E402
import explain  # noqa: E402
import gnn  # noqa: E402
import run_black as RB  # noqa: E402
import train as T  # noqa: E402
import viz  # noqa: E402
import _trained  # noqa: E402

N_GRAPHS = 30
NOISE = 0.20
DROP = 0.10


def true_dist(rec):
    adj = algorithms.build_adj(rec["n"], rec["ei"], rec["w"])
    return algorithms.dijkstra(rec["n"], adj, rec["s"])[0]


def add_noise(rec, rel, rng):
    out = dict(rec)
    out["w"] = np.maximum(rec["w"] * (1 + rng.uniform(-rel, rel, len(rec["w"]))), 1e-3)
    out["d_true"] = true_dist(out)
    out["meta"] = dict(rec.get("meta", {}))
    return out


def drop_edges(rec, p, rng):
    """删 p 比例的边；若断开则把删掉的边逐条加回直到连通。"""
    E = len(rec["w"])
    keep = rng.random(E) >= p
    for _ in range(E):
        ei, w = rec["ei"][keep], rec["w"][keep]
        G = nx.Graph()
        G.add_nodes_from(range(rec["n"]))
        for e in range(len(w)):
            G.add_edge(int(ei[e, 0]), int(ei[e, 1]))
        if nx.is_connected(G) or keep.all():
            break
        idx = np.where(~keep)[0]
        keep[int(idx[rng.integers(0, len(idx))])] = True
    out = dict(rec)
    out["ei"], out["w"] = rec["ei"][keep], rec["w"][keep]
    out["d_true"] = true_dist(out)
    out["meta"] = dict(rec.get("meta", {}), E=int(keep.sum()), dropped=float(1 - keep.mean()))
    return out


def mae(model, samples, layers=None):
    errs = []
    for rec, s in samples:
        p = explain.pred_at(model, rec, s, rec["t"], layers=layers)
        errs.append(abs(p - float(rec["d_true"][rec["t"]])))
    return float(np.mean(errs)), float(np.max(errs))


def main() -> int:
    viz.setup()
    torch.manual_seed(0)
    rng = np.random.default_rng(20261121)

    clean = T.build_samples(RB.gen_er, range(3000, 3000 + N_GRAPHS), 1, seed0=11)
    noisy, dropped = [], []
    sens = {"noise": [], "drop": []}
    for rec, s in clean:
        n_rec = add_noise(rec, NOISE, rng)
        d_rec = drop_edges(rec, DROP, rng)
        noisy.append((n_rec, s))
        dropped.append((d_rec, s))
        old, new1, new2 = rec["d_true"], n_rec["d_true"], d_rec["d_true"]
        sens["noise"].append(float(np.mean(np.abs(new1 - old) / np.maximum(old, 1e-9))))
        sens["drop"].append(float(np.mean(np.abs(new2 - old) / np.maximum(old, 1e-9))))

    arms = {"L0_identity": (_trained.identity(), 128),
            "L1_trained_minsum": (_trained.get("minsum", 8), 128),
            "L2_mean_gnn": (_trained.get("mean", 8), None)}

    rows = []
    print(f"{'arm':20s} {'clean MAE':>10s} {'noise MAE':>10s} {'drop MAE':>10s} {'退化(noise)':>12s}")
    for name, (model, layers) in arms.items():
        c, _ = mae(model, clean, layers)
        n1, n1max = mae(model, noisy, layers)
        d1, d1max = mae(model, dropped, layers)
        deg = (n1 - c) / c if c > 1e-3 else None
        rows.append({"arm": name, "clean": c, "noise": n1, "drop": d1,
                     "noise_max": n1max, "drop_max": d1max, "degradation_noise": deg})
        print(f"{name:20s} {c:10.3f} {n1:10.3f} {d1:10.3f} "
              f"{('N/A' if deg is None else f'{deg:.2f}'):>12s}")

    meta = {"n_graphs": N_GRAPHS, "noise_rel": NOISE, "drop_ratio": DROP,
            "task_sensitivity_noise": float(np.mean(sens["noise"])),
            "task_sensitivity_drop": float(np.mean(sens["drop"])),
            "arms": rows}

    out = ROOT / "results"
    (out / "robust.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    names = [r["arm"] for r in rows]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["clean"] for r in rows], 0.4, label="干净")
    ax.bar(x + 0.2, [r["noise"] for r in rows], 0.4, label="权重噪声 20%")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("MAE")
    ax.set_title("三种臂在权重噪声下的退化")
    ax.legend(fontsize=8)
    fig.savefig(out / "fig" / "robust.png")
    plt.close(fig)

    print(f"\n任务本身的变化量：噪声 {meta['task_sensitivity_noise']:.3f}，删边 "
          f"{meta['task_sensitivity_drop']:.3f}（相对距离）")

    # 规模外推：直接引用阶段 3 的结果（同生成器、同平均度，只放大 n）
    nsh = None
    bj = out / "black.json"
    if bj.exists():
        data = json.loads(bj.read_text(encoding="utf-8"))
        agg = data["aggregate"]
        nsh = {k: {"id_er": agg[f"{k}"]["id_er"]["mae"], "nshift_200": agg[f"{k}"]["nshift_200"]["mae"]}
               for k in ("mean-d8", f"minsum-d8-x128") if f"{k}" in agg}
        nsh["identity-x128"] = {"id_er": agg["identity-x128"]["id_er"]["mae"],
                                "nshift_200": agg["identity-x128"]["nshift_200"]["mae"]}
        print("规模外推（n=50 -> n=200，不重训）：")
        for k, v in nsh.items():
            print(f"  {k:18s} {v['id_er']:.4f} -> {v['nshift_200']:.4f}")

    checks = [
        ("L0 在噪声后误差仍 ≈ 0", rows[0]["noise"] < 1e-3, f"{rows[0]['noise']:.2e}"),
        ("L1 误差比 L2 小 10 倍以上", rows[1]["noise"] * 10 < rows[2]["noise"],
         f"{rows[1]['noise']:.3f} vs {rows[2]['noise']:.3f}"),
        ("扰动确实改变了任务（非常数）", meta["task_sensitivity_noise"] > 0.01,
         f"{meta['task_sensitivity_noise']:.3f}"),
        ("规模外推有明确数字", nsh is not None, "见上表"),
    ]
    # 观察（不作为判据）：黑盒臂的误差几乎不随扰动变化——
    # 这不是"稳健"，而是它本来就没在算距离（误差 22.9 已接近常数预测的水平）。
    print(f"\n观察：L2 对扰动的敏感度 |Δ|={abs(rows[2]['noise'] - rows[2]['clean']):.3f}"
          f"（相对 {abs(rows[2]['noise'] - rows[2]['clean']) / rows[2]['clean']:.1%}）"
          "——误差本来就大，扰动改变不了它，不能记为\"稳健\"。")
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 4.2" if ok else "\n[FAIL] 4.2")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
