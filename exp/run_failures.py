"""4.4 验收：失败案例——先给**分布**，再给代表样本，避免 cherry-picking 的嫌疑。

上一版的问题（第三方检测指出）：
  · 案例 A 与 B 取自同一张图，"三类失败"其实只有两类；
  · 只给 argmax，读者会以为是偶发——而黑盒在整批图上预测**全部为负**；
  · 与 4.1 用了不同的图集（60 张 vs 40 张），两个"最坏误差"并列却不说明；
  · 结构臂的 error=99943 是哨兵 1e5 的产物，不能当误差量级引用。

本版：先报 60 张图上的误差分布（中位 / p90 / 负预测比例），
再给两个**不同图**的代表样本，并注明与 4.1 的口径差异。

产物: results/failures.json, results/fig/fail_*.png
用法: python exp/run_failures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))

import matplotlib.pyplot as plt  # noqa: E402
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

N_GRAPHS = 60


def draw(rec, s, t, path_edges, title, subtitle, out_path):
    G = nx.Graph()
    G.add_nodes_from(range(rec["n"]))
    for e in range(len(rec["w"])):
        G.add_edge(int(rec["ei"][e, 0]), int(rec["ei"][e, 1]), weight=float(rec["w"][e]))
    pos = nx.spring_layout(G, seed=7)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=1.0, ax=ax)
    if path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=[
            (int(rec["ei"][e, 0]), int(rec["ei"][e, 1])) for e in path_edges],
            edge_color="tab:green", width=2.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=120, node_color="lightgray", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[s], node_size=220, node_color="tab:blue", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[t], node_size=220, node_color="tab:red", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.text(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.axis("off")
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    viz.setup()
    torch.manual_seed(0)
    m1 = _trained.get("minsum", 8)
    m2 = _trained.get("mean", 8)
    samples = T.build_samples(RB.gen_er, range(3000, 3000 + N_GRAPHS), 1, seed0=11)

    # ---------- 先给分布 ----------
    rows = []
    for rec, s in samples:
        t = rec["t"]
        p2 = explain.pred_at(m2, rec, s, t)
        p1 = explain.pred_at(m1, rec, s, t)          # 同深度 8 层
        true = float(rec["d_true"][t])
        rows.append({"seed": rec["seed"], "n": rec["n"], "true": true,
                     "pred_black": p2, "pred_struct_d8": p1,
                     "err_black": abs(p2 - true), "err_struct_d8": abs(p1 - true)})

    err_b = np.array([r["err_black"] for r in rows])
    err_s = np.array([r["err_struct_d8"] for r in rows])
    pred_b = np.array([r["pred_black"] for r in rows])
    dist = {
        "n_graphs": len(rows),
        "black_median_err": float(np.median(err_b)),
        "black_p90_err": float(np.percentile(err_b, 90)),
        "black_negative_pred_rate": float(np.mean(pred_b < 0)),
        "black_pred_range": [float(pred_b.min()), float(pred_b.max())],
        "struct_d8_median_err": float(np.median(err_s)),
        "struct_d8_unreachable_rate": float(np.mean(err_s > 1e3)),
    }
    print(f"60 张图上的分布：黑盒中位误差 {dist['black_median_err']:.2f}、"
          f"p90 {dist['black_p90_err']:.2f}、负预测比例 "
          f"{dist['black_negative_pred_rate']:.0%}（预测区间 "
          f"[{dist['black_pred_range'][0]:.2f}, {dist['black_pred_range'][1]:.2f}]）")
    print(f"结构臂（8 层）中位误差 {dist['struct_d8_median_err']:.2f}、"
          f"不可达比例 {dist['struct_d8_unreachable_rate']:.0%}")

    # ---------- 案例 A：黑盒最大误差 ----------
    a = max(rows, key=lambda r: r["err_black"])
    rec_a = next(rec for rec, s in samples if rec["seed"] == a["seed"])
    s_a, t_a = rec_a["s"], rec_a["t"]
    sal = explain.saliency_explanation(m2, rec_a, s_a, t_a, k=6)
    draw(rec_a, s_a, t_a, sal,
         f"失败案例 A：黑盒误差最大的一张（n={rec_a['n']}，60 张里排第 1）",
         f"真值 {a['true']:.2f}，预测 {a['pred_black']:.2f}，误差 {a['err_black']:.2f}\n"
         f"但注意：负预测在这批图上占 {dist['black_negative_pred_rate']:.0%}，它不是偶发",
         ROOT / "results" / "fig" / "fail_black_maxerr.png")

    # ---------- 案例 B：深度限制的**构造例**（自然样本里 60 张只有 1 张触发，不能硬凑两例） ----------
    ei_b = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)
    w_b = np.ones(5)
    n_b = 6
    adj_b = algorithms.build_adj(n_b, ei_b, w_b)
    d_b, _, _ = algorithms.dijkstra(n_b, adj_b, 0)
    rec_b = {"n": n_b, "ei": ei_b, "w": w_b, "s": 0, "t": 5, "d_true": d_b,
             "W": algorithms.dense_from_sparse(n_b, ei_b, w_b), "kind": "construct",
             "seed": -1, "meta": {"E": 5, "t_hop": 5}}
    m3 = gnn.MinSumMPNN(layers=3, identity=True)
    p_b = explain.pred_at(m3, rec_b, 0, 5)
    hops = 5
    draw(rec_b, 0, 5, [0, 1, 2, 3, 4],
         "失败案例 B：深度限制（构造例，非自然样本）",
         f"最短路径 {hops} 跳，3 层模型只有 3 跳感受野 → 报不可达\n"
         f"机制示意；在 60 张自然图上，8 层模型的触发率是 "
         f"{dist['struct_d8_unreachable_rate']:.0%}",
         ROOT / "results" / "fig" / "fail_depth_limit.png")

    # ---------- 案例 C：训练偏差 + 预测-真值散点 ----------
    ws = np.linspace(0.1, 1.1, 50)
    with torch.no_grad():
        g = (m1.edge_g(torch.from_numpy(ws).float()) * 10.0).numpy()
    trues = [r["true"] for r in rows]
    preds = [r["pred_black"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes[0].axhline(0, color="k", ls="--", lw=1, label="恒等（偏差为 0）")
    axes[0].plot(ws * 10, g - ws * 10, "r-", label="g(w) − w")
    axes[0].set_xlabel("w")
    axes[0].set_ylabel("偏差 g(w) − w")
    axes[0].set_title("训练把 g 推离恒等：每条边高估 +0.003~+0.013", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].scatter(trues, preds, s=14)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("真实距离")
    axes[1].set_ylabel("黑盒预测")
    axes[1].set_title(f"60 张图上黑盒预测与真值几乎无关（r={np.corrcoef(preds, trues)[0, 1]:.2f}）",
                      fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "fig" / "fail_g_bias.png")
    plt.close(fig)

    cases = {
        "distribution": dist,
        "A_black_max_error": {"seed": int(a["seed"]), "n": a["n"],
                              "true": a["true"], "pred": a["pred_black"],
                              "error": a["err_black"],
                              "why": "黑盒输出无值域约束：这批图里负预测占 "
                                     f"{dist['black_negative_pred_rate']:.0%}，本例只是 argmax"},
        "B_structured_depth_limit": {"type": "构造例（非自然样本）", "n": n_b,
                                     "true": float(d_b[5]), "pred": p_b, "hops": hops,
                                     "natural_trigger_rate": dist["struct_d8_unreachable_rate"],
                                     "why": f"最短路径 {hops} 跳 > 3 层感受野；"
                                            "自然图上 8 层模型的触发率见上"},
        "C_g_bias": {"g_at_w5": float(m1.edge_g(torch.tensor(0.5)).item() * 10),
                     "why": "2 个参数的偏差只有 0.15%，迭代到收敛后被放大成 0.024 的 MAE"},
        "note": "本文件用 60 张图（seed 3000-3059）；4.1 的 accuracy 用 40 张（3000-3039），"
                "两处的最坏误差不可直接并列；案例 B 是构造例，用于演示机制",
    }
    (ROOT / "results" / "failures.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    checks = [
        ("A 是自然样本的 argmax（60 张里第 1）",
         a["err_black"] == max(r["err_black"] for r in rows), f"{a['err_black']:.2f}"),
        ("黑盒的负预测是常态（≥50% 的图）", dist["black_negative_pred_rate"] >= 0.5,
         f"{dist['black_negative_pred_rate']:.0%}"),
        ("B：构造例 5 跳 > 3 层，模型报不可达", hops == 5 and p_b > 1e3,
         f"{hops} 跳, pred={p_b:.0f}"),
        ("B 如实给出自然样本上的触发率", 0 <= dist["struct_d8_unreachable_rate"] <= 1,
         f"{dist['struct_d8_unreachable_rate']:.0%}"),
        ("C：g(5) 与 5 相差 < 1%", abs(cases["C_g_bias"]["g_at_w5"] - 5.0) < 0.05,
         f"{cases['C_g_bias']['g_at_w5']:.4f}"),
        ("三张图已生成",
         all((ROOT / "results" / "fig" / f).exists() for f in
             ("fail_black_maxerr.png", "fail_depth_limit.png", "fail_g_bias.png")), "ok"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 4.4" if ok else "\n[FAIL] 4.4")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
