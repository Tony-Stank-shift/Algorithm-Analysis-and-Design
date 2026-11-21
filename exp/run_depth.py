"""3.5 验收：深度下界的构造性验证。

构造两张图 G1、G2，它们在 t 的 L 跳**计算图**内完全同构，但 d(s,t) 不同：

    G1:  s - d - c - b - a - t      （d(s,t)=5）
    G2:  s -     c - b - a - t      （d(s,t)=4，把 s-d-c 换成 s-c）

对 L=2,3：两图在 t 的 L 跳计算图内同构，所以任何 L 层消息传递模型的输出
         **完全相同**，而真值差 1 → 必有一错。
对 L=4：G2 已经能算对（s 进入 4 跳内），G1 仍不行 → 两图开始可区分。
对 L=5：都算对。

关于"计算图"的定义（第一版这里错了，被第三方检测抓出）：
    L 层之后，根节点 t 的取值只依赖
      · 距离 ≤ L 的节点的**自身特征**
      · 满足 min(dist(u),dist(v)) ≤ L-1 的**边及其权重**
    第一版只对「球内诱导子图」的边做指纹、且不含节点特征，于是构造出了
    「指纹相同但源点不同」的反例（输出 1e5 vs 0）。现在把特征与端点一起纳入签名。
    另外黑盒臂原先用 deg/deg.max() 这个**全局**归一化，感受野其实超出 L 跳，
    已改成 log1p(deg) 的逐点量（见 gnn.collate）。

用法: python exp/run_depth.py
产物: results/depth_bound.json, results/fig/depth_bound.png
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import algorithms  # noqa: E402
import gnn  # noqa: E402
import viz  # noqa: E402


def build(edges, n, w=None):
    ei = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    w = np.ones(len(ei)) if w is None else np.asarray(w, dtype=float)
    adj = algorithms.build_adj(n, ei, w)
    d, _, _ = algorithms.dijkstra(n, adj, 0)
    return {"n": n, "ei": ei, "w": w, "s": 0, "t": n - 1, "d_true": d,
            "W": algorithms.dense_from_sparse(n, ei, w), "kind": "construct",
            "seed": 0, "meta": {"E": len(w), "t_hop": 0}}


# 0=s, 1=d, 2=c, 3=b, 4=a, 5=t
G1 = build([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 6)
G2 = build([(0, 2), (2, 3), (3, 4), (4, 5)], 6)


def adjacency(rec):
    adj = [[] for _ in range(rec["n"])]
    for e in range(len(rec["w"])):
        u, v = int(rec["ei"][e, 0]), int(rec["ei"][e, 1])
        adj[u].append((v, float(rec["w"][e])))
        adj[v].append((u, float(rec["w"][e])))
    return adj


def hop_signature(rec, root, L, source):
    """L 层计算图的规范化签名：节点特征 + 相关边（含权）。

    节点集 = 所有「相关边」的端点；相关边 = min(dist(u),dist(v)) <= L-1。
    """
    adj = adjacency(rec)
    dist = {root: 0}
    frontier = [root]
    for _ in range(L):
        nxt = []
        for u in frontier:
            for v, _w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt

    edges, nodes = [], {root}
    for u in range(rec["n"]):
        for v, w in adj[u]:
            if u < v and min(dist.get(u, 10**9), dist.get(v, 10**9)) <= L - 1:
                edges.append((u, v, round(w, 9)))
                nodes.add(u)
                nodes.add(v)

    feats = sorted((v, 1 if v == source else 0, len(adj[v])) for v in nodes)
    sig = f"{feats}|{sorted(edges)}"
    return hashlib.sha256(sig.encode()).hexdigest()[:16], sorted(nodes), sorted(edges)


def main() -> int:
    viz.setup()
    torch.manual_seed(0)
    rows = []

    def show(v):
        return "不可达" if v > 1e3 else f"{v:.3f}"

    print(f"{'L':>2s} {'签名G1':>14s} {'签名G2':>14s} {'同构':>4s} "
          f"{'pred(G1)':>10s} {'pred(G2)':>10s} {'真值':>7s}")
    for L in (1, 2, 3, 4, 5):
        h1, n1, _ = hop_signature(G1, 5, L, 0)
        h2, n2, _ = hop_signature(G2, 5, L, 0)
        same = h1 == h2
        m = gnn.MinSumMPNN(layers=L, identity=True)
        p1 = float(gnn.predict(m, G1, 0)[5])
        p2 = float(gnn.predict(m, G2, 0)[5])
        rows.append({"L": L, "same_ball": same, "nodes_g1": n1, "nodes_g2": n2,
                     "pred_g1": p1, "pred_g2": p2,
                     "true_g1": float(G1["d_true"][5]), "true_g2": float(G2["d_true"][5]),
                     "err_g1": abs(p1 - G1["d_true"][5]),
                     "err_g2": abs(p2 - G2["d_true"][5]),
                     "identical_pred": abs(p1 - p2) < 1e-9})
        print(f"{L:2d} {h1:>14s} {h2:>14s} {str(same):>4s} "
              f"{show(p1):>10s} {show(p2):>10s} {'(5, 4)':>7s}")

    out = ROOT / "results"
    (out / "fig").mkdir(parents=True, exist_ok=True)
    (out / "depth_bound.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    import matplotlib.pyplot as plt
    Ls = [r["L"] for r in rows]
    CAP = 7.0
    y1 = [r["pred_g1"] if r["pred_g1"] < 1e3 else CAP for r in rows]
    y2 = [r["pred_g2"] if r["pred_g2"] < 1e3 else CAP for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(Ls, y1, "o-", label="G1 的预测")
    ax.plot(Ls, y2, "s--", label="G2 的预测")
    ax.axhline(5, color="tab:blue", lw=1, ls=":", label="G1 真值 = 5")
    ax.axhline(4, color="tab:orange", lw=1, ls=":", label="G2 真值 = 4")
    ax.text(1.05, CAP + 0.15, "不可达（模型报哨兵值）", fontsize=8, color="gray")
    ax.set_xlabel("层数 L（= Bellman-Ford 轮数）")
    ax.set_ylabel("d(s,t)")
    ax.set_title("深度下界：L≤3 两图输出完全相同，必有一错")
    ax.set_xticks(Ls)
    ax.set_ylim(0, CAP + 0.8)
    ax.legend(fontsize=8)
    fig.savefig(out / "fig" / "depth_bound.png")
    plt.close(fig)

    r2 = [r for r in rows if r["L"] == 2][0]
    r4 = [r for r in rows if r["L"] == 4][0]
    r5 = [r for r in rows if r["L"] == 5][0]
    checks = [
        ("L=2：两图计算图签名相同", r2["same_ball"], r2["nodes_g1"]),
        ("L=2：预测逐位相同", r2["identical_pred"], f"{r2['pred_g1']}"),
        ("L=2：两图都错", r2["err_g1"] > 0.5 and r2["err_g2"] > 0.5,
         f"{r2['err_g1']:.0f} / {r2['err_g2']:.0f}"),
        ("L=4：可区分（G2 对、G1 仍错）", abs(r4["pred_g1"] - r4["pred_g2"]) > 0.5
         and r4["err_g2"] < 1e-6 and r4["err_g1"] > 0.5,
         f"G2 err={r4['err_g2']:.2e}, G1 err={r4['err_g1']:.0f}"),
        ("L=5：两图都算对", r5["err_g1"] < 1e-6 and r5["err_g2"] < 1e-6,
         f"{r5['pred_g1']:.3f} / {r5['pred_g2']:.3f}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 3.5" if ok else "\n[FAIL] 3.5")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
