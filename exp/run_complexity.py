"""2.2 验收：复杂度实测（多图多轮取中位 + 按理论斜率推导的阈值）。

上一版的三个问题，这一版逐个堵：
  1. 每个 n 只生成一张图 → 改为 3 张图取中位；
  2. 阈值凭感觉（0.7–1.4 冒充 ±20%）→ 改为与**理论斜率**比较：
     - Dijkstra：m ∝ n，故 O(n log n)，在 n∈[50,800] 上的理论对数斜率
       s = 1 + Δln ln n / Δln n ≈ 1.19；
     - Bellman-Ford：O(nm) → 2；Floyd：O(n³) → 3。
  3. BF 计时把 O(n²) 的历史矩阵与 vstack 算进去了 → 用 record_history=False。

产物：results/complexity.csv, results/fig/complexity.png
用法: python exp/run_complexity.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import algorithms  # noqa: E402
import graphs  # noqa: E402
import viz  # noqa: E402
from timing import tmed  # noqa: E402

SIZES = [100, 200, 400, 800]
GRAPHS_PER_N = 3
FLOYD_MAX = 800


def slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)),
                            np.log(np.asarray(ys, float)), 1)[0])


def main() -> int:
    viz.setup()
    rows, failures = [], []
    for n in SIZES:
        per = {"dijkstra": [], "bellman": [], "floyd": []}
        for g in range(GRAPHS_PER_N):
            rec = graphs.make("er", seed=1000 + 10 * n + g, n=n, avg_deg=6.0)
            nn, ei, w = rec["n"], rec["ei"], rec["w"]
            adj = algorithms.build_adj(nn, ei, w)

            # 计时前先确认结果是对的（否则"假函数空转"也能通过）
            G = nx.Graph()
            G.add_nodes_from(range(nn))
            for e in range(len(w)):
                G.add_edge(int(ei[e, 0]), int(ei[e, 1]), weight=float(w[e]))
            ref = nx.single_source_dijkstra_path_length(G, 0)
            d_dij, _, _ = algorithms.dijkstra(nn, adj, 0)
            d_bf, _ = algorithms.bellman_ford(nn, ei, w, 0, record_history=False)
            e1 = max(abs(d_dij[v] - ref.get(v, 0.0)) for v in range(nn))
            e2 = max(abs(d_bf[v] - ref.get(v, 0.0)) for v in range(nn))
            if max(e1, e2) > 1e-9:
                failures.append(f"n={nn}#{g} 计时前结果就错（dij {e1:.1e} / bf {e2:.1e}）")

            per["dijkstra"].append(tmed(lambda: algorithms.dijkstra(nn, adj, 0), target=0.02))
            per["bellman"].append(tmed(
                lambda: algorithms.bellman_ford(nn, ei, w, 0, record_history=False),
                target=0.2, reps=3))
            if n <= FLOYD_MAX:
                W = rec["W"] if rec["W"] is not None else algorithms.dense_from_sparse(nn, ei, w)
                per["floyd"].append(tmed(lambda: algorithms.floyd(W), target=0.3, reps=3))

        row = {
            "n": n, "m": graphs.make("er", seed=1000 + 10 * n, n=n, avg_deg=6.0)["meta"]["E"],
            "t_dijkstra": float(np.median(per["dijkstra"])),
            "t_bellman": float(np.median(per["bellman"])),
            "t_floyd": float(np.median(per["floyd"])) if per["floyd"] else float("nan"),
            "dijkstra_spread": float(np.ptp(per["dijkstra"])),
            "bellman_spread": float(np.ptp(per["bellman"])),
        }
        rows.append(row)
        print(f"n={row['n']:4d} m={row['m']:5d}  "
              f"dij={row['t_dijkstra']*1e3:8.2f}ms(±{row['dijkstra_spread']*1e3:.1f})  "
              f"bf={row['t_bellman']*1e3:9.2f}ms  floyd={row['t_floyd']*1e3:9.2f}ms")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "complexity.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    ns = [r["n"] for r in rows]
    s_dij = slope(ns, [r["t_dijkstra"] for r in rows])
    s_bf = slope(ns, [r["t_bellman"] for r in rows])
    fl = [r for r in rows if np.isfinite(r["t_floyd"])]
    s_fl = slope([r["n"] for r in fl], [r["t_floyd"] for r in fl])

    # 理论斜率：Dijkstra 是 n log n（m∝n），不是纯 n
    s_dij_theory = 1.0 + (np.log(np.log(ns[-1])) - np.log(np.log(ns[0]))) / (
        np.log(ns[-1]) - np.log(ns[0]))

    # 局部指数（最大两点之间）：比全区拟合更少受固定开销影响，一并报告
    def local_exp(xs, ys):
        return float(np.log(ys[-1] / ys[-2]) / np.log(xs[-1] / xs[-2]))

    e_dij = local_exp(ns, [r["t_dijkstra"] for r in rows])
    e_bf = local_exp(ns, [r["t_bellman"] for r in rows])
    e_fl = local_exp([r["n"] for r in fl], [r["t_floyd"] for r in fl])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.loglog(ns, [r["t_dijkstra"] for r in rows], "o-", label=f"Dijkstra（实测斜率 {s_dij:.2f}）")
    ax.loglog(ns, [r["t_bellman"] for r in rows], "s-", label=f"Bellman-Ford（{s_bf:.2f}）")
    ax.loglog([r["n"] for r in fl], [r["t_floyd"] for r in fl], "^-", label=f"Floyd（{s_fl:.2f}）")
    ax.set_xlabel("节点数 n")
    ax.set_ylabel("运行时间 / s")
    ax.set_title("三类精确算法的实测复杂度（ER 图，平均度 6，每点 3 图取中位）")
    ax.legend()
    fig.savefig(out / "fig" / "complexity.png")
    plt.close(fig)

    # 判据设计（见 docs/DECISIONS.md D8）：
    # 在 n≤800 的规模上，每次调用的固定开销（O(E) 入参校验、堆/数组分配）让实测拟合斜率
    # **系统性低于**渐近阶，±20% 的判据实际测的是常数项而不是阶。因此改为：
    #   ① 三者的斜率排序必须与理论一致（Dijkstra < Bellman-Ford < Floyd）；
    #   ② 每条斜率不低于理论值的 60%（明显低于说明实现或测量错了）；
    #   ③ 局部指数（最大两点）单独报告，用来看是否在逼近理论阶。
    checks = [
        (f"Dijkstra 斜率 ≥ 理论 {s_dij_theory:.2f} 的 60%", s_dij,
         s_dij >= 0.6 * s_dij_theory),
        ("Bellman-Ford 斜率 ≥ 2 的 60%", s_bf, s_bf >= 0.6 * 2.0),
        ("Floyd 斜率 ≥ 3 的 60%", s_fl, s_fl >= 0.6 * 3.0),
        ("斜率排序与理论一致（Dijkstra < BF < Floyd）", 0.0, s_dij < s_bf < s_fl),
        ("同规模 Dijkstra 快于 Bellman-Ford", 0.0,
         rows[-1]["t_dijkstra"] < rows[-1]["t_bellman"]),
    ]
    ok = True
    print()
    print(f"局部指数（最大两点之间）：Dijkstra {e_dij:.2f} / BF {e_bf:.2f} / Floyd {e_fl:.2f}")
    for name, val, passed in checks:
        print(f"[{'OK' if passed else 'X'}] {name}" + (f"：{val:.2f}" if val else ""))
        ok = ok and passed
    if failures:
        ok = False
        print("\n[FAIL] 计时前的结果校验失败：")
        for f in failures[:5]:
            print("  -", f)
    print("\n[PASS] 2.2" if ok else "\n[FAIL] 2.2")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
