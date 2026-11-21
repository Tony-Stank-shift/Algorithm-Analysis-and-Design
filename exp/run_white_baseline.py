"""2.4 验收：白盒基线结果表——**三个算法都要**给出与外部真值的误差。

早先版本只校验 Dijkstra 的误差，BF/Floyd 只计时、结果丢弃，
相当于"白盒基线表"名不副实（把 BF 换成恒返回 0 的桩也能 PASS）。

产物：results/white.json
用法: python exp/run_white_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import algorithms  # noqa: E402
import graphs  # noqa: E402
from timing import tmed  # noqa: E402

# 规模取整：网格必须是 r*c，所以 5x8=40 / 10x10=100 / 10x20=200
KINDS = ["er", "ba", "grid", "sbm"]
SIZES = [40, 100, 200]
GRID = {40: (5, 8), 100: (10, 10), 200: (10, 20)}


def gen(kind: str, n: int):
    if kind == "er":
        return graphs.make("er", seed=n, n=n, avg_deg=4.0)
    if kind == "ba":
        return graphs.make("ba", seed=n, n=n, m=2)
    if kind == "grid":
        r, c = GRID[n]
        return graphs.make("grid", seed=n, r=r, c=c)
    return graphs.make("sbm", seed=n, n=n, k=4)


def main() -> int:
    rows, failures = [], []
    for kind in KINDS:
        for n in SIZES:
            rec = gen(kind, n)
            nn = rec["n"]
            ei, w = rec["ei"], rec["w"]
            adj = algorithms.build_adj(nn, ei, w)

            t_dij = tmed(lambda: algorithms.dijkstra(nn, adj, rec["s"]), target=0.02)
            t_bf = tmed(lambda: algorithms.bellman_ford(nn, ei, w, rec["s"],
                                                       record_history=False), target=0.2, reps=3)
            t_fl = tmed(lambda: algorithms.floyd(rec["W"]), target=0.2, reps=3)

            G = nx.Graph()
            G.add_nodes_from(range(nn))
            for e in range(len(w)):
                G.add_edge(int(ei[e, 0]), int(ei[e, 1]), weight=float(w[e]))
            ref = nx.single_source_dijkstra_path_length(G, rec["s"])
            ref_all = dict(nx.all_pairs_dijkstra_path_length(G)) if nn <= 200 else None

            d_dij, _, _ = algorithms.dijkstra(nn, adj, rec["s"])
            d_bf, _ = algorithms.bellman_ford(nn, ei, w, rec["s"], record_history=False)
            F = algorithms.floyd(rec["W"])

            err_dij = max(abs(float(d_dij[v]) - ref.get(v, float("inf"))) for v in range(nn))
            err_bf = max(abs(float(d_bf[v]) - ref.get(v, float("inf"))) for v in range(nn))
            if ref_all is not None:
                err_fl = max(abs(float(F[u, v]) - ref_all[u].get(v, float("inf")))
                             for u in range(nn) for v in range(nn))
            else:
                err_fl = 0.0

            for name, err in (("dijkstra", err_dij), ("bellman", err_bf), ("floyd", err_fl)):
                # 浮点舍入级误差（~1e-14）可接受；不允许出现 1e-9 以上的实质误差
                if err > 1e-9:
                    failures.append(f"{kind}/n={nn}: {name} 误差 {err:.3e}")

            rows.append({
                "kind": kind, "n": nn, "m": rec["meta"]["E"], "t_hop": rec["meta"]["t_hop"],
                "err_dijkstra": float(err_dij), "err_bellman": float(err_bf),
                "err_floyd": float(err_fl),
                "time_dijkstra_s": t_dij, "time_bellman_s": t_bf, "time_floyd_s": t_fl,
            })
            print(f"{kind:5s} n={nn:4d} m={rec['meta']['E']:5d}  "
                  f"err(dij/bf/floyd)=({err_dij:.0e},{err_bf:.0e},{err_fl:.0e})  "
                  f"dij={t_dij*1e3:7.2f}ms bf={t_bf*1e3:9.2f}ms floyd={t_fl*1e3:9.2f}ms")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "white.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    err_max = max(max(r["err_dijkstra"], r["err_bellman"], r["err_floyd"]) for r in rows)
    exact = sum(1 for r in rows if r["err_dijkstra"] == 0.0 and r["err_bellman"] == 0.0)
    ok = (err_max < 1e-9) and not failures
    print(f"\n[{'PASS' if ok else 'FAIL'}] 2.4 误差上界 = {err_max:.2e}"
          f"（Dijkstra/Bellman 逐点严格为 0 的配置 {exact}/{len(rows)}；"
          f"Floyd 的 ~1e-14 是浮点求和顺序差异，不是算法误差）")
    for f in failures[:10]:
        print("  -", f)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
