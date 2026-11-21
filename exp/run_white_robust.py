"""2.3 验收：白盒「结构性鲁棒性」三命题，逐例实测。

**每条断言都先用 networkx 独立校验过距离**，否则一个恒返回 0 的假 Dijkstra
也能让全部命题"成立"（上一版就是这么被骗过去的）。

命题 A（单调性）：删边后 d_{G'}(s,t) >= d_G(s,t)。
命题 B（非树边不变）：删掉的边不在最短路径树上时，所有点距离完全不变。
命题 C（单边删除可算上界）：删掉最短路 P 上的边 e=(u,v) 后
        d_{G'}(s,t) <= d_G(s,u) + d_{G'}(u,v) + d_G(v,t)，
        仅对「上界有限」的案例断言；上界为 inf 的案例单列为"平凡"。

用法: python exp/run_white_robust.py
产物: results/robust_white.json
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

N_CASES = 500
TOL = 1e-9


def drop_edge(ei, w, e):
    keep = np.ones(len(w), dtype=bool)
    keep[e] = False
    return ei[keep], w[keep]


def nx_dist(n, ei, w, s):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in range(len(w)):
        G.add_edge(int(ei[e, 0]), int(ei[e, 1]), weight=float(w[e]))
    d = nx.single_source_dijkstra_path_length(G, s)
    return np.array([d.get(v, np.inf) for v in range(n)], dtype=float)


def same(a, b):
    return np.allclose(a, b, rtol=0, atol=TOL)


def main() -> int:
    rng = np.random.default_rng(20261120)
    failures = []
    mono_bad = inv_bad = bound_bad = n_trivial = 0
    gaps, hops = [], []
    n_tree_edge_cases = n_finite_dt = 0

    for case in range(N_CASES):
        n = int(rng.integers(10, 31))
        rec = graphs.make("er", seed=10000 + case, n=n, avg_deg=3.2)
        ei, w, s, t = rec["ei"], rec["w"], rec["s"], rec["t"]
        hops.append(int(rec["meta"]["t_hop"]))

        adj = algorithms.build_adj(n, ei, w)
        d, parent, _ = algorithms.dijkstra(n, adj, s)
        dt, _, _ = algorithms.dijkstra(n, adj, t)

        # --- 先校验：每条距离都必须与 networkx 一致 ---
        if not same(d, nx_dist(n, ei, w, s)):
            failures.append(f"case {case}: d 与 networkx 不一致")
            continue
        if not same(dt, nx_dist(n, ei, w, t)):
            failures.append(f"case {case}: dt 与 networkx 不一致")
            continue
        if np.isfinite(d[t]):
            n_finite_dt += 1

        # --- 命题 A：随机删一条边，距离不减 ---
        e = int(rng.integers(0, len(w)))
        ei2, w2 = drop_edge(ei, w, e)
        d2, _, _ = algorithms.dijkstra(n, algorithms.build_adj(n, ei2, w2), s)
        if not same(d2, nx_dist(n, ei2, w2, s)):
            failures.append(f"case {case}: d2 与 networkx 不一致")
            continue
        if d2[t] < d[t] - TOL:
            mono_bad += 1

        # --- 命题 B：删非树边，全部距离不变 ---
        tree = {(min(v, int(parent[v])), max(v, int(parent[v])))
                for v in range(n) if parent[v] != -1}
        cand = [i for i in range(len(w))
                if (min(int(ei[i, 0]), int(ei[i, 1])), max(int(ei[i, 0]), int(ei[i, 1]))) not in tree]
        if cand:
            e_nt = cand[int(rng.integers(0, len(cand)))]
            ei3, w3 = drop_edge(ei, w, e_nt)
            d3, _, _ = algorithms.dijkstra(n, algorithms.build_adj(n, ei3, w3), s)
            if not same(d3, nx_dist(n, ei3, w3, s)):
                failures.append(f"case {case}: d3 与 networkx 不一致")
                continue
            n_tree_edge_cases += 1
            if not same(d3, d):
                inv_bad += 1

        # --- 命题 C：删最短路上的边，验证上界 ---
        path = algorithms.path_nodes(parent, s, t)
        if path is not None and len(path) > 1:
            k = int(rng.integers(0, len(path) - 1))
            u, v = int(path[k]), int(path[k + 1])
            eid = next(i for i in range(len(w))
                       if {int(ei[i, 0]), int(ei[i, 1])} == {u, v})
            ei4, w4 = drop_edge(ei, w, eid)
            adj4 = algorithms.build_adj(n, ei4, w4)
            duv, _, _ = algorithms.dijkstra(n, adj4, u)
            d4, _, _ = algorithms.dijkstra(n, adj4, s)
            if not same(d4, nx_dist(n, ei4, w4, s)):
                failures.append(f"case {case}: d4 与 networkx 不一致")
                continue
            rhs = d[u] + duv[v] + dt[v]
            if np.isfinite(rhs) and np.isfinite(d4[t]):
                gaps.append(float(rhs - d4[t]))
                if d4[t] > rhs + TOL:
                    bound_bad += 1
            elif np.isfinite(rhs) and not np.isfinite(d4[t]):
                bound_bad += 1  # 上界有限却不可达：真违反
            else:
                n_trivial += 1  # 上界 inf，命题平凡成立

    # --- 非退化检查：全 0 / 全 inf 的假实现必须在这里翻车 ---
    degenerate = []
    if n_finite_dt < 0.8 * N_CASES:
        degenerate.append(f"只有 {n_finite_dt}/{N_CASES} 例 d(s,t) 有限，退化")
    if not gaps:
        degenerate.append("没有任何上界有限的案例，命题 C 未被真正检验")
    if sum(1 for g in gaps if g > 1e-6) < 50:
        degenerate.append("上界余量几乎全为 0，命题 C 可能恒真")
    if mono_bad + inv_bad + bound_bad + len(failures) == 0 and not gaps:
        degenerate.append("全部零违反且无有效案例")

    result = {
        "n_cases": N_CASES,
        "monotonicity_violations": mono_bad,
        "non_tree_edge_violations": inv_bad,
        "non_tree_edge_cases": n_tree_edge_cases,
        "upper_bound_violations": bound_bad,
        "bound_finite_cases": len(gaps),
        "bound_trivial_cases": n_trivial,
        "t_hop_min": int(min(hops)),
        "t_hop_median": float(np.median(hops)),
        "bound_gap_min": float(np.min(gaps)) if gaps else None,
        "bound_gap_median": float(np.median(gaps)) if gaps else None,
        "bound_gap_max": float(np.max(gaps)) if gaps else None,
        "tolerance": TOL,
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "robust_white.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    if gaps:
        print(f"上界余量：min={np.min(gaps):.4f}  median={np.median(gaps):.4f}  "
              f"max={np.max(gaps):.4f}（有限 {len(gaps)} 例 / 平凡 {n_trivial} 例）")
    print(f"t 的跳数分布：min={min(hops)}  median={np.median(hops)}")
    checks = [
        (f"命题 A 单调性（{N_CASES} 例）", mono_bad == 0, f"{mono_bad} 次违反"),
        (f"命题 B 非树边不变（{n_tree_edge_cases} 例）", inv_bad == 0, f"{inv_bad} 次违反"),
        (f"命题 C 可算上界（有限 {len(gaps)} 例）", bound_bad == 0, f"{bound_bad} 次违反"),
        ("独立校验：无一条距离与 networkx 不符", not failures, f"{len(failures)} 处不符"),
        ("非退化：确实存在非零余量", sum(1 for g in gaps if g > 1e-6) >= 50,
         f"{sum(1 for g in gaps if g > 1e-6)} 例余量 > 1e-6"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}: {detail}")
        ok = ok and passed
    if degenerate:
        ok = False
        print("\n[FAIL] 退化检查：")
        for d in degenerate:
            print("  -", d)
    for f in failures[:5]:
        print("  !", f)
    print("\n[PASS] 2.3" if ok else "\n[FAIL] 2.3")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
