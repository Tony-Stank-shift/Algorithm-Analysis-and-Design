"""2.1 验收：手写算法 vs 独立真相源。

四件事：
  1. 1000 张随机图，Dijkstra 与 networkx 逐点对拍。
  2. 100 张 n<=7 的图，Bellman-Ford 第 k 轮 == 「至多 k 条边的最短路」（DFS 暴力枚举）。
  3. Floyd 与 networkx 全源对拍。
  4. 负权 / 负环 / 非退化检查——包括「桩实现必须失败」的负控制。

用法: python exp/check_algorithms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import algorithms  # noqa: E402

TOL = 1e-12


def random_small(n, rng, p=None):
    p = rng.uniform(0.25, 0.85) if p is None else p
    G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1 << 30)))
    while not nx.is_connected(G):
        comps = [sorted(c) for c in nx.connected_components(G)]
        G.add_edge(comps[0][0], comps[1][0])
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    ei = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    w = rng.uniform(0.5, 9.5, size=len(ei))
    return n, ei, w


def to_nx(n, ei, w):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for e in range(len(w)):
        G.add_edge(int(ei[e, 0]), int(ei[e, 1]), weight=float(w[e]))
    return G


def brute_leq_k(n, ei, w, s, K):
    """DFS 枚举简单路径，返回 (K+1,n)：至多 k 条边的最短路。"""
    adj = algorithms.build_adj(n, ei, w)
    best = np.full((K + 1, n), np.inf)

    def dfs(u, length, used, depth):
        for k in range(depth, K + 1):
            if length < best[k, u] - 1e-12:
                best[k, u] = length
            else:
                break
        if depth == K:
            return
        for v, wuv, _ in adj[u]:
            if v in used:
                continue
            used.add(v)
            dfs(v, length + wuv, used, depth + 1)
            used.discard(v)

    dfs(s, 0.0, {s}, 0)
    return best


def main() -> int:
    failures = []
    rng = np.random.default_rng(20261120)

    # ---------- 1. Dijkstra vs networkx ----------
    nonzero = 0
    for i in range(1000):
        n = int(rng.integers(2, 12))
        n, ei, w = random_small(n, rng)
        adj = algorithms.build_adj(n, ei, w)
        s = int(rng.integers(0, n))
        dist, _, _ = algorithms.dijkstra(n, adj, s)
        if abs(float(dist[s])) > TOL:
            failures.append(f"[dijkstra] 源点距离非 0: {dist[s]}")
        ref = nx.single_source_dijkstra_path_length(to_nx(n, ei, w), s)
        for v in range(n):
            r = ref.get(v, float("inf"))
            if not np.isclose(dist[v], r, rtol=1e-12, atol=1e-12):
                failures.append(f"[dijkstra] case {i} node {v}: {dist[v]} != {r}")
                break
        nonzero += int(np.sum(np.asarray(dist) > TOL))
    if not failures:
        print(f"[OK] 1) Dijkstra vs networkx：1000 图 0 差异（非零距离共 {nonzero} 个，非退化）")
    else:
        print(f"[X] 1) Dijkstra 对拍出现 {len(failures)} 处差异")

    # ---------- 2. Bellman-Ford 轮数语义 ----------
    n_bf = 0
    for i in range(100):
        n = int(rng.integers(4, 8))
        n, ei, w = random_small(n, rng, p=0.9)
        s = int(rng.integers(0, n))
        K = n - 1
        dist, D = algorithms.bellman_ford(n, ei, w, s)
        ref = brute_leq_k(n, ei, w, s, K)
        if not np.allclose(D, ref, rtol=0, atol=1e-9):
            bad = np.argwhere(~np.isclose(D, ref, rtol=0, atol=1e-9))[:3]
            failures.append(f"[bellman] case {i} 轮数语义不符，位置 {bad.tolist()}")
        else:
            n_bf += 1
        # 不存历史时结果必须一致
        d2, hist = algorithms.bellman_ford(n, ei, w, s, record_history=False)
        if hist is not None or not np.allclose(d2, dist, atol=1e-12):
            failures.append("[bellman] record_history=False 结果不一致")
    print(f"[{'OK' if n_bf == 100 else 'X'}] 2) Bellman-Ford 第 k 轮 == 至多 k 条边：{n_bf}/100")

    # ---------- 3. Floyd vs networkx ----------
    n_fl = 0
    for i in range(100):
        n = int(rng.integers(3, 10))
        n, ei, w = random_small(n, rng)
        W = algorithms.dense_from_sparse(n, ei, w)
        F = algorithms.floyd(W)
        ref = dict(nx.all_pairs_dijkstra_path_length(to_nx(n, ei, w)))
        ok = True
        for u in range(n):
            for v in range(n):
                r = ref[u].get(v, float("inf"))
                if not np.isclose(F[u, v], r, rtol=1e-12, atol=1e-12):
                    failures.append(f"[floyd] case {i} ({u},{v}): {F[u,v]} != {r}")
                    ok = False
                    break
            if not ok:
                break
        n_fl += int(ok)
    print(f"[{'OK' if n_fl == 100 else 'X'}] 3) Floyd vs networkx 全源对拍：{n_fl}/100")

    # ---------- 4. 负权 / 负环 ----------
    n, ei, w = 3, np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64), \
        np.asarray([1.0, 1.0, -3.0])
    adj = algorithms.build_adj(n, ei, w)
    try:
        algorithms.dijkstra(n, adj, 0)
        failures.append("[neg] dijkstra() 未对负权报错（只测 helper 是不够的）")
    except ValueError:
        print("[OK] 4a) dijkstra() 本身对负权报错")

    # 负环在另一个连通块里：只从 0 出发会漏
    ei2 = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    w2 = np.asarray([1.0, -1.0])
    if not algorithms.has_negative_cycle(4, ei2, w2):
        failures.append("[neg] has_negative_cycle 漏掉其它连通块里的负环")
    else:
        print("[OK] 4b) has_negative_cycle 能发现断开连通块里的负环")

    # 无负环的图必须返回 False（负控制）
    ei3 = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    w3 = np.asarray([1.0, 2.0, 3.0])
    if algorithms.has_negative_cycle(4, ei3, w3):
        failures.append("[neg] 正权图被误判为有负环")
    else:
        print("[OK] 4c) 正权图不被误判")

    if failures:
        print("\n[FAIL]")
        for f in failures[:10]:
            print("  -", f)
        return 1
    print("\n[PASS] 2.1 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
