"""四类可控图分布：ER / BA / grid / SBM。

每条记录是一个 dict：
    kind, seed, n, ei (E,2 int64), w (E,), s, t, d_true (n,), W (n<=512 时给稠密), meta

保证
----
1. **完全可复现**：同一个 (kind, seed, 参数) 生成的 ei/w 逐字节相同。
2. **连通**：生成后若断连，用确定性补边策略连起来，n 保持不变。
3. **有长跳结构**：t 取 s 的最远点，避免 s-t 只差一跳导致实验无区分度。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import networkx as nx
import numpy as np

import algorithms


def _seed_int(kind: str, seed: int, salt: str = "") -> int:
    """把 (kind, seed, salt) 稳定映射成 63 位整数，避免各生成器共用同一随机流。"""
    h = hashlib.sha256(f"{kind}:{seed}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**63 - 1)


def _connect_deterministic(G: nx.Graph) -> nx.Graph:
    """若图断连，按「每个连通块的最小节点」顺序把相邻块用一条边连起来。"""
    comps = [sorted(c) for c in nx.connected_components(G)]
    if len(comps) <= 1:
        return G
    comps.sort(key=lambda c: c[0])
    for i in range(len(comps) - 1):
        G.add_edge(comps[i][0], comps[i + 1][0])
    return G


def _finalize(G: nx.Graph, kind: str, seed: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    G = _connect_deterministic(G)
    mapping = {v: i for i, v in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    n = int(G.number_of_nodes())
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    ei = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(_seed_int(kind, seed, "weights"))
    w = rng.uniform(1.0, 10.0, size=len(ei))

    adj = algorithms.build_adj(n, ei, w)
    dist, _, _ = algorithms.dijkstra(n, adj, 0)

    # t 取「跳数最远」的点（BFS），并列时用加权距离打破。
    # 早先版本取加权距离最远，实测 500 个案例里有 62 例（12.4%）s-t 只差 1–2 跳，
    # 与「要有长跳结构、否则深度拉不开差距」的初衷不符，故改为跳数优先。
    hop = nx.single_source_shortest_path_length(G, 0)
    max_hop = max(hop.values())
    cand = [v for v, h in hop.items() if h == max_hop]
    t = int(min(cand, key=lambda v: (-float(dist[v]), v)))

    rec = {
        "kind": kind, "seed": seed, "n": n,
        "ei": ei, "w": w, "s": 0, "t": t, "d_true": dist,
        "W": algorithms.dense_from_sparse(n, ei, w) if n <= 512 else None,
        "meta": dict(meta, E=int(len(w)), t_hop=int(max_hop)),
    }
    return rec


# ---------------------------------------------------------------- 四类生成器
def erdos_renyi(n: int, seed: int, avg_deg: float = 4.0) -> Dict[str, Any]:
    p = avg_deg / max(n - 1, 1)
    G = nx.erdos_renyi_graph(n, p, seed=_seed_int("er", seed))
    return _finalize(G, "er", seed, {"avg_deg": avg_deg, "p": p})


def barabasi_albert(n: int, seed: int, m: int = 2) -> Dict[str, Any]:
    G = nx.barabasi_albert_graph(n, m, seed=_seed_int("ba", seed))
    return _finalize(G, "ba", seed, {"m": m})


def grid(r: int, c: int, seed: int) -> Dict[str, Any]:
    G = nx.grid_2d_graph(r, c)
    return _finalize(G, "grid", seed, {"r": r, "c": c})


def sbm(n: int, seed: int, k: int = 4, p_in: float = 0.15,
        p_out: float = 0.01) -> Dict[str, Any]:
    sizes = [n // k] * k
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs, seed=_seed_int("sbm", seed))
    return _finalize(G, "sbm", seed, {"k": k, "p_in": p_in, "p_out": p_out})


def make(kind: str, seed: int, **kw) -> Dict[str, Any]:
    """统一入口。kind in {er, ba, grid, sbm}。"""
    if kind == "er":
        return erdos_renyi(seed=seed, **kw)
    if kind == "ba":
        return barabasi_albert(seed=seed, **kw)
    if kind == "grid":
        return grid(seed=seed, **kw)
    if kind == "sbm":
        return sbm(seed=seed, **kw)
    raise ValueError(f"未知图类型: {kind}")


def fingerprint(rec: Dict[str, Any]) -> str:
    """记录的指纹，用于验证可复现性。"""
    h = hashlib.sha256()
    h.update(rec["ei"].tobytes())
    h.update(rec["w"].tobytes())
    h.update(f"{rec['kind']}:{rec['seed']}:{rec['n']}".encode())
    return h.hexdigest()[:16]
