"""手写图算法基线：Dijkstra / Bellman-Ford / Floyd-Warshall。

设计约定
--------
1. 只依赖 numpy + heapq。networkx 不参与实现，只在测试里做对拍。
2. 无向图，边权为任意实数；Dijkstra 要求非负。
3. Bellman-Ford 用**同步更新**（第 k+1 轮只读第 k 轮的值），
   这样第 k 轮严格等于「至多 k 条边的最短路」——这正是 GNN 第 k 层
   消息传递要对应的语义。原地更新会让「轮数=边数」这条性质失效。
4. 统一用 (n, ei, w) 的稀疏表示：ei 形状 (E,2) 的 int 数组，
   w 形状 (E,) 的 float 数组。稠密 W 只在画图和教学例子时构造。
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

INF = float("inf")

Adj = List[List[Tuple[int, float, int]]]  # 每个点：[(邻居, 权重, 边号), ...]


# ---------------------------------------------------------------- 基础设施
def build_adj(n: int, ei: np.ndarray, w: np.ndarray) -> Adj:
    """把边表转成邻接表（无向图，两个方向都加）。"""
    adj: Adj = [[] for _ in range(n)]
    for e in range(len(w)):
        u, v = int(ei[e, 0]), int(ei[e, 1])
        adj[u].append((v, float(w[e]), e))
        adj[v].append((u, float(w[e]), e))
    return adj


def dense_from_sparse(n: int, ei: np.ndarray, w: np.ndarray) -> np.ndarray:
    """边表 -> 稠密矩阵（无边的位置为 inf，对角线 0）。"""
    W = np.full((n, n), INF)
    np.fill_diagonal(W, 0.0)
    for e in range(len(w)):
        u, v = int(ei[e, 0]), int(ei[e, 1])
        W[u, v] = min(W[u, v], float(w[e]))
        W[v, u] = min(W[v, u], float(w[e]))
    return W


def _validate_adj(adj: Adj) -> None:
    """Dijkstra 的前置条件：非负且有限的边权。

    必须在 dijkstra 内部检查，而不是让调用方"记得"先调一个 helper——
    否则负权会静默返回垃圾值（实测会给出源点为负、parent 成环的结果，
    path_nodes 于是死循环）。
    """
    for nbrs in adj:
        for _v, w, _e in nbrs:
            if not np.isfinite(w) or w < 0:
                raise ValueError(f"Dijkstra 要求非负且有限的边权，遇到 w={w}")


# ---------------------------------------------------------------- Dijkstra
def dijkstra(n: int, adj: Adj, s: int):
    """单源最短路（非负权）。返回 (dist, parent, parent_edge)。"""
    _validate_adj(adj)
    dist = np.full(n, INF)
    parent = np.full(n, -1, dtype=np.int64)
    pedge = np.full(n, -1, dtype=np.int64)
    dist[s] = 0.0
    pq = [(0.0, s)]
    done = np.zeros(n, dtype=bool)
    while pq:
        d, u = heapq.heappop(pq)
        if done[u]:
            continue
        done[u] = True
        for v, wuv, eid in adj[u]:
            nd = d + wuv
            if nd < dist[v] - 1e-12:  # 浮点容差，避免伪更新
                dist[v] = nd
                parent[v] = u
                pedge[v] = eid
                heapq.heappush(pq, (nd, v))
    return dist, parent, pedge


def dijkstra_check_nonneg(w: np.ndarray) -> None:
    if np.any(w < 0):
        raise ValueError("Dijkstra 要求非负边权；含负权请改用 Bellman-Ford")


# ---------------------------------------------------------------- Bellman-Ford
def bellman_ford(n: int, ei: np.ndarray, w: np.ndarray, s: int,
                 rounds: Optional[int] = None, record_history: bool = True):
    """同步 Bellman-Ford。

    返回 (dist, D)，D 形状 (K+1, n)，D[k] 即「至多 k 条边」的最短路长度。
    K 默认 n-1（再多也不会变，除非有负环）。
    record_history=False 时 D 为 None，只返回最终距离（计时实验用，避免
    把 O(n^2) 的历史矩阵开销算进算法本体）。
    """
    if rounds is None:
        rounds = n - 1
    d = np.full(n, INF)
    d[s] = 0.0
    D = [d.copy()] if record_history else None
    for _ in range(rounds):
        nd = d.copy()
        for e in range(len(w)):
            u, v = int(ei[e, 0]), int(ei[e, 1])
            cand = d[u] + float(w[e])
            if cand < nd[v]:
                nd[v] = cand
            cand = d[v] + float(w[e])
            if cand < nd[u]:
                nd[u] = cand
        d = nd
        if record_history:
            D.append(d.copy())
    return d, (np.vstack(D) if record_history else None)


def has_negative_cycle(n: int, ei: np.ndarray, w: np.ndarray) -> bool:
    """检测**任意连通块**里的负环。

    初始化 d=0（等价于加一个到所有点权重 0 的超级源），而不是只从某个 s 出发——
    否则负环在别的连通块里就会被漏掉（实测会漏）。
    """
    d = np.zeros(n)
    for _ in range(n):
        changed = False
        for e in range(len(w)):
            u, v = int(ei[e, 0]), int(ei[e, 1])
            if d[u] + w[e] < d[v] - 1e-12:
                d[v] = d[u] + w[e]
                changed = True
            if d[v] + w[e] < d[u] - 1e-12:
                d[u] = d[v] + w[e]
                changed = True
        if not changed:
            return False
    return True


# ---------------------------------------------------------------- Floyd
def floyd(W: np.ndarray) -> np.ndarray:
    """全源最短路（稠密，允许 inf）。O(n^3)。

    用**预分配缓冲区**做逐轮 min-plus，避免每轮新建 n×n 临时数组——
    上一版每轮分配一个临时矩阵，n=400 时实测斜率被内存带宽拉到 3.6+，
    掩盖了算法本身的 O(n^3)。
    """
    D = W.astype(float).copy()
    np.fill_diagonal(D, 0.0)
    n = D.shape[0]
    T = np.empty_like(D)
    for k in range(n):
        np.add(D[:, k][:, None], D[k, :][None, :], out=T)
        np.minimum(D, T, out=D)
    return D


# ---------------------------------------------------------------- 路径还原
def path_nodes(parent: np.ndarray, s: int, t: int) -> Optional[List[int]]:
    """由 parent 指针还原 s->t 的路径；不可达返回 None。"""
    if t == s:
        return [s]
    if parent[t] == -1:
        return None
    out = [t]
    while out[-1] != s:
        p = int(parent[out[-1]])
        if p == -1:
            return None
        out.append(p)
    return out[::-1]


def path_edges(pedge: np.ndarray, s: int, t: int) -> Optional[List[int]]:
    """由 parent_edge 还原 s->t 路径上的边号序列。"""
    if t == s:
        return []
    if pedge[t] == -1:
        return None
    out = [int(pedge[t])]
    cur = t
    # pedge 只记录「进入该点的边」，需要父指针才能继续回溯
    return out  # 见 path_edges_full


def path_edges_full(parent: np.ndarray, pedge: np.ndarray, s: int,
                    t: int) -> Optional[List[int]]:
    """完整版：返回 s->t 路径上的边号序列。"""
    nodes = path_nodes(parent, s, t)
    if nodes is None:
        return None
    out = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        # 找到 a-b 这条边的编号：取 b 的入边即可（树上唯一）
        if int(parent[b]) == int(a):
            out.append(int(pedge[b]))
        else:
            out.append(int(pedge[a]))
    return out
