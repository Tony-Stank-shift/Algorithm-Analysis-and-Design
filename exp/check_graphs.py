"""1.3 验收：四类图各生成 10 张、同 seed 逐字节相同、d_true 与 networkx 一致。

判据必须能**证伪**：所以除了"对得上"，还要断言那些桩实现会违反的事实
（n 等于请求值、E>0、连通、d_true 全有限、d_true[t] 最大、t 确实是最远跳）。

用法: python exp/check_graphs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

import graphs  # noqa: E402

# (kind, 参数, 期望的 n（None 表示按参数算）)
CASES = [
    ("er", dict(n=40, avg_deg=4.0), 40),
    ("ba", dict(n=40, m=2), 40),
    ("grid", dict(r=5, c=8), 40),
    ("sbm", dict(n=40, k=4), 40),
]


def to_nx(rec):
    G = nx.Graph()
    G.add_nodes_from(range(rec["n"]))
    for e in range(len(rec["w"])):
        G.add_edge(int(rec["ei"][e, 0]), int(rec["ei"][e, 1]), weight=float(rec["w"][e]))
    return G


def main() -> int:
    failures = []
    print(f"{'kind':6s} {'n':>4s} {'E':>5s} {'t':>3s} {'t_hop':>5s} {'d(s,t)':>8s} {'fp':>16s}")
    for kind, kw, want_n in CASES:
        for seed in range(10):
            tag = f"{kind}#{seed}"
            a = graphs.make(kind, seed=seed, **kw)
            b = graphs.make(kind, seed=seed, **kw)

            # --- 可复现 ---
            if graphs.fingerprint(a) != graphs.fingerprint(b):
                failures.append(f"{tag} 指纹不一致")
            if not (np.array_equal(a["ei"], b["ei"]) and np.array_equal(a["w"], b["w"])):
                failures.append(f"{tag} ei/w 不一致")

            # --- 结构性判据（桩实现会在这里被抓住）---
            if a["n"] != want_n:
                failures.append(f"{tag} n={a['n']} != 期望 {want_n}")
            if a["meta"]["E"] <= 0:
                failures.append(f"{tag} E=0")
            G = to_nx(a)
            if not nx.is_connected(G):
                failures.append(f"{tag} 不连通")
            if not np.all(np.isfinite(a["d_true"])):
                failures.append(f"{tag} d_true 含 inf（不连通或实现错）")
            hop = nx.single_source_shortest_path_length(G, a["s"])
            max_hop = max(hop.values())
            if a["meta"]["t_hop"] != max_hop:
                failures.append(f"{tag} t_hop 不是最远跳数")
            if hop[a["t"]] != max_hop:
                failures.append(f"{tag} t 不是跳数最远点")
            # 定义是「跳数最远优先，并列时取加权最远」，所以要在这个并列集里比
            same_hop = [v for v, h in hop.items() if h == max_hop]
            if a["d_true"][a["t"]] < max(a["d_true"][v] for v in same_hop) - 1e-9:
                failures.append(f"{tag} 同跳数内 t 不是加权最远")

            # --- d_true 与 networkx 对拍（只比有限值，避免 inf==inf 恒真）---
            ref = nx.single_source_dijkstra_path_length(G, a["s"])
            for v in range(a["n"]):
                r = ref.get(v, float("inf"))
                if not np.isfinite(r):
                    failures.append(f"{tag} networkx 认为不可达 {v}")
                    break
                if not np.isclose(a["d_true"][v], r, rtol=1e-12, atol=1e-12):
                    failures.append(f"{tag} d_true[{v}]={a['d_true'][v]} != {r}")
                    break

            if seed == 0:
                print(f"{kind:6s} {a['n']:4d} {a['meta']['E']:5d} {a['t']:3d} "
                      f"{a['meta']['t_hop']:5d} {a['d_true'][a['t']]:8.3f} "
                      f"{graphs.fingerprint(a):>16s}")

    if failures:
        print("\n[FAIL]")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print("\n[PASS] 1.3 四类图 x 10 seed：可复现 + 结构判据 + d_true 与 networkx 一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
