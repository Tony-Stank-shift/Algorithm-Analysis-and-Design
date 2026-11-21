"""3.1 验收：消息传递框架的语义检查。

关键检查：min-sum 在 identity 模式下与同步 Bellman-Ford **逐点一致**。
注意口径：模型在 float32 下工作，且边权先 ÷10 再 ×10，所以是
「在 float32 容差（1e-5）内逐点相等」，不是逐位相等。
（第三方检测曾在仓库外用 300 张随机权重图实测 max|Δ|=4.15e-6；该检测脚本未入库，
本文件只覆盖 toy 图与 1 张 ER 图。）

  4 点图  s-a(1), a-b(1), b-t(1), s-t(10)
  2 层  ->  d(t) = 10（走不到 3 条边的路）
  3 层  ->  d(t) = 3
另外检查：批处理（不相交并集）与单图一致；sum 臂前向 shape 正确；
断点保存/加载后输出一致。

用法: python exp/check_gnn.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import algorithms  # noqa: E402
import gnn  # noqa: E402
import graphs  # noqa: E402
import train as train_mod  # noqa: E402


def toy():
    ei = np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    w = np.asarray([1.0, 1.0, 1.0, 10.0])
    n = 4
    adj = algorithms.build_adj(n, ei, w)
    d, _, _ = algorithms.dijkstra(n, adj, 0)
    return {"n": n, "ei": ei, "w": w, "s": 0, "t": 3, "d_true": d,
            "W": algorithms.dense_from_sparse(n, ei, w), "kind": "toy", "seed": 0,
            "meta": {"E": len(w), "t_hop": 2}}, adj


def main() -> int:
    torch.manual_seed(0)
    rec, _adj = toy()
    _, D = algorithms.bellman_ford(rec["n"], rec["ei"], rec["w"], 0)
    print("Bellman-Ford 各轮:", np.round(D, 3).tolist())

    ok = True
    for layers, expect in ((1, 10.0), (2, 10.0), (3, 3.0)):
        m = gnn.MinSumMPNN(layers=layers, identity=True)
        pred = gnn.predict(m, rec, 0).numpy()
        # 模型在归一化空间用 LARGE=1e4 表示"不可达"，输出时乘回 W_SCALE；
        # Bellman-Ford 用 inf。比较前把 inf 映射到 LARGE*W_SCALE。
        ref = np.where(np.isinf(D[layers]), gnn.LARGE * gnn.W_SCALE, D[layers])
        exact = np.allclose(pred, ref, atol=1e-5)
        got = pred[3]
        good = abs(got - expect) < 1e-6 and exact
        ok = ok and good
        print(f"[{'OK' if good else 'X'}] {layers} 层 min-sum: d(t)={got:.4f}（期望 {expect}）"
              f"，整体与 BF 第 {layers} 轮一致={exact}")

    # 批处理 == 单图
    rec2 = graphs.make("er", seed=3, n=12, avg_deg=3.0)
    m = gnn.MinSumMPNN(layers=4, identity=True)
    b = gnn.collate([(rec, 0), (rec2, 2)])
    out = m(b)
    p1 = gnn.predict(m, rec, 0)
    p2 = gnn.predict(m, rec2, 2)
    same = torch.allclose(out * gnn.W_SCALE, torch.cat([p1, p2]), atol=1e-5)
    ok = ok and same
    print(f"[{'OK' if same else 'X'}] 不相交并集批处理 == 逐图推理: {same}")

    # sum 臂 smoke
    sm = gnn.SumMPNN(layers=2, hidden=16, agg="mean")
    y = sm(b)
    shape_ok = tuple(y.shape) == (b["n_total"],)
    ok = ok and shape_ok
    print(f"[{'OK' if shape_ok else 'X'}] SumMPNN 前向 shape={tuple(y.shape)}")

    # 断点保存/加载：训练几轮拿到非平凡参数，存盘再读回，输出必须一致
    m2 = gnn.SumMPNN(layers=2, hidden=16, agg="sum")
    train_mod.train_model(m2, [(rec, 0), (rec2, 2)], epochs=3, seed=0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ckpt.pt"
        train_mod.save(m2, p)
        m3 = gnn.SumMPNN(layers=2, hidden=16, agg="sum")
        train_mod.load(m3, p)
        same_ckpt = torch.allclose(m2(b), m3(b), atol=1e-6)
    ok = ok and same_ckpt
    print(f"[{'OK' if same_ckpt else 'X'}] 断点保存/加载后输出一致: {same_ckpt}")

    print("\n[PASS] 3.1" if ok else "\n[FAIL] 3.1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
