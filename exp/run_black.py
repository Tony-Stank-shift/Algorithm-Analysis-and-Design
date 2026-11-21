"""3.2 / 3.3 / 3.4：训练两臂（mean 黑盒 / min-sum 结构），测准确率。

训练集：ER 图，n∈[12,50]，平均度 3.0（稀疏 → 直径长 → 深度才有区分度）
测试集：同分布 ER / 跨生成器 BA、grid、SBM / 规模外推 n=100、200

三档对照（这是报告的核心阶梯）：
  L0  identity 恒等 min-sum（= 不训练的 Bellman-Ford，跑满 128 轮）
  L1  min-sum 训练后（2 个参数）
  L2  mean-GNN（黑盒，10 万参数量级）
另外每个臂都报「同训练深度」和「推理迭代到收敛」两种口径。

判据（脚本内断言，不达标返回非 0）：
  ① L0 在 id_er 上 MAE < 1e-3（它本来就该是精确解）
  ② 某个深度的 min-sum 迭代到收敛后 id_er MAE < 1
  ③ mean 臂至少有 1 个 seed 收敛（MAE < 10），否则说明实验设置坏了
  ④ 所有损失有限、且末轮 < 首轮

用法:
  python exp/run_black.py --quick     # 冒烟
  python exp/run_black.py             # 正式
产物: results/black.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gnn  # noqa: E402
import graphs  # noqa: E402
import train as T  # noqa: E402

DEPTHS = [2, 4, 8]
ARMS = ["mean", "minsum"]
EXTRAP_LAYERS = 128  # 结构臂推理时多迭代的轮数（远超任何图的直径）
G_PROBE = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]  # 采样 g(w)，报告里要展示学到的算子长什么样


def gen_er(sd, n_lo=12, n_hi=50):
    n = int(np.random.default_rng(sd).integers(n_lo, n_hi + 1))
    return graphs.make("er", seed=sd, n=n, avg_deg=3.0)


def gen_ba(sd, n_lo=12, n_hi=50):
    n = int(np.random.default_rng(sd).integers(n_lo, n_hi + 1))
    return graphs.make("ba", seed=sd, n=n, m=2)


def gen_grid(sd, r=7, c=7):
    return graphs.make("grid", seed=sd, r=r, c=c)


def gen_sbm(sd, n_lo=12, n_hi=50):
    n = int(np.random.default_rng(sd).integers(n_lo, n_hi + 1))
    return graphs.make("sbm", seed=sd, n=n, k=4)


def gen_nshift(n):
    """规模外推：同生成器、同平均度，只把 n 放大。"""
    return lambda sd: graphs.make("er", seed=sd, n=n, avg_deg=3.0)


def make_model(arm: str, depth: int):
    if arm == "mean":
        return gnn.SumMPNN(layers=depth, hidden=64, agg="mean")
    return gnn.MinSumMPNN(layers=depth)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def params_of(model) -> dict:
    if isinstance(model, gnn.MinSumMPNN) and not model.identity:
        a, b = float(model.a.detach()), float(model.b.detach())
        import torch.nn.functional as F
        scale = float(F.softplus(model.a.detach()))
        return {"a": a, "b": b, "scale": scale,
                "g_probe": {f"w={w}": scale * w + b for w in G_PROBE}}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--train-graphs", type=int, default=200)
    ap.add_argument("--sources", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    if args.quick:
        args.train_graphs, args.sources, args.epochs, args.seeds = 12, 1, 2, 1

    t0 = time.time()
    tr = T.build_samples(gen_er, range(2000, 2000 + args.train_graphs), args.sources, seed0=7)
    tests = {
        "id_er": T.build_samples(gen_er, range(3000, 3040), 2, seed0=11),
        "ba": T.build_samples(gen_ba, range(3100, 3120), 2, seed0=12),
        "grid": T.build_samples(gen_grid, range(3200, 3220), 2, seed0=13),
        "sbm": T.build_samples(gen_sbm, range(3300, 3320), 2, seed0=14),
        "nshift_100": T.build_samples(gen_nshift(100), range(3400, 3410), 2, seed0=15),
        "nshift_200": T.build_samples(gen_nshift(200), range(3500, 3506), 2, seed0=16),
    }
    print(f"训练样本 {len(tr)}；测试集 " + ", ".join(f"{k}={len(v)}" for k, v in tests.items()))

    # ---------- L0：不训练的恒等 min-sum = Bellman-Ford ----------
    base = gnn.MinSumMPNN(layers=EXTRAP_LAYERS, identity=True)
    baseline = {name: T.evaluate(base, s, layers=EXTRAP_LAYERS) for name, s in tests.items()}
    print("L0 未训练恒等 min-sum（= BF 跑满）：id_er MAE="
          f"{baseline['id_er']['mae']:.6f}")

    runs = []
    for arm in ARMS:
        for depth in DEPTHS:
            for seed in range(args.seeds):
                torch.manual_seed(seed)          # 必须在建模型之前（否则初始化不可复现）
                model = make_model(arm, depth)
                tt = time.time()
                losses = T.train_model(model, tr, epochs=args.epochs, seed=seed)
                rec = {"arm": arm, "depth": depth, "seed": seed,
                       "n_params": n_params(model), "params": params_of(model),
                       "losses": losses, "loss_first": losses[0], "loss_last": losses[-1],
                       "tests": {n: T.evaluate(model, s) for n, s in tests.items()},
                       "tests_extrap": {}}
                if arm == "minsum":
                    rec["tests_extrap"] = {n: T.evaluate(model, s, layers=EXTRAP_LAYERS)
                                           for n, s in tests.items()}
                runs.append(rec)
                print(f"  {arm:6s} d={depth} seed={seed}  "
                      f"mse {losses[0]:.4f}->{losses[-1]:.4f}  "
                      f"id_er MAE={rec['tests']['id_er']['mae']:.3f}  "
                      f"n200 MAE={rec['tests']['nshift_200']['mae']:.3f}  "
                      f"({time.time() - tt:.1f}s)")
            print()

    KEYS = ("mae", "med_err", "hit_abs05", "hit_rel5", "max_err")

    agg = {}
    for arm in ARMS:
        for depth in DEPTHS:
            getter_rs = [r for r in runs if r["arm"] == arm and r["depth"] == depth]
            agg[f"{arm}-d{depth}"] = {
                t: {k: float(np.mean([r["tests"][t][k] for r in getter_rs])) for k in KEYS}
                for t in tests
            }
    for depth in DEPTHS:
        rs = [r for r in runs if r["arm"] == "minsum" and r["depth"] == depth]
        agg[f"minsum-d{depth}-x{EXTRAP_LAYERS}"] = {
            t: {k: float(np.mean([r["tests_extrap"][t][k] for r in rs])) for k in KEYS}
            for t in tests
        }
    agg[f"identity-x{EXTRAP_LAYERS}"] = {
        t: {k: baseline[t][k] for k in KEYS} for t in tests}

    od = ROOT / "results"
    od.mkdir(exist_ok=True)
    (od / "black.json").write_text(json.dumps({
        "config": vars(args), "baseline_identity": baseline,
        "runs": runs, "aggregate": agg,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    def show(title, prefix_filter):
        print(f"\n=== {title} ===")
        print(f"{'arm':20s}" + "".join(f"{t:>12s}" for t in tests))
        for key, row in agg.items():
            if not prefix_filter(key):
                continue
            print(f"{key:20s}" + "".join(f"{row[t]['mae']:12.3f}" for t in tests))

    show("平均 MAE（同训练深度 / 恒等基线）", lambda k: "-x" not in k)
    show(f"平均 MAE（结构臂推理迭代 {EXTRAP_LAYERS} 轮）", lambda k: "-x" in k)

    # ---------- 断言 ----------
    collapsed = {f"{arm}-d{d}": sum(1 for r in runs if r["arm"] == arm
                                    and r["depth"] == d and r["tests"]["id_er"]["mae"] > 10)
                 for arm in ARMS for d in DEPTHS}
    best = min(agg[f"minsum-d{d}-x{EXTRAP_LAYERS}"]["id_er"]["mae"] for d in DEPTHS)
    checks = [
        ("L0 未训练恒等臂 = 精确解（id_er MAE < 1e-3）",
         baseline["id_er"]["mae"] < 1e-3, f"{baseline['id_er']['mae']:.2e}"),
        ("结构臂迭代到收敛后 id_er MAE < 1", best < 1.0, f"{best:.3f}"),
        ("mean 臂至少 1 个 seed 收敛（MAE<10）",
         any(collapsed[f"mean-d{d}"] < args.seeds for d in DEPTHS),
         ", ".join(f"d{d}:{collapsed[f'mean-d{d}']}/{args.seeds}" for d in DEPTHS)),
        ("所有损失有限且末轮 < 首轮",
         all(np.isfinite(r["losses"]).all() and r["loss_last"] < r["loss_first"] for r in runs),
         "ok"),
        ("结构臂参数已落盘（a,b,g 采样）",
         all(r["params"] for r in runs if r["arm"] == "minsum"), "ok"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print(f"\n总耗时 {time.time() - t0:.1f}s")
    print("[PASS] run_black" if ok else "[FAIL] run_black")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
