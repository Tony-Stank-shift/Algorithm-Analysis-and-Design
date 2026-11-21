"""4.1–4.3 验收：随机加边 vs FGA 攻击。

预算 ε = 新增边数 / 原始边数，取 {1%, 3%, 5%, 10%}。
判据：
  · 干净准确率 ≈ 0.81（与 run_node_class 一致）
  · 攻击使准确率下降，且随 ε 单调
  · 同预算下 FGA 明显强于随机加边
  · 极端预算（10%）下也不至于把模型打成随机（否则说明攻击过强、不可信）

产物: results/node_attack.json
用法: python exp/run_node_attack.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import attack  # noqa: E402
import cora  # noqa: E402
import node_gnn  # noqa: E402

EPS = [0.01, 0.03, 0.05, 0.10]
SEED = 0


def main() -> int:
    d = cora.load_cora(ROOT / "data" / "cora")
    X_np = cora.normalize_features(d["features"])
    A_np = attack.dense_from_edges(d["n"], d["edges"])
    A_t, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]), X_np)
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}

    torch.manual_seed(SEED)
    model = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
    model, _, _ = node_gnn.train(model, A_t, X, y, splits, seed=SEED)
    clean = attack.accuracy_under(model, A_np, X, y, splits["test_idx"])
    n_edges = d["edges"].shape[0]
    print(f"干净 test acc = {clean:.4f}（原始边 {n_edges}）")

    out = {"clean": clean, "n_edges": int(n_edges), "eps": EPS, "arms": {}}
    t0 = time.time()
    for name, fn in (("random", attack.random_attack), ("fga", attack.fga_attack)):
        accs = []
        for eps in EPS:
            n_add = max(1, int(round(eps * n_edges)))
            if name == "random":
                A2 = fn(A_np, n_add, seed=SEED)
            else:
                A2 = fn(model, A_np, X, y, splits["test_idx"], n_add, seed=SEED)
            acc = attack.accuracy_under(model, A2, X, y, splits["test_idx"])
            accs.append({"eps": eps, "n_add": n_add, "acc": acc, "drop": clean - acc})
            print(f"  {name:6s} eps={eps:.0%} (+{n_add:4d} 边): acc {acc:.4f} "
                  f"↓{clean - acc:.4f}")
        out["arms"][name] = accs
    out["seconds"] = round(time.time() - t0, 1)

    res = ROOT / "results" / "node_attack.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    rnd = [a["drop"] for a in out["arms"]["random"]]
    fga = [a["drop"] for a in out["arms"]["fga"]]
    mono = all(b >= a - 1e-9 for a, b in zip(rnd, rnd[1:])) or \
        all(b >= a - 1e-9 for a, b in zip(fga, fga[1:]))
    checks = [
        ("干净 acc 与 run_node_class 一致（±0.02）", abs(clean - 0.8111) < 0.02, f"{clean:.4f}"),
        ("攻击使准确率下降", max(rnd) > 0.01 and max(fga) > 0.01,
         f"random {max(rnd):.3f} / fga {max(fga):.3f}"),
        ("下降随预算单调（至少一个臂）", mono, f"{['%.3f' % x for x in rnd]}"),
        ("同预算下 FGA 强于随机（10% 处）", fga[-1] > rnd[-1] + 0.02,
         f"{fga[-1]:.3f} vs {rnd[-1]:.3f}"),
        ("FGA 在 10% 预算下未把模型打成随机（<0.85 的下降）", fga[-1] < 0.85,
         f"↓{fga[-1]:.3f}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 4.1/4.2/4.3" if ok else "\n[FAIL] 4.1/4.2/4.3")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
