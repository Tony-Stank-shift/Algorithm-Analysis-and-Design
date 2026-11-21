"""2.2–2.3 验收：Cora 上训练 2 层 GCN 与 MLP 对照。

判据：
  · GCN test accuracy ≥ 78%（原论文 81.5%，复现到 78+ 说明实现正确）
  · MLP 显著更低（无图结构时 Cora 大约 55–60%）
  · 训练曲线与早停记录完整

产物: results/node_class.json
用法: python exp/run_node_class.py
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

import cora  # noqa: E402
import node_gnn  # noqa: E402

SEEDS = range(8)


def main() -> int:
    d = cora.load_cora(ROOT / "data" / "cora")
    X_np = cora.normalize_features(d["features"])
    A_np = cora.normalized_adj(d["adj"])
    A, X = node_gnn.to_torch(A_np, X_np)
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}

    out = {}
    for arm, cls in (("gcn", node_gnn.GCN), ("mlp", node_gnn.MLP)):
        accs, rows = [], []
        t0 = time.time()
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = cls(d["d"], 16, d["classes"], dropout=0.5)
            model, hist, best = node_gnn.train(model, A, X, y, splits, seed=seed)
            model.eval()
            with torch.no_grad():
                logits = model(A, X)
            tr = node_gnn.accuracy(logits, y, splits["train_idx"])
            va = node_gnn.accuracy(logits, y, splits["val_idx"])
            te = node_gnn.accuracy(logits, y, splits["test_idx"])
            accs.append(te)
            rows.append({"seed": seed, "train": tr, "val": va, "test": te,
                         "best_epoch": best["epoch"], "epochs_run": len(hist),
                         "val_loss": best["val_loss"]})
            print(f"  {arm} seed={seed}: train {tr:.4f} val {va:.4f} test {te:.4f} "
                  f"(best ep {best['epoch']}, {len(hist)} epochs)")
        out[arm] = {
            "seeds": rows,
            "test_mean": float(np.mean(accs)),
            "test_std": float(np.std(accs)),
            "test_min": float(np.min(accs)),
            "test_max": float(np.max(accs)),
            "seconds": round(time.time() - t0, 1),
            "params": int(sum(p.numel() for p in model.parameters())),
            "history_last": hist,
        }
        print(f"{arm}: test {out[arm]['test_mean']:.4f} ± {out[arm]['test_std']:.4f} "
              f"(min {out[arm]['test_min']:.4f}), {out[arm]['params']} 参数, "
              f"{out[arm]['seconds']}s")

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / "node_class.json").write_text(
        json.dumps({"config": {"hidden": 16, "dropout": 0.5, "lr": 0.01,
                               "weight_decay": 5e-4, "epochs": 200, "patience": 100,
                               "seeds": list(SEEDS), "n_nodes": d["n"],
                               "n_edges": int(d["edges"].shape[0])},
                    "arms": out}, indent=2, ensure_ascii=False), encoding="utf-8")

    gcn = out["gcn"]["test_mean"]
    mlp = out["mlp"]["test_mean"]
    checks = [
        ("GCN test acc ≥ 0.78", gcn >= 0.78, f"{gcn:.4f}"),
        ("MLP 明显更低（GCN − MLP ≥ 0.15）", gcn - mlp >= 0.15, f"{gcn - mlp:.4f}"),
        ("GCN 的 8 个种子都 ≥ 0.75", out["gcn"]["test_min"] >= 0.75,
         f"min {out['gcn']['test_min']:.4f}"),
        ("MLP 落在 0.50–0.65（Cora 无结构的合理区间）", 0.50 <= mlp <= 0.65,
         f"{mlp:.4f}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 2.2/2.3" if ok else "\n[FAIL] 2.2/2.3")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
