"""负控制：验收脚本必须能「翻车」。

第三方检测用「桩实现」一次性打穿了四条判据；这个脚本把同样的手法反过来用：
**把关键函数换成桩，验收脚本必须返回非 0**。如果换成桩还 PASS，说明那条判据
测的是恒真式，等于没测。

覆盖：阶段 1–3 的全部验收脚本。

跑法: python exp/check_negative_controls.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exp"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import algorithms  # noqa: E402
import explain  # noqa: E402
import gnn  # noqa: E402
import graphs  # noqa: E402


def zero_dijkstra(n, adj, s):
    return (np.zeros(n), np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64))


def zero_bellman(n, ei, w, s, rounds=None, record_history=True):
    return np.zeros(n), None


def zero_floyd(W):
    return np.zeros_like(W)


def fake_graphs_make(kind, seed, **kw):
    n = int(kw.get("n", kw.get("r", 1) * kw.get("c", 1)))
    return {"kind": kind, "seed": seed, "n": n,
            "ei": np.zeros((0, 2), dtype=np.int64), "w": np.zeros(0),
            "s": 0, "t": 0,
            "d_true": np.concatenate([[0.0], np.full(n - 1, np.inf)]),
            "W": None, "meta": {"E": 0, "t_hop": 0}}


def zero_minsum_forward(self, batch, layers=None):
    return torch.zeros(batch["n_total"])


def zero_predict(model, rec, s):
    return torch.zeros(rec["n"])


def zero_pred_at(model, rec, s, t, layers=None):
    return 0.0


def empty_explanation(model, rec, s, t, layers):
    return {"nodes": [], "edges": [], "reached": False, "path_length": 0.0,
            "pred": 0.0, "structural_ok": False, "valid": False}


# (模块, 被替换的对象, 属性名, 桩, 说明, 额外 argv)
CONTROLS = [
    ("run_white_robust", algorithms, "dijkstra", zero_dijkstra,
     "dijkstra 恒返回 0", []),
    ("run_white_baseline", algorithms, "bellman_ford", zero_bellman,
     "bellman_ford 恒返回 0", []),
    ("check_algorithms", algorithms, "floyd", zero_floyd,
     "floyd 恒返回 0", []),
    ("check_graphs", graphs, "make", fake_graphs_make,
     "图生成器换成 0 条边、d_true=inf", []),
    ("check_gnn", gnn.MinSumMPNN, "forward", zero_minsum_forward,
     "min-sum 前向恒返回 0", []),
    ("run_depth", gnn, "predict", zero_predict,
     "predict 恒返回 0", []),
    ("run_black", gnn.MinSumMPNN, "forward", zero_minsum_forward,
     "min-sum 前向恒返回 0", ["--quick"]),
    ("run_robust", explain, "pred_at", zero_pred_at,
     "预测恒返回 0", []),
    ("run_interp", explain, "min_sum_explanation", empty_explanation,
     "解释恒为空", []),
]


def main() -> int:
    ok = True
    # 桩实现会照常写 results/*.json（这是被测脚本的正常行为），
    # 所以跑完必须把结果目录恢复回去，否则负控制会把真实结果覆盖成假数据
    # ——上一轮就这么把 white.json 写成过 err=95.4。
    results = ROOT / "results"
    backup = None
    if results.exists():
        import shutil
        import tempfile
        backup = Path(tempfile.mkdtemp(prefix="negctl_")) / "results"
        shutil.copytree(results, backup)

    for module_name, owner, attr, stub, desc, extra_argv in CONTROLS:
        orig = getattr(owner, attr)
        old_argv = sys.argv[:]
        setattr(owner, attr, stub)
        sys.argv = [module_name] + extra_argv
        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)
            try:
                rc = mod.main()
            except SystemExit as exc:
                rc = int(exc.code or 0)
            except Exception:  # noqa: BLE001 桩导致异常也算"抓住了"
                rc = 1
        finally:
            setattr(owner, attr, orig)
            sys.argv = old_argv
            importlib.reload(mod)
        caught = rc != 0
        ok = ok and caught
        print(f"[{'OK' if caught else 'X'}] {module_name:18s} {desc:28s} → "
              f"{'正确失败' if caught else '仍然 PASS（判据无效）'}")

    if backup is not None:
        import shutil
        shutil.rmtree(results, ignore_errors=True)
        shutil.copytree(backup, results)
        shutil.rmtree(backup.parent, ignore_errors=True)
        print("（已恢复 results/，桩实现写的假结果不会留在产物里）")
    print("\n[PASS] 负控制：所有验收脚本都能被桩实现触发失败" if ok
          else "\n[FAIL] 负控制：有验收脚本对桩实现无感")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
