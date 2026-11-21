"""2.4 验收：训练是否同种子逐位可复现，且与提交的结果对得上。

检测指出上一版有两个证据链问题：
  · 子脚本写进仓库目录（不是只读，异常时会留残留）→ 改用临时目录；
  · 只验 seed 0 的哈希，没和 results/node_class.json 对账 → 现在直接对账 2 个 seed。

用法: python exp/check_node_repro.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import cora  # noqa: E402
import node_gnn  # noqa: E402

CHILD = r'''
import hashlib, json, sys, torch
sys.path.insert(0, r"{src}")
import cora, node_gnn
d = cora.load_cora(r"{data}")
A, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]), cora.normalize_features(d["features"]))
y = torch.from_numpy(d["labels"])
splits = {{k: d[k] for k in ("train_idx", "val_idx", "test_idx")}}
seed = int(sys.argv[1])
torch.manual_seed(seed)
m = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
m, _, _ = node_gnn.train(m, A, X, y, splits, seed=seed)
buf = b"".join(p.detach().numpy().tobytes() for p in m.parameters())
acc = node_gnn.accuracy(m(A, X), y, splits["test_idx"])
print(json.dumps({{"hash": hashlib.sha256(buf).hexdigest()[:16], "test": acc}}))
'''


def train_once(seed: int):
    d = cora.load_cora(ROOT / "data" / "cora")
    A, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]),
                             cora.normalize_features(d["features"]))
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}
    torch.manual_seed(seed)
    m = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
    m, _, _ = node_gnn.train(m, A, X, y, splits, seed=seed)
    buf = b"".join(p.detach().numpy().tobytes() for p in m.parameters())
    acc = node_gnn.accuracy(m(A, X), y, splits["test_idx"])
    return hashlib.sha256(buf).hexdigest()[:16], acc


def main() -> int:
    ref = json.loads((ROOT / "results" / "node_class.json").read_text(encoding="utf-8"))
    ref_seeds = {r["seed"]: r["test"] for r in ref["arms"]["gcn"]["seeds"]}

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "child.py"
        script.write_text(CHILD.format(src=ROOT / "src", data=ROOT / "data" / "cora"),
                          encoding="utf-8")
        for seed in (0, 1):
            h1, a1 = train_once(seed)
            h2, a2 = train_once(seed)
            proc = subprocess.run([sys.executable, str(script), str(seed)],
                                  capture_output=True, text=True)
            child = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
            h3 = child.get("hash", "ERR")
            same_in = h1 == h2
            same_cross = h1 == h3
            match_json = abs(a1 - ref_seeds[seed]) < 1e-9
            print(f"  seed={seed}: 进程内 {h1}/{h2} 子进程 {h3} "
                  f"test={a1:.4f} JSON={ref_seeds[seed]:.4f}")
            ok = ok and same_in and same_cross and match_json
            if not (same_in and same_cross):
                print(f"    [X] seed {seed} 复现失败")
            if not match_json:
                print(f"    [X] seed {seed} 与 results/node_class.json 不符")

    print(f"[{'OK' if ok else 'X'}] 同种子两次一致、跨进程一致、且与提交的 JSON 对得上")
    print("\n[PASS] 2.4" if ok else "\n[FAIL] 2.4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
