"""2.4 验收：训练是否同种子逐位可复现。

做法：同进程内用同一种子训练两次（中间清掉随机状态），比对 state_dict 的 sha256。
主报告踩过的坑是多线程归约顺序导致跨进程不可复现，所以本脚本还会在子进程里
再训一次，三个哈希必须完全相同。

用法: python exp/check_node_repro.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import cora  # noqa: E402
import node_gnn  # noqa: E402

CHILD = r'''
import hashlib, sys, torch
sys.path.insert(0, r"{}")
import cora, node_gnn
d = cora.load_cora(r"{}")
A, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]), cora.normalize_features(d["features"]))
y = torch.from_numpy(d["labels"])
splits = {{k: d[k] for k in ("train_idx", "val_idx", "test_idx")}}
torch.manual_seed(0)
m = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
m, _, _ = node_gnn.train(m, A, X, y, splits, seed=0)
buf = b"".join(p.detach().numpy().tobytes() for p in m.parameters())
print(hashlib.sha256(buf).hexdigest()[:16])
'''


def train_once(seed: int = 0) -> str:
    d = cora.load_cora(ROOT / "data" / "cora")
    A, X = node_gnn.to_torch(cora.normalized_adj(d["adj"]),
                             cora.normalize_features(d["features"]))
    y = torch.from_numpy(d["labels"])
    splits = {k: d[k] for k in ("train_idx", "val_idx", "test_idx")}
    torch.manual_seed(seed)
    m = node_gnn.GCN(d["d"], 16, d["classes"], dropout=0.5)
    m, _, _ = node_gnn.train(m, A, X, y, splits, seed=seed)
    buf = b"".join(p.detach().numpy().tobytes() for p in m.parameters())
    return hashlib.sha256(buf).hexdigest()[:16]


def main() -> int:
    h1 = train_once(0)
    h2 = train_once(0)
    script = Path(ROOT / "exp" / "_repro_child.py")
    script.write_text(CHILD.format(ROOT / "src", ROOT / "data" / "cora"), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    script.unlink()
    h3 = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 else "ERR"
    print(f"  进程内第一次: {h1}\n  进程内第二次: {h2}\n  独立子进程  : {h3}")

    same_in = h1 == h2
    same_cross = h1 == h3
    print(f"[{'OK' if same_in else 'X'}] 同进程两次一致")
    print(f"[{'OK' if same_cross else 'X'}] 跨进程一致")
    ok = same_in and same_cross
    print("\n[PASS] 2.4" if ok else "\n[FAIL] 2.4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
