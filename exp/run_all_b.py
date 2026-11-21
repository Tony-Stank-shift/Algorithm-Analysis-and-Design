"""一条命令跑完 B 线（节点分类 + 解释 + 攻击）的全部验收与实验。

前置：`python exp/fetch_cora.py` 已下载 data/cora/（约 570 KB，不入库）。
如果数据缺失，本脚本会直接报错并提示，而不是静默跳过。

用法:
    python exp/run_all_b.py             # 约 4 分钟（8 种子 GCN + 解释 + 攻击）
    python exp/run_all_b.py --quick     # 只跑验收脚本（约 40 秒）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    ("数据体检", "check_cora.py"),
    ("GCN 与手算对照", "check_node_gnn.py"),
    ("同种子复现对账", "check_node_repro.py"),
]

RUNS = [
    ("训练 GCN 与 MLP（8 种子）", "run_node_class.py"),
    ("SBM 解释真值检验", "run_node_planted.py"),
    ("Cora 解释与稳定性", "run_node_explain.py"),
    ("随机加边 vs FGA", "run_node_attack.py"),
    ("攻击前后的解释稳定性", "run_node_attack_explain.py"),
    ("三条负控制", "check_b_negative_controls.py"),
    ("结果图", "plot_b.py"),
]


def run(script: str) -> int:
    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, str(ROOT / "exp" / script)],
                          cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑验收脚本")
    args = ap.parse_args()

    if not (ROOT / "data" / "cora" / "ind.cora.x").exists():
        print("缺少 data/cora/，请先运行：python exp/fetch_cora.py")
        return 2

    steps = CHECKS + ([] if args.quick else RUNS)
    results = []
    t0 = time.time()
    for i, (name, script) in enumerate(steps, 1):
        t = time.time()
        rc = run(script)
        dt = time.time() - t
        status = "PASS" if rc == 0 else f"FAIL({rc})"
        results.append((name, status, dt))
        print(f"[{i:2d}/{len(steps)}] {name:26s} {status:10s} {dt:6.1f}s")

    failed = [r for r in results if r[1].startswith("FAIL")]
    print("\n" + "=" * 52)
    print(f"总计 {time.time() - t0:.1f}s；失败 {len(failed)} 步")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
