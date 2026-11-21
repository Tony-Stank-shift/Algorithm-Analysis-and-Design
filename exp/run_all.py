"""一键复现：按依赖顺序跑完全部验收脚本与实验。

用法:
    python exp/run_all.py              # 全量（约 10 分钟，含 48 次训练）
    python exp/run_all.py --skip-train # 跳过训练与统计（约 2 分钟）

每个步骤都会打印自己的判据；任何一步返回非 0，本脚本最终也返回非 0。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("环境自检", "check_env.py", False),
    ("图生成器可复现性", "check_graphs.py", False),
    ("三算法独立对拍", "check_algorithms.py", False),
    ("白盒基线表", "run_white_baseline.py", False),
    ("复杂度实测", "run_complexity.py", False),
    ("删边鲁棒性三命题", "run_white_robust.py", False),
    ("消息传递语义", "check_gnn.py", False),
    ("深度下界构造", "run_depth.py", False),
    ("双臂训练（48 次）", "run_black.py", True),
    ("非配对统计检验", "analyze_black.py", True),
    ("准确率总表", "run_accuracy.py", True),
    ("鲁棒性评测", "run_robust.py", False),
    ("可解释性评测", "run_interp.py", False),
    ("失败案例", "run_failures.py", False),
    ("负控制（9 个桩实现）", "check_negative_controls.py", False),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过训练与依赖其结果的两步（复用已有 results/black.json）")
    args = ap.parse_args()

    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if args.skip_train:
        # 让 _trained.get() 在缺缓存时报错，而不是悄悄重训 40 轮
        env["JARVIS_NO_TRAIN"] = "1"

    results = []
    t_all = time.time()
    for i, (name, script, needs_train) in enumerate(STEPS, 1):
        if args.skip_train and needs_train:
            results.append((name, script, "跳过", 0.0))
            print(f"[{i:2d}/{len(STEPS)}] {name:22s} 跳过（--skip-train）")
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(ROOT / "exp" / script)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        dt = time.time() - t0
        status = "PASS" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        results.append((name, script, status, dt))
        print(f"[{i:2d}/{len(STEPS)}] {name:22s} {status:10s} {dt:6.1f}s")
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
            for line in tail:
                print(f"        {line}")

    total = time.time() - t_all
    failed = [r for r in results if r[2].startswith("FAIL")]
    print("\n" + "=" * 56)
    print(f"{'步骤':24s} {'状态':10s} 耗时")
    for name, _script, status, dt in results:
        print(f"{name:24s} {status:10s} {dt:6.1f}s")
    print("=" * 56)
    print(f"总计 {total:.1f}s；失败 {len(failed)} 步")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
