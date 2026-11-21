"""环境自检：缺依赖或没装对，直接以非 0 退出码终止。

用法: python exp/check_env.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python      : {sys.version.split()[0]}")
    print(f"platform    : {platform.platform()}")

    missing = []
    try:
        import numpy as np
        print(f"numpy       : {np.__version__}")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"numpy ({exc})")

    try:
        import networkx as nx
        print(f"networkx    : {nx.__version__}")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"networkx ({exc})")

    try:
        import matplotlib
        print(f"matplotlib  : {matplotlib.__version__}")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"matplotlib ({exc})")

    try:
        import torch
        print(f"torch       : {torch.__version__}")
        print(f"cuda        : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print("注意        : 本实验按 CPU 设计，GPU 仅是可选项")
        print(f"threads     : {torch.get_num_threads()}")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"torch ({exc})")

    if missing:
        print("\n[FAIL] 缺少依赖：" + ", ".join(missing), file=sys.stderr)
        return 1

    print("\n[PASS] 环境就绪（CPU 可跑）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
