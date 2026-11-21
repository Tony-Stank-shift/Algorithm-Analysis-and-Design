"""下载 Planetoid 的 Cora 数据（8 个 pickle，合计约 570 KB）。

为什么不从 LINQS 官方站下：linqs-data.soe.ucsc.edu 的证书链在本机校验失败
（UNABLE_TO_VERIFY_LEAF_SIGNATURE），而 GitHub 上的 planetoid 官方镜像可达。
文件放在 data/cora/，**不进 git**（.gitignore 已排除 data/）。

用法: python exp/fetch_cora.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "cora"
BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
FILES = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = DEST / f"ind.cora.{name}"
        if out.exists() and out.stat().st_size > 0:
            print(f"  已存在 {out.name} ({out.stat().st_size} B)")
            continue
        url = f"{BASE}/ind.cora.{name}"
        try:
            urllib.request.urlretrieve(url, out)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {url}: {exc}")
            return 1
        print(f"  下载 {out.name} ({out.stat().st_size} B)")
    total = sum((DEST / f"ind.cora.{n}").stat().st_size for n in FILES)
    print(f"[PASS] 8 个文件就绪，共 {total / 1024:.0f} KB，位于 {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
