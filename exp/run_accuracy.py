"""4.1 验收：准确率总表——三臂 × 六个测试集，MAE / 命中率 / **最坏误差**。

最坏误差单独列出：均值会把"少数灾难性错误"藏起来，而这正是黑盒臂的特征。

产物: results/accuracy.json
用法: python exp/run_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

TESTS = ["id_er", "ba", "grid", "sbm", "nshift_100", "nshift_200"]
ARMS = ["identity-x128", "minsum-d8-x128", "mean-d8", "minsum-d8"]


def main() -> int:
    data = json.loads((ROOT / "results" / "black.json").read_text(encoding="utf-8"))
    agg = data["aggregate"]

    table = {}
    for arm in ARMS:
        row = agg.get(arm)
        if row is None:
            continue
        table[arm] = {t: {"mae": row[t]["mae"], "hit_abs05": row[t]["hit_abs05"],
                          "max_err": row[t]["max_err"]} for t in TESTS}

    # 白盒（精确解）不再硬编码 0/1/0：那行「永远通过」的常数没有测量意义。
    # 精确性由 2.1（1000 图对拍）与 2.4（三算法误差）两处独立测量覆盖，这里只引用。
    white = json.loads((ROOT / "results" / "white.json").read_text(encoding="utf-8"))
    white_max = max(max(r["err_dijkstra"], r["err_bellman"], r["err_floyd"]) for r in white)
    print(f"注：精确解的误差见 2.4——12 组配置最大 {white_max:.2e}"
          f"（Dijkstra/Bellman 严格 0，Floyd 为浮点舍入）；对拍见 2.1（1000 图 0 差异）")

    print("\nMAE（括号内为命中率 |err|<=0.5，最后一项为最坏误差）")
    hdr = f"{'arm':18s}" + "".join(f"{t:>26s}" for t in TESTS)
    print(hdr)
    for arm in ARMS:
        if arm not in table:
            continue
        cells = []
        for t in TESTS:
            c = table[arm][t]
            cells.append(f"{c['mae']:8.3f}({c['hit_abs05']:.2f})/{c['max_err']:7.2f}")
        print(f"{arm:18s}" + "".join(f"{x:>26s}" for x in cells))

    (ROOT / "results" / "accuracy.json").write_text(
        json.dumps({"table": table, "white_max_err": white_max,
                    "note": "精确解行已移除（硬编码常数不算测量）；见 white.json"},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # 判据：① 白盒误差（来自 white.json 的真实测量）≤1e-9；
    #       ② 最坏误差确实远大于平均；③ L1 比 L2 小一个数量级以上
    exact_ok = white_max < 1e-9
    mean_bad = table["mean-d8"]
    worst_ratio = max(mean_bad[t]["max_err"] / max(mean_bad[t]["mae"], 1e-9) for t in TESTS)
    l1 = table["minsum-d8-x128"]
    l1_worst = max(l1[t]["max_err"] / max(l1[t]["mae"], 1e-9) for t in TESTS)
    checks = [
        ("白盒误差来自 white.json 的真实测量且 ≤1e-9", exact_ok, f"{white_max:.2e}"),
        ("黑盒的最坏误差/均值 ≥ 3（均值确实藏了灾难性错误）", worst_ratio >= 3.0,
         f"{worst_ratio:.1f}×"),
        ("结构臂的最坏误差/均值 < 黑盒", l1_worst < worst_ratio,
         f"{l1_worst:.1f}× vs {worst_ratio:.1f}×"),
        ("L1 的 MAE < L2 的 1/10",
         table["minsum-d8-x128"]["id_er"]["mae"] * 10 < table["mean-d8"]["id_er"]["mae"],
         f"{table['minsum-d8-x128']['id_er']['mae']:.3f} vs "
         f"{table['mean-d8']['id_er']['mae']:.3f}"),
    ]
    ok = True
    print()
    for name, passed, detail in checks:
        print(f"[{'OK' if passed else 'X'}] {name}：{detail}")
        ok = ok and passed
    print("\n[PASS] 4.1" if ok else "\n[FAIL] 4.1")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
