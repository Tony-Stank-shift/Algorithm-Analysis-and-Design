"""用 results/black.json 做统计：**非配对**检验 + bootstrap 置信区间 + 效应量。

为什么不用配对检验：8 个 seed 是 8 个**不同的模型**，seed 下标只是训练随机流的编号，
第 i 号 seed 的两个臂之间没有自然配对关系。按 seed 下标配对会人为压低 p 值。
这里用 Mann-Whitney U（非配对）+ 均值差的 percentile bootstrap CI。

用法: python exp/analyze_black.py
产物: results/black_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

TESTS = ["id_er", "ba", "grid", "sbm", "nshift_100", "nshift_200"]
DEPTHS = [2, 4, 8]
BOOT = 20000


def boot_ci(a, b, rng, alpha=0.05):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diffs = np.empty(BOOT)
    for i in range(BOOT):
        diffs[i] = (a[rng.integers(0, len(a), len(a))].mean()
                    - b[rng.integers(0, len(b), len(b))].mean())
    return (float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


def main() -> int:
    data = json.loads((ROOT / "results" / "black.json").read_text(encoding="utf-8"))
    runs = data["runs"]
    base = data["baseline_identity"]
    rng = np.random.default_rng(20261120)

    def vals(arm, depth, test, extrap=False):
        key = "tests_extrap" if extrap else "tests"
        return [r[key][test]["mae"] for r in runs
                if r["arm"] == arm and r["depth"] == depth]

    print("=== L0：未训练恒等 min-sum（= Bellman-Ford 跑满）===")
    for t in TESTS:
        print(f"  {t:11s} MAE={base[t]['mae']:.6f}  max={base[t]['max_err']:.6f}")

    out = {}
    print(f"\n{'比较':16s} {'结构臂':>9s} {'黑盒臂':>9s} {'降幅':>7s} {'MWU p':>9s} {'[95% CI]':>20s}")
    for depth in DEPTHS:
        for test in TESTS:
            a = vals("minsum", depth, test, extrap=True)
            b = vals("mean", depth, test)
            if not a or not b:
                continue
            p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            lo, hi = boot_ci(a, b, rng)
            rec = {
                "minsum_mean": float(np.mean(a)), "minsum_std": float(np.std(a, ddof=1)),
                "mean_mean": float(np.mean(b)), "mean_std": float(np.std(b, ddof=1)),
                "minsum_median": float(np.median(a)), "mean_median": float(np.median(b)),
                "reduction": float(1 - np.mean(a) / np.mean(b)) if np.mean(b) > 0 else None,
                "mwu_p": p, "diff_ci95": [lo, hi], "n_per_arm": len(a),
            }
            out[f"d{depth}/{test}"] = rec
            print(f"d{depth}/{test:8s} {rec['minsum_mean']:9.3f} {rec['mean_mean']:9.3f} "
                  f"{100 * rec['reduction']:6.1f}% {p:9.2e} [{lo:7.2f},{hi:7.2f}]")

    same = {}
    print("\n=== 同深度（不迭代到收敛）===")
    for depth in DEPTHS:
        for test in TESTS:
            a = vals("minsum", depth, test, extrap=False)
            b = vals("mean", depth, test)
            lo, hi = boot_ci(a, b, rng)
            same[f"d{depth}/{test}"] = {
                "minsum_mean": float(np.mean(a)), "mean_mean": float(np.mean(b)),
                "mwu_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
                "diff_ci95": [lo, hi],
            }
            print(f"d{depth}/{test:9s} 结构臂 {np.mean(a):8.3f}  黑盒臂 {np.mean(b):8.3f}  "
                  f"p={same[f'd{depth}/{test}']['mwu_p']:.2e}")

    collapsed = {}
    print("\n=== 塌缩统计（id_er，MAE>10 记为塌缩）===")
    for arm in ("mean", "minsum"):
        for depth in DEPTHS:
            v = vals(arm, depth, "id_er")
            collapsed[f"{arm}-d{depth}"] = {
                "n": len(v), "collapsed": int(sum(1 for x in v if x > 10)),
                "values": [round(x, 3) for x in v]}
            print(f"  {arm}-d{depth}: {collapsed[f'{arm}-d{depth}']['collapsed']}/{len(v)}  "
                  f"{collapsed[f'{arm}-d{depth}']['values']}")

    g = [r["params"] for r in runs if r["arm"] == "minsum" and r["depth"] == 8][0]
    print("\n=== 训练后的边权变换 g（d=8，第 1 个 seed）===")
    print(f"  a={g['a']:.4f}  b={g['b']:.4f}  scale=softplus(a)={g['scale']:.4f}")
    for k, v in g["g_probe"].items():
        print(f"  {k}: g={v:.4f}   （恒等应为 {k.split('=')[1]}）")

    (ROOT / "results" / "black_stats.json").write_text(json.dumps(
        {"nonpaired_extrapolated": out, "same_depth": same,
         "collapse": collapsed, "learned_g": g, "baseline_identity": base},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
