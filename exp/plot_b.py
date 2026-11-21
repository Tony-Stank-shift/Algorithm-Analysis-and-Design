"""5.2 验收：把 B 线的结果画成两张图。

产物:
  results/fig/b_attack.png   随机加边 vs FGA 的预算曲线
  results/fig/b_explain.png  解释质量（消融重要性 + oracle 效率）

用法: python exp/plot_b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import viz  # noqa: E402


def main() -> int:
    viz.setup()
    res = ROOT / "results"
    fig_dir = res / "fig"
    fig_dir.mkdir(exist_ok=True)

    at = json.loads((res / "node_attack.json").read_text(encoding="utf-8"))
    eps = [a["eps"] * 100 for a in at["arms"]["random"]]
    rnd = [a["drop"] for a in at["arms"]["random"]]
    fga = [a["drop"] for a in at["arms"]["fga"]]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(eps, rnd, "o-", label="随机加边")
    ax.plot(eps, fga, "s-", label="FGA")
    ax.set_xlabel("攻击预算 ε（新增边 / 原始边，%）")
    ax.set_ylabel("准确率下降")
    ax.set_title(f"Cora 上边注入攻击（干净 acc = {at['clean']:.3f}）", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "b_attack.png")
    plt.close(fig)

    pl = json.loads((res / "node_planted.json").read_text(encoding="utf-8"))
    labels = ["解释边", "oracle 上界", "随机边"]
    vals = [pl["delta_top_mean"], pl["delta_oracle_mean"], pl["delta_random_mean"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].bar(labels, vals, color=["tab:blue", "tab:gray", "tab:orange"])
    axes[0].set_ylabel("平均消融重要性 Δ")
    axes[0].set_title(f"解释器 {vals[0] / max(vals[2], 1e-12):.1f}× 于随机，"
                      f"达到 oracle 的 {pl['oracle_efficiency_median']:.0%}", fontsize=9)
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    eff = np.array([r["delta_top"] / r["delta_oracle"]
                    for r in pl["rows"] if r["delta_oracle"] > 1e-3])
    axes[1].scatter([r["delta_oracle"] for r in pl["rows"]],
                    [r["delta_top"] for r in pl["rows"]], s=16)
    lim = max(max(r["delta_oracle"] for r in pl["rows"]),
              max(r["delta_top"] for r in pl["rows"])) * 1.1
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="解释器 = oracle")
    axes[1].set_xlabel("oracle 边集的消融 Δ")
    axes[1].set_ylabel("解释器边集的消融 Δ")
    axes[1].set_title(f"越靠近对角线越好（中位效率 {np.median(eff):.2f}，n={len(eff)}）",
                      fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "b_explain.png")
    plt.close(fig)

    ok = (fig_dir / "b_attack.png").exists() and (fig_dir / "b_explain.png").exists()
    print(f"[{'OK' if ok else 'X'}] 两张图已生成：b_attack.png / b_explain.png")
    print("\n[PASS] 5.2" if ok else "\n[FAIL] 5.2")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
