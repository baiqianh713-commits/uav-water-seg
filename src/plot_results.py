"""画论文图 5-2：同分布主网格的预算曲线（含误差棒）。

这张图的作用是**诚实呈现负结果**，不是展示方法优势。
三条曲线（random / hybrid / kcenter）的误差棒大幅重叠，
10 格配对 t 检验全部 p>0.05 —— 图要让读者一眼看出这一点，
所以误差棒必须画、必须是 ±1 个种子标准差，不能改用标准误把它缩小。

为什么不画标准误（SEM）
----------------------
SEM = sd/sqrt(n)，画出来误差棒只有标准差的 1/3.2，视觉上三条线就"分开"了。
但这里要回答的问题是"换一个种子，结果会落在哪个范围"，
那是标准差而不是均值的精度。用 SEM 会把负结果画成正结果。

运行
----
    python -m src.plot_results

输出 runs/fig5_2_budget_curve.{png,pdf}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP_DIR = Path("runs/field_sweep")
BASELINE = Path("runs/field_baseline/result.json")   # 全量 227 张，固定轮数 E=12
OUT_DIR = Path("runs")

# 策略的显示名与样式。random 用黑色实线作基准，便于与其余两条比较。
#
# dodge：三条曲线在同一个 N 上取值，误差棒会完全重叠、互相挡住。
# 所以把每条曲线的横坐标轻微左右错开一点点。
# 横轴是对数轴，错开要用**乘法**（x * 0.97）而不是加法，否则小 N 处挪得过多。
# 这只是显示上的错位，不改变任何数值。
STYLE = {
    "random":  dict(label="random (baseline)", color="0.25", marker="o", ls="-", dodge=0.97),
    "kcenter": dict(label="k-center greedy",   color="#1f77b4", marker="s", ls="--", dodge=1.00),
    "hybrid":  dict(label="hybrid (α=0.5)",    color="#d62728", marker="^", ls="-.", dodge=1.03),
}


def load_sweep(metric: str = "test_iou") -> dict[str, dict[int, list[float]]]:
    """读 150 组结果，按 {策略: {预算: [各种子的值]}} 组织。

    只读各实验目录下的 result.json（一目录一条结果），
    不读 runs/all_results.jsonl —— 后者是追加日志，重跑会重复计数。
    """
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fp in sorted(SWEEP_DIR.glob("*/result.json")):
        try:
            r = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  [!] {fp} 解析失败，已跳过")
            continue
        out[r["strategy"]][r["budget"]].append(r[metric])
    return out


def main() -> None:
    data = load_sweep()
    if not data:
        raise SystemExit(f"{SWEEP_DIR} 下没有结果")

    budgets = sorted(data["random"].keys())
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for strat in ("random", "kcenter", "hybrid"):
        if strat not in data:
            continue
        means, sds, ns = [], [], []
        for n in budgets:
            v = np.array(data[strat][n])
            means.append(v.mean())
            sds.append(v.std(ddof=1))
            ns.append(len(v))
        st = STYLE[strat]
        xs = [n * st["dodge"] for n in budgets]
        ax.errorbar(xs, means, yerr=sds,
                    capsize=3.5, lw=1.6, ms=5.5, elinewidth=1.1,
                    label=f"{st['label']}  (n={ns[0]} seeds)",
                    color=st["color"], marker=st["marker"], ls=st["ls"],
                    zorder=3)

    # 全量基线：同一协议（固定轮数 E=12）下的 227 张结果
    if BASELINE.exists():
        full = json.loads(BASELINE.read_text(encoding="utf-8"))["test_iou"]
        ax.axhline(full, color="seagreen", ls=":", lw=1.6, zorder=2)
        # 标签画在虚线**下方**（va="top"）：画上方会被坐标轴顶边裁掉
        ax.text(budgets[-1] * 0.93, full - 0.0015,
                f"full data, N=227, same protocol: {full:.4f}",
                fontsize=8, color="seagreen", ha="right", va="top")

    ax.set_xscale("log")
    # 留边，否则最右一条曲线（dodge 后在 N=51.5）的误差棒会被右边框切掉
    ax.set_xlim(budgets[0] * 0.85, budgets[-1] * 1.18)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Annotation budget N (log scale)")
    ax.set_ylabel("Test IoU (aggregate, 85 frames)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    # 标题直接写出结论，避免读者把重叠的误差棒误读成方法有效
    ax.set_title("Budget curve, fixed-epoch protocol: error bars (±1 SD over\n"
                 "seeds) overlap; all 10 paired tests give p > 0.05",
                 fontsize=9.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fp = OUT_DIR / f"fig5_2_budget_curve.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")


if __name__ == "__main__":
    main()
