"""画论文图 5-5：训练量—预算曲线族。

这张图是第五章最强论点（训练协议决定实验结论）唯一依赖的图。

图要说明什么
------------
横轴是总梯度步数（对数轴），纵轴是测试集 IoU，每条曲线对应一个标注预算 N。
在曲线族上再标出两种常用协议各自实际取到的点：

  - 固定轮数 E=12：总步数 = ceil(N/batch) * E，所以步数 ∝ N。
    低预算（N=10 只有 12 步）取在曲线最左端的未收敛段 → 低预算被系统性低估。
  - 固定步数 S=150：步数恒定，但每张图被看的次数 ≈ S*batch/N ∝ 1/N。
    全量（N=227 每图只被看 5.3 次）同样未收敛 → 高预算被系统性低估。

两个协议的失真落在曲线族的**相反两侧**，这就是 5.4.1 的对称性缺陷。

运行
----
    python -m src.plot_convergence

输出 runs/fig5_5_convergence.png（300 dpi，投稿可用）与同名 .pdf（矢量）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下也能出图，必须在 import pyplot 之前设
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 配置

BUDGETS = [10, 20, 50, 227]      # 227 = 全量 train
STEP_GRID = [150, 300, 600, 1200]
BATCH = 8                        # configs/field_steps.yaml 的 train.batch_size
EPOCHS_FIXED = 12                # 固定轮数协议用的 E

# 固定轮数协议（E=12）在 seed 0 上的实测值。
# 来源：runs/field_sweep/field_random_n{N}_s0（N=10/20/50）
#       runs/field_baseline（N=227，即全量，348 步）
# 这里写死是为了让图与正文表格严格一致；数字可用 analyze_sweep.py 复核。
FIXED_EPOCH_IOU = {10: 0.8955, 20: 0.9340, 50: 0.9660, 227: 0.9848}

OUT_DIR = Path("runs")
GRID_DIR = Path("runs/conv_grid")


def steps_per_epoch(n: int) -> int:
    """每轮的梯度步数。

    train.py 里 DataLoader 设的是 `drop_last=len(train_ds) > bs`，
    即训练集比一个 batch 大时**丢掉最后不满 batch 的那一批**，
    所以每轮步数是 floor(N/batch) 而不是 ceil(N/batch)。
    这个 floor/ceil 的区别会让 N=227 的步数差 12 步，
    故不自己推、而与 train.py 的实现严格对齐（日志里打印的值可核对）。
    """
    if n <= BATCH:          # 比一个 batch 还小，drop_last=False，整批算一步
        return 1
    return n // BATCH       # floor


def fixed_epoch_steps(n: int) -> int:
    """固定轮数协议（E=12）下的总梯度步数 —— 与 N 成正比，这正是缺陷来源。"""
    return steps_per_epoch(n) * EPOCHS_FIXED


def actual_steps(n: int, target: int) -> int:
    """固定步数协议下的**实际**步数。

    train.py 由目标步数反推轮数时取了 ceil，所以实际步数往往略多于目标：
    N=227 每轮 28 步，目标 150 -> 训练 6 轮 -> 实际 168 步。
    横轴用实际步数才对得上。
    """
    spe = steps_per_epoch(n)
    return -(-target // spe) * spe


def load_grid() -> dict[tuple[int, int], float]:
    """读曲线族的 16 个 result.json，返回 {(预算, 步数): test_iou}。

    只读每个实验目录自己的 result.json（一目录一条结果），
    不读 runs/all_results.jsonl——那是追加日志，重跑会重复计数。
    """
    out: dict[tuple[int, int], float] = {}
    for n in BUDGETS:
        for s in STEP_GRID:
            fp = GRID_DIR / f"n{n}_s{s}" / "result.json"
            if not fp.exists():
                print(f"  [!] 缺 {fp}，该点将在图上跳过")
                continue
            out[(n, s)] = json.loads(fp.read_text(encoding="utf-8"))["test_iou"]
    return out


def main() -> None:
    grid = load_grid()
    if not grid:
        raise SystemExit("runs/conv_grid 下没有任何 result.json，先跑 scripts/convergence_grid.sh")

    # 两栏：(a) 全量程，看固定轮数协议造成的伪预算曲线；
    #       (b) 放大收敛区，看各预算训到收敛后其实挤在一起。
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(10.2, 4.3),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    colors = plt.cm.viridis([0.05, 0.35, 0.62, 0.88])
    fe_x = [fixed_epoch_steps(n) for n in BUDGETS]
    fe_y = [FIXED_EPOCH_IOU[n] for n in BUDGETS]

    for a in (ax, axz):
        for color, n in zip(colors, BUDGETS):
            # 横轴用实际步数（N=227 的 150/300/600/1200 实为 168/308/616/1204）
            xs = [actual_steps(n, s) for s in STEP_GRID if (n, s) in grid]
            ys = [grid[(n, s)] for s in STEP_GRID if (n, s) in grid]
            label = f"N = {n}" + (" (full)" if n == 227 else "")
            a.plot(xs, ys, "o-", color=color, lw=1.8, ms=5, label=label, zorder=3)
        a.set_xscale("log")
        a.set_xticks(STEP_GRID)
        a.set_xticklabels([str(s) for s in STEP_GRID])
        # 关掉对数轴的次刻度标签（否则会冒出 2x10^2 这类标签，挤占横轴）
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.grid(alpha=0.25, ls="-", lw=0.5)
        a.set_xlabel("Total gradient steps (log scale)")

    # ---- (a) 叠加两种协议各自实际取到的点
    ax.plot(fe_x, fe_y, "s--", color="crimson", lw=1.4, ms=7,
            mfc="none", mew=1.8, label="Fixed-epoch protocol (E=12)", zorder=4)
    # 标注放在点的左上/右下交替，避免压住曲线和坐标轴
    offsets = {10: (8, 6), 20: (8, 6), 50: (8, 6), 227: (-6, -26)}
    for n, x, y in zip(BUDGETS, fe_x, fe_y):
        ax.annotate(f"N={n}, {x} st.", (x, y), textcoords="offset points",
                    xytext=offsets[n], fontsize=8, color="crimson")
    ax.axvline(150, color="darkorange", ls=":", lw=1.6, zorder=2)
    # 168 实际步 × batch 8 / 227 张 = 每图仅 5.9 次
    ax.annotate("Fixed-step protocol S=150:\nfull data sees each image only 5.9x",
                xy=(150, 0.9743), xytext=(26, 0.955),
                fontsize=8, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_ylim(0.885, 0.995)
    ax.set_ylabel("Test IoU (aggregate, 85 frames)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_title("(a) The fixed-epoch protocol traces an apparent\n"
                 "budget curve that is really a training-amount curve",
                 fontsize=9.5)

    # ---- (b) 放大收敛区
    axz.axvline(150, color="darkorange", ls=":", lw=1.6, zorder=2)
    axz.set_ylim(0.972, 0.989)
    axz.set_title("(b) Zoom: once each budget is trained to convergence,\n"
                  "N = 10 and N = 227 differ by 0.13 pt", fontsize=9.5)
    # (b) 不重复画图例，(a) 已有

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fp = OUT_DIR / f"fig5_5_convergence.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")


if __name__ == "__main__":
    main()
