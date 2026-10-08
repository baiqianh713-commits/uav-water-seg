"""画论文图 4-3（三数据集最近邻距离分布）与图 4-4（难度维度活跃度热图）。

为什么要重算而不是读 JSON
--------------------------
`data/processed/*/redundancy_train.json` 里只存了最近邻距离的**摘要统计**
（均值、中位数、7 个分位点），没有逐样本的距离数组，画不出分布曲线。
`diagnose_redundancy.nn_distance_stats()` 返回的 `raw` 字段在写 JSON 时被丢掉了。
所以这里重新提特征、重算距离。

特征提取是确定性的（无随机），三个数据集的样本数都小于诊断脚本的
`--max_samples 1500`、不触发随机抽样，所以重算结果应与 JSON 里的
摘要统计**逐位一致**。脚本会做这个校验：若中位数对不上就直接报错退出，
而不是拿一组新数字悄悄替换论文表里的旧数字。

图 4-3 要说明什么
-----------------
横轴是每个样本到其最近邻的距离（嵌入已 L2 归一化），纵轴是累积占比。
曲线越靠左 = 近邻越近 = 冗余越高。自采数据的曲线应当远在另两个数据集左侧，
这是「冗余感知选帧在本数据上有发挥空间」这一前提的直接证据。

图 4-4 要说明什么
-----------------
四个难度维度在三个数据集上的**平均触发强度**。
同一维度横着看，可看出它是否随「接近低空无人机视角」而变化；
竖着看，可看出每个数据集上哪些维度是死的（几乎不触发）。
⚠️ 这里统一用维度均值作为活跃度，不混入与误差的相关系数 ——
两者量纲不同，放进同一张热图会误导。

运行
----
    python -m src.plot_redundancy

输出 runs/fig4_3_redundancy.{png,pdf}、runs/fig4_4_difficulty.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import list_samples
from .diagnose_redundancy import nn_distance_stats
from .selection import extract_features

OUT_DIR = Path("runs")

# (显示名, 数据根目录, split, 颜色)。顺序按「离低空无人机视角由远到近」排。
DATASETS = [
    ("LoveDA (satellite)",    "data/processed/loveda_small", "train", "#7f7f7f"),
    ("RivAIrSet (UAV)",       "data/processed/rivair",       "train", "#1f77b4"),
    ("Ours (low-alt. UAV)",   "data/processed/field",        "train", "#c0392b"),
]

DIM_NAMES = ["Glare", "Shadow", "Flatness", "Blue-green"]

# 近重复的两个阈值，与 diagnose_redundancy 里报告的口径一致
THRESHOLDS = [0.05, 0.10]


def collect(root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """提特征并返回（逐样本最近邻距离, 四维难度指标矩阵）。"""
    stems = list_samples(root, split)
    embed, diffs, stems = extract_features(root, split, stems)
    nn = nn_distance_stats(embed)
    verify_against_json(root, split, nn, len(stems))
    return nn["raw"], diffs


def verify_against_json(root: str, split: str, nn: dict, n_now: int) -> None:
    """与已存盘的诊断 JSON 对账，分两种情况处理。

    1. **样本数相同而中位数不同** —— 说明特征提取或图像本身变过了。
       此时论文表里的旧数字不再可信，直接抛异常，必须人工查清。
    2. **样本数本就不同** —— 口径不同，不是错误。
       RivAIrSet 的诊断 JSON 是在该数据集尚未划分时跑的（n=231，即全部帧），
       而现在 train 只有 138 张。两个数字各自都对，但样本集不同、不可比。
       本图统一用各数据集的 **train split**（即选帧方法实际面对的候选池），
       故此处只打印提示，图与正文一律采用本次重算值。
    """
    fp = Path(root) / f"redundancy_{split}.json"
    if not fp.exists():
        print(f"  [i] {fp} 不存在，跳过对账")
        return
    old = json.loads(fp.read_text(encoding="utf-8"))
    old_med, old_n = old["nn_distance"]["median"], old["n_samples"]
    new = nn["median"]

    if old_n != n_now:
        print(f"  [i] {root}: 诊断 JSON 是在 n={old_n} 时跑的，现 train split n={n_now}，"
              f"口径不同（中位数 {old_med:.4f} → {new:.4f}）；本图用重算值")
        return
    if abs(old_med - new) > 1e-6:
        raise SystemExit(
            f"对账失败：{root} 样本数未变（n={old_n}），但最近邻距离中位数"
            f"由 {old_med:.6f} 变为 {new:.6f}。这说明数据或特征提取变过了，须先查清。"
        )
    print(f"  [v] {root}: n={n_now}，中位数 {new:.4f}，与 {fp.name} 一致")


def plot_cdf(data: dict[str, np.ndarray]) -> None:
    """图 4-3：最近邻距离的累积分布。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for name, root, split, color in DATASETS:
        d = np.sort(data[name])
        # 经验 CDF：第 i 个样本（从 1 计）对应累积占比 i/n
        y = np.arange(1, len(d) + 1) / len(d)
        ax.step(d, y, where="post", color=color, lw=1.9,
                label=f"{name}, n={len(d)}  (median {np.median(d):.3f})")

    for thr in THRESHOLDS:
        ax.axvline(thr, color="0.6", ls=":", lw=1.0, zorder=1)
        ax.text(thr, 1.015, f"{thr:g}", fontsize=7.5, color="0.45", ha="center")

    # 在两个阈值处标出各数据集的近重复占比 —— 这是正文引用的数字
    for name, root, split, color in DATASETS:
        d = data[name]
        frac = float((d < THRESHOLDS[0]).mean())
        # LoveDA 的占比只有 1.5%，标在曲线上会贴到横轴上读不出来，故单独上抬
        dy = 7 if frac < 0.05 else -2
        ax.annotate(f"{frac:.1%}", xy=(THRESHOLDS[0], frac),
                    xytext=(8, dy), textcoords="offset points",
                    fontsize=7.5, color=color)

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distance to nearest neighbour (L2, normalised embedding)")
    ax.set_ylabel("Cumulative fraction of samples")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Redundancy of the candidate pool: our continuous low-altitude\n"
                 "footage has far closer neighbours than either public dataset",
                 fontsize=9.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fp = OUT_DIR / f"fig4_3_redundancy.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")

    # 打印正文 4.6.1 表格所需的全部数字，三数据集同一口径（各自的 train split）
    print("\n冗余度（各数据集的 train split，口径一致）：")
    print(f"  {'dataset':<24}{'n':>6}{'median':>10}{'<0.05':>9}{'<0.10':>9}")
    for name, root, split, _c in DATASETS:
        d = data[name]
        print(f"  {name:<24}{len(d):>6}{np.median(d):>10.4f}"
              f"{(d < 0.05).mean():>9.1%}{(d < 0.10).mean():>9.1%}")


def plot_heatmap(diffs: dict[str, np.ndarray]) -> None:
    """图 4-4：四个难度维度在三数据集上的平均触发强度。"""
    names = [d[0] for d in DATASETS]
    mat = np.array([diffs[n].mean(axis=0) for n in names])      # (3 数据集, 4 维度)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    # 色标取**对数**：各维度的触发强度跨了 0.005~0.185 共一个半数量级，
    # 线性色标下蓝绿那一格（0.185）会把色阶占满，
    # 反光维度 0.008→0.018→0.042 的五倍梯度（本图的主要论点）就看不出颜色差了。
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=0.004, vmax=0.2))

    ax.set_xticks(range(len(DIM_NAMES)))
    ax.set_xticklabels(DIM_NAMES, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)

    # 每格写上数值；格子深时字改白色，否则看不清。
    # 阈值按对数色标判断，不能再用 mat.max()*0.55 那种线性判断。
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.09 else "0.15")

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean activation (log scale)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title("Difficulty priors are domain-dependent: a dimension that is\n"
                 "informative in one domain can be inert in another",
                 fontsize=9.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fp = OUT_DIR / f"fig4_4_difficulty.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")

    # 顺带把数值打印出来，便于直接抄进正文表格
    print("\n难度维度平均触发强度（行=数据集，列=维度）：")
    print("  " + " " * 24 + "".join(f"{d:>12}" for d in DIM_NAMES))
    for n, row in zip(names, mat):
        print(f"  {n:<24}" + "".join(f"{v:>12.4f}" for v in row))


def main() -> None:
    dists: dict[str, np.ndarray] = {}
    diffs: dict[str, np.ndarray] = {}
    for name, root, split, _color in DATASETS:
        print(f"正在处理 {name} ({root}/{split}) ...")
        dists[name], diffs[name] = collect(root, split)

    plot_cdf(dists)
    plot_heatmap(diffs)


if __name__ == "__main__":
    main()
