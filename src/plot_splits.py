"""画论文图 4-1：防泄漏数据划分示意图。

图要说明什么
------------
本数据只有 5 个架次，且条件 C 只有 1 个架次 —— 按架次随机划分会让整个 C 组
落进单一集合。所以改成**架次内按时间序切分**：每个架次沿时间轴切成
train / val / test 三段，切点前后各丢 3 个采样帧作**缓冲带**（图中灰色斜纹）。

缓冲带是防泄漏的关键：相邻采样帧间隔仅 6 秒、飞行速度 0.45~0.74 m/s，
画面重叠超过 98%，切点处若不留缓冲，train 的最后一帧和 test 的第一帧
几乎是同一张图，测试集会被污染。

横轴用「架次内的采样序号」而非原始视频帧号：采样步长恒为 180 帧（6 秒），
两者成正比，但序号读起来直观。

运行
----
    python -m src.plot_splits

输出 runs/fig4_1_splits.{png,pdf}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SPLITS_JSON = Path("data/processed/field_splits.json")
OUT_DIR = Path("runs")

# 四类的配色。drop（缓冲带）用灰色加斜纹，视觉上明确区别于三个正式集合。
SPLIT_STYLE = {
    "train": dict(color="#3a6ea5", hatch=None,  label="train (227)"),
    "val":   dict(color="#e8a33d", hatch=None,  label="val (29)"),
    "test":  dict(color="#c0392b", hatch=None,  label="test (85)"),
    "drop":  dict(color="#cfcfcf", hatch="///", label="buffer, discarded (60)"),
}
ORDER = ["train", "val", "test", "drop"]


def load_lanes() -> dict[tuple[str, str], list[str]]:
    """读划分文件，整理成 {(条件, 架次): [按时间排序的 split 标签]}。

    key 的格式是 `{条件}__{架次}__f{原始视频帧号}`，
    按帧号排序即为时间序。
    """
    raw = json.loads(SPLITS_JSON.read_text(encoding="utf-8"))
    tmp: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for key, split in raw.items():
        cond, flight, frame = key.split("__")
        tmp[(cond, flight)].append((int(frame.lstrip("f")), split))
    return {k: [s for _, s in sorted(v)] for k, v in sorted(tmp.items())}


def runs_of(labels: list[str]) -> list[tuple[int, int, str]]:
    """把标签序列压成连续区段 [(起始序号, 长度, 标签), ...]。

    这样每个区段画一个矩形，而不是逐帧画 401 个小方块。
    顺带也验证了划分确实是「连续分段」而不是逐帧随机 ——
    若某个架次被压出很多细碎区段，图上立刻能看出来。
    """
    out: list[tuple[int, int, str]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i - start, labels[start]))
            start = i
    return out


def main() -> None:
    lanes = load_lanes()
    fig, ax = plt.subplots(figsize=(8.6, 3.5))

    yticklabels = []
    for row, ((cond, flight), labels) in enumerate(lanes.items()):
        y = len(lanes) - 1 - row          # 第一个架次画在最上面
        for start, length, split in runs_of(labels):
            st = SPLIT_STYLE[split]
            ax.barh(y, length, left=start, height=0.62,
                    color=st["color"], hatch=st["hatch"],
                    edgecolor="white", linewidth=0.6, zorder=3)
        # 架次号取文件名里的 4 位序号（DJI_..._0281_D -> 0281）
        flight_id = flight.split("_")[-2]
        yticklabels.append(f"{cond.split('_')[0]}  /  flight {flight_id}")
        # 行末标出该架次的采样帧数
        ax.text(len(labels) + 1.2, y, f"n={len(labels)}",
                va="center", fontsize=8, color="0.3")

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(list(reversed(yticklabels)), fontsize=9)
    ax.set_xlabel("Sampled frame index within flight "
                  "(chronological, 1 sample / 6 s)")
    ax.set_xlim(-1.5, max(len(v) for v in lanes.values()) + 8)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.grid(axis="x", alpha=0.2, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=SPLIT_STYLE[s]["color"], hatch=SPLIT_STYLE[s]["hatch"],
                     edgecolor="white", label=SPLIT_STYLE[s]["label"]) for s in ORDER]
    ax.legend(handles=handles, ncol=4, fontsize=8.5,
              loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    ax.set_title("Leakage-free split: each flight is cut along its own time axis,\n"
                 "with 3 samples discarded on each side of every cut point",
                 fontsize=9.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fp = OUT_DIR / f"fig4_1_splits.{ext}"
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        print(f"已写出 {fp}")


if __name__ == "__main__":
    main()
