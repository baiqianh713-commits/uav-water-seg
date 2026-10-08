"""画论文图 4-2：标注质检机制示意（三栏）。

⚠️ 一个必须说明的限制
---------------------
本图**不是**当时那批错误标注的原始截图。首批 9 帧中 6 帧的空标注已在
10-01 全部修正（现 341/341 帧、0 空标注），错误版 JSON 未留备份、
仓库当时也还没有提交记录，原始错误状态已不可复现。

因此本图呈现的是**质检机制本身**，而非错误的原始记录：

  (a) 原图
  (b) 质检用的「类水区域」检测结果 —— 这是 `qc_overlay.water_like_ratio`
      的真实输出，占比数字是实测值。空标注之所以能被抓住，
      就是因为这个占比远超 15% 的告警阈值。
  (c) 现行（已修正）标注的叠加图，并框出芦苇线处的阴影带 ——
      即 4.4 节所述「边界画在芦苇秆根部、阴影带计为水」这一口径的落点。

论文中引用的「6/9 帧空标注」「边界内偏 4.1%/4.0%/3.8%」是当时的实测记录
（见 docs/experiment-log.md 10-01），本图不重新生成这些数字，也不以
重建的示例充当原始证据。

运行
----
    python -m src.plot_qc [--stem <样本名>]

输出 runs/fig4_2_qc.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .qc_overlay import water_like_ratio

ANNOT_DIR = Path("data/annotate")
OUT_DIR = Path("runs")

# 默认用这一帧：水体占比高、芦苇线清晰，两类错误的发生位置都能看到。
DEFAULT_STEM = "A_50m_normal__DJI_20260929163433_0281_D__f000900"

# 告警阈值，与 qc_overlay 里一致：无多边形而类水占比超此值即报警
QC_THRESHOLD = 0.15


def water_like_mask(img: np.ndarray) -> np.ndarray:
    """返回「类水区域」的布尔掩膜。

    判据与 qc_overlay.water_like_ratio 完全相同（那个函数只返回占比、
    不返回掩膜，所以这里把同样的三条排除规则写出来以便可视化）：
      - reed   : 绿通道明显高于蓝/红 → 植被
      - deck   : 红明显高于蓝 → 木栈道
      - bright : 任一通道过亮 → 过曝路面或反光白斑
    三者之外的灰绿低饱和区域算作「类水」。
    这不是分割器，宁可高估 —— 它只用于提醒「这帧你可能忘了标」。
    """
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    reed = (g - np.maximum(b, r)) > 12
    deck = (r - b) > 18
    bright = img.max(axis=2).astype(np.int16) > 235
    return ~reed & ~deck & ~bright


def load_mask(stem: str, shape: tuple[int, int]) -> np.ndarray:
    """把 AnyLabeling 的多边形标注栅格化成二值掩膜。"""
    data = json.loads((ANNOT_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    mask = np.zeros(shape, dtype=np.uint8)
    for shp in data.get("shapes", []):
        pts = np.asarray(shp["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default=DEFAULT_STEM, help="用哪一帧出图")
    args = ap.parse_args()

    img_bgr = cv2.imread(str(ANNOT_DIR / f"{args.stem}.png"))
    if img_bgr is None:
        raise SystemExit(f"读不到 {ANNOT_DIR / (args.stem + '.png')}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    wl = water_like_mask(img_bgr)           # 注意：判据按 BGR 通道写的
    ratio = water_like_ratio(img_bgr)
    mask = load_mask(args.stem, (h, w))
    water_frac = float(mask.mean())

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))

    # ---- (a) 原图
    axes[0].imshow(img)
    axes[0].set_title("(a) Frame as annotated", fontsize=9.5)

    # ---- (b) 质检的类水检测
    # 用洋红叠加，与 (c) 的青色区分开，避免读者把两栏当成同一件事
    vis = img.copy()
    vis[wl] = (vis[wl] * 0.45 + np.array([255, 0, 170]) * 0.55).astype(np.uint8)
    axes[1].imshow(vis)
    axes[1].set_title(f"(b) QC water-like detector: {ratio:.1%} of frame\n"
                      f"(alarm if no polygon and ratio > {QC_THRESHOLD:.0%})",
                      fontsize=9.5)

    # ---- (c) 现行标注 + 芦苇线局部
    vis2 = img.copy()
    m = mask.astype(bool)
    vis2[m] = (vis2[m] * 0.65 + np.array([0, 255, 255]) * 0.35).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis2, contours, -1, (255, 255, 0), 3)
    axes[2].imshow(vis2)
    axes[2].set_title(f"(c) Current (corrected) annotation: "
                      f"water = {water_frac:.1%}\n"
                      "boundary drawn at the reed stem base", fontsize=9.5)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Annotation quality control. Panel (b) is the actual output of the "
                 "detector that caught 6 silently-empty annotations among the first 9 "
                 "frames;\nthose files have since been corrected, so this figure "
                 "illustrates the mechanism rather than reproducing the original errors.",
                 fontsize=8.5, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    for ext in ("png", "pdf"):
        fp = OUT_DIR / f"fig4_2_qc.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")
    print(f"\n用帧：{args.stem}")
    print(f"  类水占比（质检判据）= {ratio:.4f}")
    print(f"  现行标注水体占比     = {water_frac:.4f}")


if __name__ == "__main__":
    main()
