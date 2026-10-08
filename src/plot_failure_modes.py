"""画论文图 5-3：两种迁移设置下**残余误差方向**的对比。

这张图要说明什么
----------------
同一套模型与训练流程，换一个测试域，误差的**方向**会整个翻过来：

  跨条件（常光 A+B 训练 -> 暗光 C 测试）：残差几乎全是**假阳性**，
      即把非水判成水（过度预测）。recall 0.9986 > precision 0.9873。
  跨数据集（自采 -> RivAIrSet）：误差以**漏检**为主，
      即把水判成非水。recall 0.4221 < precision 0.5542。

两个测试域的水体占比分别约 0.56 与 0.19 —— 误差方向与标签先验同向。

🔴 读这张图时的四条纪律
----------------------
1. **跨条件那一行不是「失效」。** 它收敛后 IoU 达 0.9859，与同分布基本持平。
   这里展示的是**残余误差的方向**（残差几乎全为假阳性），不是失效程度。
   只有跨数据集那一行（IoU 0.3151）才构成实际失效。
   二者可比的是**方向**，不可比的是**严重程度**。
2. **「误差方向与标签先验同向」是相关，不是因果。** 本研究未通过调整判决
   阈值、类别重加权等干预来验证因果，不得写成「因为先验低所以漏检」。
3. **示例图不是人工挑的。** 每行展示的那张图由**逐图 IoU 的中位数**自动选出
   （见 pick_example），脚本会把选中的文件名打印出来备查。
   但单张图终究只是示例 —— 所以第 5 列画的是**整个测试集**的误差构成，
   声称以第 5 列为准，示例图只负责让读者看见它长什么样。
4. **跨条件的测试集是 C 组唯一 1 个架次的 69 帧**，帧间高度相关，
   等效独立样本远少于 69。不得按 69 个独立样本做显著性声称。

权重从哪来
----------
原始两次运行开着 cleanup_checkpoints，权重已被删，故按**同一份 config.yaml**
重训了一遍（只覆盖两个清理开关），输出在 runs/fig53_*。

🔴 两次重训的复现情况**不一样**，这一点必须随图说明：

  跨条件：55 轮逐轮 loss 与原始运行**完全一致**，test IoU 0.9859 分毫不差。
  跨数据集：第 1 轮 loss 即有 4e-4 的差异（原始 0.3287 / 重训 0.3291），
      域内 val IoU 仍几乎相同（0.9892 -> 0.9891，差 1e-4），
      但**域外 test IoU 由 0.3151 掉到 0.2203**（recall 0.4221 -> 0.2681，
      precision 几乎不变）。

已查明的事实（不要写成超出这些证据的结论）：
  - 代码与数据自 10-02 起未改动，原始运行在 10-03，故非代码/数据差异；
  - 今日同环境下连续两次跑同一配置，结果**完全一致** —— 即当下环境是确定性的；
  - CPU 线程数会改变结果：固定同一套学习率调度时，8 / 14 / 18 线程给出不同的
    首轮 loss。原始运行当时 CPU 被其它实验占满，线程划分与今日不同。
  ⚠️ 未能找到可精确复现 0.3151 的线程配置，故**不声称已定位全部原因**。

这对本图意味着什么：图上标的数字一律取自**实际画出来的那套权重**（重训），
同时把原始值并列写出。本图的论点是误差**方向**，而方向在两次运行中一致
（跨数据集始终 recall < precision），故论点不受影响；但任何一个精确的
跨数据集 IoU 数值都不具备跨环境可复现性。

本脚本对两件事的处理是分开的：
  (A) 本脚本的推理结果 vs 该次运行自己的 result.json —— **不符即退出**。
      这保证图上画的像素与图上标的数字出自同一套权重。
  (B) 重训 vs 原始运行 —— **不退出，但一律打印差值**，并强制检查误差方向
      未翻转（方向若翻转，本图的论点就不成立，那时必须退出）。

运行
----
    PYTHONIOENCODING=utf-8 python -m src.plot_failure_modes

输出 runs/fig5_3_failure_modes.{png,pdf,svg}
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.data import IMAGENET_MEAN, IMAGENET_STD, WaterSegDataset
from src.models import build_model
from src.utils import load_config

OUT_DIR = Path("runs")

# 误差图的配色。用「正确=灰、过度预测=红、漏检=蓝」三色，
# 红蓝是本图唯一要读者记住的对立；真阴性留白，免得整张图糊成一片。
C_TP = "#9e9e9e"   # 正确判为水
C_FP = "#d62728"   # 过度预测（非水判成水）
C_FN = "#1f77b4"   # 漏检（水判成非水）

# 两个设置。row_label 是画在最左侧、**竖排**的行标题，所以必须短：
# 竖排文字的长度吃的是子图的高度，一行写满就会把整张图顶变形。
SETTINGS = [
    dict(key="xcond",
         rerun=Path("runs/fig53_xcond"),
         original=Path("runs/field_xcond_conv_s1200"),
         row_label="(1) Cross-condition:  train on daylight (A+B)  $\\rightarrow$  test on low-light (C)"),
    dict(key="x2rivair",
         rerun=Path("runs/fig53_x2rivair"),
         original=Path("runs/field2rivair_conv"),
         row_label="(2) Cross-dataset:  train on own survey  $\\rightarrow$  test on RivAIrSet"),
]


# ---------------------------------------------------------------- 推理

def load_model(run_dir: Path, cfg: dict) -> torch.nn.Module:
    """载入该次运行**最后一个 epoch** 的权重。

    🔴 必须是 last.pt 而不是 best.pt：两次运行都是 select_by_val=False，
    正文报告的测试指标就是用最后一个 epoch 的模型算的。拿 best.pt 画图
    会得到一个与正文数字不对应的模型。
    """
    ckpt_path = run_dir / "last.pt"
    if not ckpt_path.exists():
        raise SystemExit(
            f"缺 {ckpt_path}。该次重训要用 train.cleanup_checkpoints=false "
            f"train.cleanup_last=false 才会留下权重。")
    model = build_model(cfg["model"])
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def infer_all(model: torch.nn.Module, cfg: dict) -> tuple[list[dict], dict]:
    """在整个测试集上推理，返回 (逐图记录, 聚合统计)。

    预处理与 train.py 的测试路径保持一致：整张图、不裁剪、不增强、阈值 0.5。
    任何一处不一致，算出来的指标就对不上 result.json，核对那一步会拦下来。
    """
    root = cfg["data"]["root"]
    split = cfg["data"].get("test_split", "test")
    thr = cfg["train"].get("threshold", 0.5)

    ds = WaterSegDataset(root, split, augment=False, crop_size=None)
    # batch_size=1：各测试图尺寸不一定相同，逐张跑最稳，也便于逐图记账
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    records: list[dict] = []
    TP = FP = FN = TN = 0
    for i, batch in enumerate(loader):
        logits = model(batch["image"])
        prob = torch.sigmoid(logits.float())[0, 0].numpy()
        pred = prob > thr
        gt = batch["mask"].numpy()[0, 0].astype(bool)

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt).sum())
        fn = int(np.logical_and(~pred, gt).sum())
        tn = int(np.logical_and(~pred, ~gt).sum())
        TP, FP, FN, TN = TP + tp, FP + fp, FN + fn, TN + tn

        denom = tp + fp + fn
        records.append(dict(stem=ds.stems[i], tp=tp, fp=fp, fn=fn, tn=tn,
                            iou=1.0 if denom == 0 else tp / denom,
                            water_frac=float(gt.mean())))

    eps = 1e-9
    agg = dict(iou=TP / (TP + FP + FN + eps),
               precision=TP / (TP + FP + eps),
               recall=TP / (TP + FN + eps),
               n_images=len(records),
               tp=TP, fp=FP, fn=FN,
               water_frac=float(np.mean([r["water_frac"] for r in records])))
    return records, agg


def verify(agg: dict, original: Path, rerun: Path) -> dict:
    """两件事分开核：(A) 图内自洽，硬性；(B) 与原始运行的差异，如实报告。

    返回原始运行的 result.json，供图上并列标注。
    容差 2e-3 允许 CPU 浮点累加顺序带来的末位差异，但拦得住
    「载错权重 / 用错 split / 阈值不同」这类真错误。
    """
    ref = json.loads((original / "result.json").read_text(encoding="utf-8"))
    mine = json.loads((rerun / "result.json").read_text(encoding="utf-8"))

    # ---- (A) 本脚本的推理 vs 该次运行自己的 result.json：必须一致 ----
    # 这验证的是本脚本的预处理/阈值/split 与 train.py 的测试路径完全相同。
    # 这一条不过，图上画的像素与图上标的数字就不是一回事，必须退出。
    for key, my_val in (("test_iou", agg["iou"]),
                        ("test_precision", agg["precision"]),
                        ("test_recall", agg["recall"])):
        if abs(mine[key] - my_val) > 2e-3:
            raise SystemExit(
                f"{rerun.name}: 本脚本算的 {key}={my_val:.4f} 与该次运行 "
                f"result.json 的 {mine[key]:.4f} 不一致，说明推理路径有差异。")
    if agg["n_images"] != mine["test_n_images"]:
        raise SystemExit(f"{rerun.name}: 测试图张数 {agg['n_images']} != {mine['test_n_images']}")

    # ---- (B) 重训 vs 原始运行：如实报告，不掩盖，但不因此停工 ----
    # 本图的论点是误差**方向**。方向若在两次运行间翻转，论点就不成立 ——
    # 那种情况必须退出；只是数值漂移则照常出图，并把两个值都标在图上。
    dir_ref = "FN" if ref["test_recall"] < ref["test_precision"] else "FP"
    dir_now = "FN" if mine["test_recall"] < mine["test_precision"] else "FP"
    if dir_ref != dir_now:
        raise SystemExit(
            f"{rerun.name}: 误差方向在重训后翻转（原始以 {dir_ref} 为主，"
            f"重训以 {dir_now} 为主）。本图的论点正是方向，方向不稳则不能出图。")
    return ref


def pick_example(records: list[dict]) -> dict:
    """选展示用的那张图：**逐图 IoU 的中位数**那一张。

    为什么要定死规则：挑最好看的那张会把图变成广告，挑最差的会夸大失效。
    中位数是唯一一个不需要解释的选择。偶数张时取排序后靠左的那张
    （numpy 的 median 会插值出一个不存在的值，所以这里按下标取）。
    """
    order = sorted(records, key=lambda r: r["iou"])
    return order[(len(order) - 1) // 2]


# ---------------------------------------------------------------- 画图

def denorm(img_t: torch.Tensor) -> np.ndarray:
    """把 Dataset 归一化过的张量还原成可显示的 RGB。"""
    arr = img_t.numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(arr, 0, 1)


def load_example(cfg: dict, stem: str, model: torch.nn.Module):
    """把选中的那张图重新跑一遍，拿到可画的 RGB / GT / 预测。"""
    root, split = cfg["data"]["root"], cfg["data"].get("test_split", "test")
    ds = WaterSegDataset(root, split, stems=[stem], augment=False, crop_size=None)
    sample = ds[0]
    with torch.no_grad():
        logits = model(sample["image"][None])
    prob = torch.sigmoid(logits.float())[0, 0].numpy()
    pred = prob > cfg["train"].get("threshold", 0.5)
    gt = sample["mask"].numpy()[0].astype(bool)
    return denorm(sample["image"]), gt, pred


def error_rgb(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """三色误差图：灰 = 命中，红 = 过度预测，蓝 = 漏检，白 = 正确的非水。"""
    out = np.ones((*gt.shape, 3))
    for mask, color in ((np.logical_and(pred, gt), C_TP),
                        (np.logical_and(pred, ~gt), C_FP),
                        (np.logical_and(~pred, gt), C_FN)):
        out[mask] = matplotlib.colors.to_rgb(color)
    return out


def draw_row(axes, cfg, model, rec, agg, row_label: str, ref: dict,
             *, show_col_titles: bool) -> None:
    """画一行：输入 / 真值 / 预测 / 误差图 / 全测试集误差构成。

    🔴 排版的硬约束是**印出来的字号**，不是屏幕上好不好看。
    这张图在论文里按页宽排（约 6.3 in），所以画布宽度也必须定在这个量级 ——
    画成 13 in 再缩到页宽，等于把所有字号砍掉一半，9 pt 的列标题印出来只有
    4 pt，谁也读不了。所以这里画布只有 7.2 in，字号写的就是印出来的字号。

    由此带来的第二个约束：每格只有约 1.3 in 宽，塞不下任何多行的数字块。
    故全测试集的统计数字不挂在第 5 列下面，而是**横跨整行宽度**写在行下方，
    那里有 6.9 in 可用，等宽字体能排下 100 多个字符。
    """
    rgb, gt, pred = load_example(cfg, rec["stem"], model)

    for col, (data, title) in enumerate((
        (rgb, "Input"),
        (gt, "Ground truth\n(white = water)"),
        (pred, "Prediction"),
        (error_rgb(gt, pred), "Error map"),
    )):
        ax = axes[col]
        ax.imshow(data, cmap=None if data.ndim == 3 else "gray",
                  vmin=None if data.ndim == 3 else 0, vmax=None if data.ndim == 3 else 1)
        if show_col_titles:
            ax.set_title(title, fontsize=7.5, linespacing=1.25)
        ax.set_xticks([]); ax.set_yticks([])

    # 行标题：**横排**写在整行的正上方，左对齐到 Input panel 的左边。
    # 不走 ylabel —— ylabel 是竖排的，竖排文字的长度吃的是子图高度，而每格
    # 只有约 0.73 in 高，这句话一定会被截断。
    axes[0].annotate(row_label, xy=(0, 1.0), xycoords="axes fraction",
                     textcoords="offset points",
                     xytext=(0, 24 if show_col_titles else 5),
                     ha="left", va="bottom", fontsize=8.5, fontweight="bold",
                     annotation_clip=False)

    # 第 5 列：整个测试集的误差构成。示例图只是示例，声称以这一列为准。
    ax = axes[4]
    fp_share = agg["fp"] / (agg["fp"] + agg["fn"] + 1e-9)
    ax.barh([0], [fp_share * 100], color=C_FP, height=1.0)
    ax.barh([0], [(1 - fp_share) * 100], left=[fp_share * 100], color=C_FN, height=1.0)
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([]); ax.set_xticks([0, 50, 100])
    ax.tick_params(labelsize=6.5, pad=1.5)
    ax.set_box_aspect(0.26)          # 压扁成一根横条，不要占满整个格子
    ax.axvline(50, color="0.3", lw=0.7, ls=":")
    if show_col_titles:
        ax.set_title("Error composition,\nwhole test set", fontsize=7.5, linespacing=1.25)
    ax.set_xlabel("share of all\nerror pixels (%)", fontsize=6.5, labelpad=1.5,
                  linespacing=1.2)
    for x, v in ((fp_share * 100 / 2, fp_share),
                 (fp_share * 100 + (1 - fp_share) * 100 / 2, 1 - fp_share)):
        ax.text(x, 0, f"{v*100:.0f}%", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

    # 行下方的数字块：横跨整行，等宽字体对齐。
    # 数字一律取自**实际画出来的这套权重**；若与原始运行有差异，把原始值并列
    # 写出来，不藏起来（跨数据集一侧确有差异，见模块说明）。
    lines = [f"n={agg['n_images']} images    test-set water fraction {agg['water_frac']:.2f}"
             f"    IoU {agg['iou']:.4f}   P {agg['precision']:.4f}   R {agg['recall']:.4f}",
             f"median-IoU example shown (selected automatically, not hand-picked): "
             f"{rec['stem']}, IoU {rec['iou']:.3f}"]
    if abs(ref["test_iou"] - agg["iou"]) > 2e-3:
        lines.append(f"original run: IoU {ref['test_iou']:.4f}   P {ref['test_precision']:.4f}   "
                     f"R {ref['test_recall']:.4f}  -  same config and seed, "
                     f"not reproducible in this environment")
    axes[0].annotate("\n".join(lines), xy=(0, 0), xycoords="axes fraction",
                     textcoords="offset points", xytext=(0, -8),
                     ha="left", va="top", fontsize=6.4, family="monospace",
                     linespacing=1.45, annotation_clip=False)


def main() -> None:
    # 🔴 这里**不用** tight_layout：第 5 列底下挂着一段很高的多行说明文字，
    #    tight_layout 会为了塞下它把坐标区压到负宽度，然后放弃排版并警告
    #    （上一版就是这样，第 2 行的数字被切成了碎片）。
    #    改成手工 subplots_adjust，把每一块的位置写死，所见即所得。
    #    画布宽度 7.2 in = 论文页宽的量级，**不要放大**：放大再缩回去就等于
    #    等比例砍掉所有字号（见 draw_row 的说明）。
    #    画布高度按图像自身的宽高比定：无人机图是 16:9，每格约 1.3 in 宽
    #    就只有约 0.73 in 高，格子给高了，imshow 会把图缩在格子中间、上下留白。
    #    下面这套边距是按 0.73 in 的行高反推的：
    #      顶部 0.46 in 放第 1 行的行标题 + 两行列标题，
    #      行间 0.52 in 放第 1 行的两行数字和第 2 行的行标题，
    #      底部 0.72 in 放第 2 行的三行数字和最下方的图例。
    # hspace 要同时容下：第 1 行下方的两行数字、第 5 列的两行轴标签、
    # 以及第 2 行的行标题。给小了第 2 行的标题会压在第 1 行的数字上。
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.55),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.95]))
    fig.subplots_adjust(left=0.022, right=0.988, top=0.872, bottom=0.255,
                        wspace=0.07, hspace=1.02)

    printed = []
    for row, s in enumerate(SETTINGS):
        cfg = load_config(s["rerun"] / "config.yaml")
        model = load_model(s["rerun"], cfg)
        records, agg = infer_all(model, cfg)
        ref = verify(agg, s["original"], s["rerun"])
        rec = pick_example(records)
        draw_row(axes[row], cfg, model, rec, agg, s["row_label"], ref,
                 show_col_titles=(row == 0))
        printed.append((s, rec, agg, ref))

    handles = [Patch(color=C_TP, label="Correct water (TP)"),
               Patch(color=C_FP, label="Over-prediction: non-water called water (FP)"),
               Patch(color=C_FN, label="Missed detection: water called non-water (FN)")]
    # 图例放最下方：放顶上会和列标题抢同一条横带（上一版就压在一起了）。
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.002),
               handlelength=1.6, columnspacing=1.6, handletextpad=0.5)
    for ext in ("png", "pdf", "svg"):
        fp = OUT_DIR / f"fig5_3_failure_modes.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")

    print("\n核心数字（请与 paper-ch5.md 5.5.4 逐个核对）")
    print("=" * 72)
    for s, rec, agg, ref in printed:
        fp_share = agg["fp"] / (agg["fp"] + agg["fn"] + 1e-9)
        print(f"  {s['key']:<9} IoU {agg['iou']:.4f}  P {agg['precision']:.4f}  "
              f"R {agg['recall']:.4f}  水体占比 {agg['water_frac']:.2f}")
        print(f"            误差构成：过度预测 {fp_share*100:.1f}% / "
              f"漏检 {(1-fp_share)*100:.1f}%   （n={agg['n_images']} 张）")
        print(f"            示例图（逐图 IoU 中位数，自动选取）：{rec['stem']}  "
              f"IoU {rec['iou']:.3f}")
        d = agg["iou"] - ref["test_iou"]
        if abs(d) > 2e-3:
            print(f"            ⚠️ 与原始运行不一致：原始 IoU {ref['test_iou']:.4f} / "
                  f"R {ref['test_recall']:.4f}，本次 {agg['iou']:.4f} / "
                  f"{agg['recall']:.4f}（ΔIoU {d:+.4f}）")
            print(f"               同配置同种子，跨环境不可复现；误差方向未变，"
                  f"故本图论点不受影响。")
        else:
            print(f"            ✅ 与原始运行一致（ΔIoU {d:+.4f}）")
    print("\n  🔴 跨条件设置 IoU 0.9859，谈不上『失效』——该行展示的是残余误差的")
    print("     方向，不是失效程度。只有跨数据集一行才构成实际失效。")
    print("  🔴 误差方向与标签先验同向，但本研究未做干预，不得写成因果。")
    print("  🔴 跨条件测试集是 C 组唯一 1 个架次的 69 帧，帧间高度相关。")
    print("  🔴 跨数据集的精确 IoU 数值不具备跨环境可复现性（见模块说明），")
    print("     正文引用该设置的数值时必须随附这一限制。")


if __name__ == "__main__":
    main()
