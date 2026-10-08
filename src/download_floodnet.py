"""下载 FloodNet 数据集（无人机低空实拍），带宽优化 + 官方划分。

数据来源
--------
FloodNet-Challenge (Rahnemoonfar et al., IEEE Access 2021, doi:10.1109/ACCESS.2021.3090981)
本脚本用的是 ReDNetPlus 仓库的 Google Drive 镜像，因为它**标签完整**：
  train 1445 / val 450 / test 448，影像与标签 100% 配对，官方划分现成。
（官方 Track1 链接是半监督设定，部分影像无标签。）

许可：CDLA-Permissive-1.0，要求引用原文并保留来源链接。
**论文中必须引用 Rahnemoonfar et al. 2021。**

为什么这个数据集适合本研究
--------------------------
- **真无人机低空实拍**（DJI Mavic Pro，约 200ft），4000×3000，GSD 约 1.5cm
  —— 与 LoveDA（0.3m 卫星）完全不同，镜面反光等低空特征真实存在
- 标签含 water(5) 与 pool(8)，可做二值水体分割
- 灾后同区域连续航拍，**帧间冗余度远高于 LoveDA**，
  这正是冗余感知选帧方法所需的场景

带宽优化
--------
影像约 5MB/张，标签仅 16KB —— 差约 300 倍。所以：
  1. 先下载**全部标签**（很快），算出每张图的水体占比
  2. 再按占比分层挑选子集，**只下载选中的影像**
这与 download_loveda.py 是同一策略。

用法
----
  # 先看结构，不下载
  python -m src.download_floodnet --list_only

  # 下载子集（标签全下，影像按需）
  python -m src.download_floodnet --out data/raw/floodnet --n_train 300 --n_val 100 --n_test 100
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

FOLDER_ID = "1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD"

# FloodNet 的 10 个类别。已实测确认 water = 5。
FLOODNET_CLASSES = {
    0: "background", 1: "building-flooded", 2: "building-non-flooded",
    3: "road-flooded", 4: "road-non-flooded", 5: "water",
    6: "tree", 7: "vehicle", 8: "pool", 9: "grass",
}
# 默认只取自然水体 + 泳池。是否把 flooded road/building (1,3) 计入水体，
# 取决于论文对"水体"的定义 —— 水利巡检通常关注常态水体。
# 这个选择必须在论文里写明。
WATER_IDS = [5, 8]

SPLIT_DIRS = {
    "train": ("train/train-org-img", "train/train-label-img"),
    "val": ("val/val-org-img", "val/val-label-img"),
    "test": ("test/test-org-img", "test/test-label-img"),
}


def list_remote() -> list[tuple[str, str]]:
    """列出 Drive 文件夹内容（不下载）。

    返回 [(归一化路径, file_id), ...]。
    gdown 返回的文件对象是不可变的 namedtuple，所以不原地改 path，
    而是在这里就把 Windows 的反斜杠统一成正斜杠后输出元组。
    """
    import gdown
    files = gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True)
    if not files:
        raise SystemExit(
            "无法列出 Google Drive 文件夹。可能是网络问题或 Drive 限流，稍后重试。"
        )
    return [(f.path.replace("\\", "/"), f.id) for f in files]


def download_batch(items: list[tuple[str, str]], workers: int, desc: str) -> list[str]:
    """并行下载。items 为 [(file_id, 本地路径), ...]，返回成功的 stem 列表。

    用线程池：瓶颈是网络 I/O 而非 CPU。
    已存在且非空的文件会跳过，因此中断后重跑可续传。
    """
    import gdown

    ok: list[str] = []

    def work(item: tuple[str, str]) -> tuple[str, bool]:
        fid, out = item
        p = Path(out)
        stem = p.stem
        if p.exists() and p.stat().st_size > 0:
            return stem, True
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Drive 偶发限流，失败重试两次
            for attempt in range(3):
                try:
                    gdown.download(id=fid, output=out, quiet=True)
                    if p.exists() and p.stat().st_size > 0:
                        return stem, True
                except Exception:
                    if attempt == 2:
                        return stem, False
            return stem, False
        except Exception:
            return stem, False

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, it) for it in items]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
            stem, success = fut.result()
            if success:
                ok.append(stem)

    return sorted(ok)


def water_ratio(mask_path: Path, water_ids: list[int]) -> float:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return -1.0
    if m.ndim == 3:
        m = m[..., 0]
    return float(np.isin(m, water_ids).mean())


def stratified_pick(ratios: dict[str, float], n: int, seed: int = 0) -> list[str]:
    """按水体占比分层挑选，保证正负样本都有代表性。

    ⚠️ 注意：这里用了标签信息，只用于**构造实验数据集**，
    不是论文里的选帧策略。选帧必须在无标注前提下进行（见 selection.py）。
    """
    rng = np.random.default_rng(seed)
    items = [(k, v) for k, v in ratios.items() if v >= 0]
    if n >= len(items):
        return sorted(k for k, _ in items)

    bands = {
        "none": [k for k, v in items if v < 1e-6],
        "low": [k for k, v in items if 1e-6 <= v < 0.05],
        "mid": [k for k, v in items if 0.05 <= v < 0.25],
        "high": [k for k, v in items if v >= 0.25],
    }
    print("  水体占比分层:")
    for name, mem in bands.items():
        print(f"    {name:<6} {len(mem):>5} 张")

    quota = {"none": int(n * 0.15), "low": int(n * 0.25), "mid": int(n * 0.35)}
    quota["high"] = n - sum(quota.values())

    picked: list[str] = []
    shortfall = 0
    for name, want in quota.items():
        mem = bands[name]
        take = min(want, len(mem))
        shortfall += want - take
        if take:
            picked.extend(rng.choice(mem, size=take, replace=False).tolist())

    if shortfall > 0:
        rest = [k for k, _ in items if k not in set(picked)]
        if rest:
            picked.extend(
                rng.choice(rest, size=min(shortfall, len(rest)), replace=False).tolist()
            )

    return sorted(set(picked))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="data/raw/floodnet")
    ap.add_argument("--n_train", type=int, default=300, help="-1 表示全部")
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--water_ids", type=int, nargs="+", default=None,
                    help=f"水体类别索引，默认 {WATER_IDS}（water + pool）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list_only", action="store_true", help="只列结构，不下载")
    args = ap.parse_args()

    water_ids = args.water_ids or WATER_IDS
    out = Path(args.out)

    print("正在列出远端文件（首次较慢）...")
    files = list_remote()
    print(f"远端共 {len(files)} 个文件")

    # 按 split 归类，建立 stem -> {"img": file_id, "lab": file_id} 索引
    index: dict[str, dict[str, dict]] = {s: {} for s in SPLIT_DIRS}
    for path, fid in files:
        d = os.path.dirname(path)
        base = os.path.basename(path)
        for split, (img_dir, lab_dir) in SPLIT_DIRS.items():
            if d == img_dir:
                stem = os.path.splitext(base)[0]
                index[split].setdefault(stem, {})["img"] = fid
            elif d == lab_dir and base.endswith("_lab.png"):
                stem = base[: -len("_lab.png")]
                index[split].setdefault(stem, {})["lab"] = fid

    print("\n各 split 配对情况:")
    for split in SPLIT_DIRS:
        paired = sum(1 for v in index[split].values() if "img" in v and "lab" in v)
        print(f"  {split:<6} 共 {len(index[split]):>5} stem，完整配对 {paired:>5}")

    if args.list_only:
        print("\n(--list_only 模式，未下载)")
        return

    wants = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    stats = {}

    for split, n_want in wants.items():
        if n_want == 0:
            continue
        pairs = {k: v for k, v in index[split].items() if "img" in v and "lab" in v}
        if not pairs:
            print(f"[!] {split} 无完整配对，跳过")
            continue

        print(f"\n{'=' * 60}\n{split}: 池 {len(pairs)}，目标 {n_want if n_want > 0 else '全部'}\n{'=' * 60}")

        # --- 第一步：下载全部标签（小文件，顺带得到全量统计）---
        mask_dir = out / split / "masks"
        got = download_batch(
            [(v["lab"], str(mask_dir / f"{k}.png")) for k, v in pairs.items()],
            args.workers, f"{split} 标签",
        )
        print(f"标签成功 {len(got)}/{len(pairs)}")
        if not got:
            continue

        # --- 第二步：算水体占比并分层挑选 ---
        ratios = {s: water_ratio(mask_dir / f"{s}.png", water_ids) for s in got}
        valid = np.array([v for v in ratios.values() if v >= 0])
        print(f"全量水体占比: 均值={valid.mean():.4f} 中位={np.median(valid):.4f} "
              f"最大={valid.max():.4f} 无水={int((valid < 1e-6).sum())}/{len(valid)}")

        picked = (sorted(got) if n_want < 0
                  else stratified_pick(ratios, n_want, seed=args.seed))
        print(f"挑选 {len(picked)} 张")

        # --- 第三步：只下选中的影像 ---
        img_dir = out / split / "images"
        got_img = download_batch(
            [(pairs[s]["img"], str(img_dir / f"{s}.jpg")) for s in picked],
            args.workers, f"{split} 影像",
        )
        print(f"影像成功 {len(got_img)}/{len(picked)}")

        # 清理未配对的标签，保证 images/masks 一一对应
        keep = set(got_img)
        removed = 0
        for p in mask_dir.glob("*.png"):
            if p.stem not in keep:
                p.unlink()
                removed += 1
        print(f"清理未配对标签 {removed} 个，最终 {len(keep)} 对")

        arr = np.array([ratios[s] for s in got_img if s in ratios])
        stats[split] = {
            "n_pairs": len(keep),
            "population_size": len(pairs),
            "population_water_ratio_mean": float(valid.mean()),
            "subset_water_ratio_mean": float(arr.mean()) if len(arr) else 0.0,
            "subset_no_water": int((arr < 1e-6).sum()) if len(arr) else 0,
        }

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "download_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": "FloodNet via ReDNetPlus Google Drive mirror",
            "folder_id": FOLDER_ID,
            "citation": "Rahnemoonfar et al., FloodNet: A High Resolution Aerial "
                        "Imagery Dataset for Post Flood Scene Understanding, "
                        "IEEE Access 9:89644-89654, 2021. doi:10.1109/ACCESS.2021.3090981",
            "license": "CDLA-Permissive-1.0 (需引用原文)",
            "water_ids": water_ids,
            "class_map": FLOODNET_CLASSES,
            "splits": stats,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 完成 -> {out / 'download_meta.json'}")
    print("注意：影像为 .jpg，标签为 .png。转换时用 prepare_data.py generic。")
    print("论文中必须引用 Rahnemoonfar et al. 2021（CDLA-Permissive 许可要求）。")


if __name__ == "__main__":
    main()
