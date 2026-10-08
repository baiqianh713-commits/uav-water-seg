"""按**场景**划分 train/val/test，杜绝空间信息泄漏。

为什么必须这么做
----------------
切图会把一张 1024² 原图切成多个 512² 图块（重叠切图时更甚）。
若对**图块**做随机划分，同一张原图的相邻图块会同时落入训练集和测试集，
模型在测试时见到的其实是训练过的地面区域，指标会被显著虚高。

这是遥感分割论文里常见但不易察觉的方法论缺陷。本脚本按图块文件名中的
**原图名**（场景 ID）分组划分，保证同一原图的所有图块只属于一个集合。

对你的自采无人机数据，场景 ID 应当是**飞行架次或地点**，而不是原图名——
同一架次的连续航拍帧几乎相同，按帧划分同样会泄漏。用 --scene_regex 指定。

论文里应明确写出该协议，例如：
  "Splits are constructed at the scene level so that tiles cropped from the
   same source image never appear in more than one split, precluding spatial
   leakage between overlapping tiles."

用法
----
  # 图块名形如 `1234_y0x512.png`，场景 ID 取 `_y` 之前的部分（默认规则）
  python -m src.make_splits --root data/processed/loveda_small \
      --from_split train --ratios 0.6 0.2 0.2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def scene_of(stem: str, pattern: str) -> str:
    """从图块名提取场景 ID。

    默认规则 `^(.*?)_y\\d+x\\d+$` 对应 prepare_data.py 的命名 `{原图名}_y{y}x{x}`。
    不匹配时退回整个 stem（即该图块自成一个场景）。
    """
    m = re.match(pattern, stem)
    return m.group(1) if m else stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="预处理后的数据集根目录")
    ap.add_argument("--from_split", default="train", help="从哪个 split 重新划分")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    help="train val test 比例（按场景数，不是图块数）")
    ap.add_argument("--scene_regex", default=r"^(.*?)_y\d+x\d+$",
                    help="从图块名提取场景 ID 的正则，第 1 组为场景 ID")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", action="store_true",
                    help="移动文件而非复制（省磁盘，但会破坏原 split）")
    args = ap.parse_args()

    root = Path(args.root)
    src_img = root / args.from_split / "images"
    src_mask = root / args.from_split / "masks"
    if not src_img.is_dir():
        raise SystemExit(f"找不到 {src_img}")

    stems = sorted(p.stem for p in src_img.glob("*.png"))
    if not stems:
        raise SystemExit(f"{src_img} 下没有图块")

    # 按场景分组
    groups: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        groups[scene_of(s, args.scene_regex)].append(s)

    scenes = sorted(groups)
    print(f"共 {len(stems)} 个图块，来自 {len(scenes)} 个场景 "
          f"(平均每场景 {len(stems) / len(scenes):.1f} 块)")
    if len(scenes) == len(stems):
        print("[!] 每个图块自成一个场景——请确认 --scene_regex 是否匹配你的命名规则，"
              "否则起不到防泄漏作用。")

    # 划分的是**场景**，不是图块
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(scenes))
    r_tr, r_va, _ = args.ratios
    n_tr = int(round(len(scenes) * r_tr))
    n_va = int(round(len(scenes) * r_va))

    assign = {
        "train": [scenes[i] for i in order[:n_tr]],
        "val": [scenes[i] for i in order[n_tr:n_tr + n_va]],
        "test": [scenes[i] for i in order[n_tr + n_va:]],
    }

    op = shutil.move if args.move else shutil.copy2
    meta = {"scene_regex": args.scene_regex, "seed": args.seed,
            "ratios": args.ratios, "splits": {}}

    for split, scene_list in assign.items():
        img_out = root / split / "images"
        mask_out = root / split / "masks"
        # 若目标 split 与来源相同，先清空避免新旧混杂
        if split == args.from_split and not args.move:
            print(f"[!] 目标 split '{split}' 与来源相同，跳过以免自我覆盖")
            continue
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        tile_stems: list[str] = []
        for scene in scene_list:
            for stem in groups[scene]:
                op(str(src_img / f"{stem}.png"), str(img_out / f"{stem}.png"))
                op(str(src_mask / f"{stem}.png"), str(mask_out / f"{stem}.png"))
                tile_stems.append(stem)

        # 统计水体占比，确认各 split 分布相近（差太多说明划分不平衡）
        ratios = []
        for stem in tile_stems:
            m = cv2.imread(str(mask_out / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                ratios.append(float((m > 127).mean()))
        arr = np.array(ratios) if ratios else np.array([0.0])

        meta["splits"][split] = {
            "n_scenes": len(scene_list),
            "n_tiles": len(tile_stems),
            "water_ratio_mean": float(arr.mean()),
            "water_ratio_median": float(np.median(arr)),
            "n_no_water_tiles": int((arr < 1e-6).sum()),
            "scenes": sorted(scene_list),
        }
        print(f"  {split:<6} {len(scene_list):>4} 场景 / {len(tile_stems):>4} 图块  "
              f"水体占比均值={arr.mean():.4f} 中位={np.median(arr):.4f} "
              f"无水块={int((arr < 1e-6).sum())}")

    # 交叉验证：确认任意两个 split 的场景集合无交集
    print("\n泄漏检查（任意两 split 的场景交集应为空）:")
    names = [k for k in assign if k in meta["splits"]]
    leak = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = set(meta["splits"][names[i]]["scenes"])
            b = set(meta["splits"][names[j]]["scenes"])
            inter = a & b
            status = f"!! 泄漏 {len(inter)} 个场景" if inter else "OK 无交集"
            if inter:
                leak = True
            print(f"  {names[i]:<6} vs {names[j]:<6}: {status}")

    with open(root / "splits_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'[!] 存在泄漏，请检查' if leak else '[OK] 场景级划分完成，无空间泄漏'}"
          f" -> {root / 'splits_meta.json'}")


if __name__ == "__main__":
    main()
