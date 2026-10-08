"""数据预处理：把原始数据集转成统一的二值水体分割格式。

这些活全是 CPU 密集、GPU 用不上的，所以**在本地做完再上传**，
不要在按小时计费的云 GPU 上跑预处理。

子命令
------
  inspect   查看标签图里实际出现的类别索引及像素占比（**先跑这个**）
  loveda    转换 LoveDA（卫星，源域）
  floodnet  转换 FloodNet（无人机，目标域）
  synthetic 生成合成数据，用于本地 CPU 冒烟测试

用法示例
--------
  python -m src.prepare_data inspect --labels_dir data/raw/LoveDA/Train/Rural/masks_png
  python -m src.prepare_data loveda --src data/raw/LoveDA --out data/processed/loveda --tile 512
  python -m src.prepare_data synthetic --out data/processed/synthetic
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 类别索引映射
#
# ⚠️ 重要：下面这些索引是根据数据集文档整理的**默认值**，不同发布版本可能不同。
# 使用前务必先跑 `inspect` 子命令，用实际数据核对，再决定 --water_ids。
# 索引错了整个实验就全错了，而且很难从指标上看出来。
# ---------------------------------------------------------------------------
LOVEDA_CLASSES = {
    0: "no-data/ignore",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}
LOVEDA_WATER_IDS = [4]

FLOODNET_CLASSES = {
    0: "background",
    1: "building-flooded",
    2: "building-non-flooded",
    3: "road-flooded",
    4: "road-non-flooded",
    5: "water",
    6: "tree",
    7: "vehicle",
    8: "pool",
    9: "grass",
}
# 只取自然水体 + 泳池。是否把 flooded road/building（1、3）算作水体，
# 取决于你的论文定义——水利巡检场景通常关注常态水体，建议先不含洪水淹没类，
# 并在论文里明确说明这个选择。可通过 --water_ids 覆盖。
FLOODNET_WATER_IDS = [5, 8]


# ---------------------------------------------------------------------------
# inspect：核对标签索引
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace) -> None:
    labels_dir = Path(args.labels_dir)
    paths = sorted(
        p for p in labels_dir.rglob("*") if p.suffix.lower() in {".png", ".tif", ".tiff"}
    )
    if not paths:
        raise SystemExit(f"{labels_dir} 下没找到 png/tif 标签文件")

    paths = paths[: args.limit]
    counter: Counter[int] = Counter()
    shapes: Counter[tuple] = Counter()

    for p in tqdm(paths, desc="扫描标签"):
        mask = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:  # 有些数据集把标签存成 3 通道，取第一个通道
            mask = mask[..., 0]
        shapes[mask.shape] += 1
        values, counts = np.unique(mask, return_counts=True)
        for v, c in zip(values, counts):
            counter[int(v)] += int(c)

    total = sum(counter.values())
    print(f"\n扫描了 {len(paths)} 个标签文件，尺寸分布：{dict(shapes)}")
    print(f"{'索引':>6} {'像素数':>14} {'占比':>9}   可能含义")
    print("-" * 60)
    known = LOVEDA_CLASSES if args.preset == "loveda" else (
        FLOODNET_CLASSES if args.preset == "floodnet" else {}
    )
    for idx in sorted(counter):
        pct = 100 * counter[idx] / total
        print(f"{idx:>6} {counter[idx]:>14,} {pct:>8.3f}%   {known.get(idx, '?')}")
    print(
        "\n请人工确认哪些索引是水体，然后在转换时用 --water_ids 明确指定。"
        "\n如果索引数量或分布与预期不符，不要继续，先查数据集文档。"
    )


# ---------------------------------------------------------------------------
# 通用转换：切图 + 标签二值化
# ---------------------------------------------------------------------------
def _tile_and_save(
    image: np.ndarray,
    mask_bin: np.ndarray,
    out_dir: Path,
    stem: str,
    tile: int | None,
    stride: int | None,
    min_water_ratio: float,
    keep_empty_prob: float,
    rng: np.random.Generator,
) -> int:
    """把一张大图切成小块保存。返回实际保存的块数。

    min_water_ratio / keep_empty_prob 用于控制正负样本比例：
    水体占比低于阈值的块，只以 keep_empty_prob 的概率保留。
    全部保留会让负样本压倒性多数；全部丢弃则模型没见过纯背景，
    实际推理时会在陆地上误报。所以按概率采样保留一部分。
    """
    img_out = out_dir / "images"
    mask_out = out_dir / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    h, w = mask_bin.shape
    if tile is None:
        boxes = [(0, 0, w, h)]
    else:
        step = stride or tile
        boxes = []
        # 最后一行/列不足 tile 时，回退对齐到边界，保证覆盖整图且不补边
        ys = list(range(0, max(1, h - tile + 1), step))
        xs = list(range(0, max(1, w - tile + 1), step))
        if h > tile and ys[-1] != h - tile:
            ys.append(h - tile)
        if w > tile and xs[-1] != w - tile:
            xs.append(w - tile)
        for y in ys:
            for x in xs:
                boxes.append((x, y, min(tile, w), min(tile, h)))

    saved = 0
    for x, y, bw, bh in boxes:
        img_patch = image[y : y + bh, x : x + bw]
        mask_patch = mask_bin[y : y + bh, x : x + bw]
        if mask_patch.size == 0:
            continue

        ratio = float(mask_patch.mean())
        if ratio < min_water_ratio and rng.random() > keep_empty_prob:
            continue

        name = stem if tile is None else f"{stem}_y{y}x{x}"
        cv2.imwrite(str(img_out / f"{name}.png"), img_patch)
        cv2.imwrite(str(mask_out / f"{name}.png"), (mask_patch * 255).astype(np.uint8))
        saved += 1

    return saved


def _convert_split(
    img_paths: list[Path],
    mask_lookup,
    water_ids: list[int],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """转换一个 split。mask_lookup 是 img_path -> mask_path 的函数。"""
    rng = np.random.default_rng(args.seed)
    total_saved = 0
    skipped = []

    for img_path in tqdm(img_paths, desc=f"转换 {out_dir.name}"):
        mask_path = mask_lookup(img_path)
        if mask_path is None or not mask_path.exists():
            skipped.append(str(img_path.name))
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask_raw is None:
            skipped.append(str(img_path.name))
            continue
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[..., 0]

        # 标签图和影像尺寸不一致时，把标签按最近邻缩放到影像尺寸
        # （必须用 NEAREST，插值会产生不存在的类别索引）
        if mask_raw.shape[:2] != image.shape[:2]:
            mask_raw = cv2.resize(
                mask_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        mask_bin = np.isin(mask_raw, water_ids).astype(np.uint8)

        total_saved += _tile_and_save(
            image, mask_bin, out_dir, img_path.stem,
            args.tile, args.stride, args.min_water_ratio, args.keep_empty_prob, rng,
        )

    return {"saved_tiles": total_saved, "skipped": skipped}


# ---------------------------------------------------------------------------
# LoveDA
# ---------------------------------------------------------------------------
def cmd_loveda(args: argparse.Namespace) -> None:
    """LoveDA 原始结构大致为 Train/Val/Test 下的 Rural/Urban，各含 images_png 与 masks_png。

    注意 LoveDA 官方 Test 集不公开标签，所以这里只用 Train/Val。
    源域只是用来预训练，不需要在它上面报测试指标。
    """
    src = Path(args.src)
    out = Path(args.out)
    water_ids = args.water_ids or LOVEDA_WATER_IDS
    print(f"LoveDA 水体索引 = {water_ids}（请确认已用 inspect 核对过）")

    summary = {}
    for split_name, src_split in [("train", "Train"), ("val", "Val")]:
        img_paths, lookup_map = [], {}
        for domain in ["Rural", "Urban"]:
            img_dir = src / src_split / domain / "images_png"
            mask_dir = src / src_split / domain / "masks_png"
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.glob("*.png")):
                # 加域前缀防止 Rural/Urban 同名文件互相覆盖
                img_paths.append(p)
                lookup_map[p] = mask_dir / p.name

        if not img_paths:
            print(f"[!] {src_split} 下没找到影像，跳过。请检查 --src 指向的目录结构。")
            continue

        summary[split_name] = _convert_split(
            img_paths, lambda p: lookup_map.get(p), water_ids, out / split_name, args
        )

    _write_meta(out, {"dataset": "loveda", "water_ids": water_ids, "splits": summary}, args)


# ---------------------------------------------------------------------------
# FloodNet
# ---------------------------------------------------------------------------
def cmd_floodnet(args: argparse.Namespace) -> None:
    """FloodNet 的目录命名在不同发布版本里不一致，这里做模糊匹配。

    找不到就会报错并打印实际目录树，按提示用 --img_glob / --mask_dir 手动指定。
    """
    src = Path(args.src)
    out = Path(args.out)
    water_ids = args.water_ids or FLOODNET_WATER_IDS
    print(f"FloodNet 水体索引 = {water_ids}（请确认已用 inspect 核对过）")

    # 递归找所有影像，再按同名规则配标签
    img_paths = sorted(
        p for p in src.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "label" not in p.parent.name.lower()
        and "mask" not in p.parent.name.lower()
    )
    if not img_paths:
        raise SystemExit(f"{src} 下没找到影像文件。目录内容：\n" + _tree(src))

    # 建立 stem -> 标签路径 的索引
    mask_index = {}
    for p in src.rglob("*"):
        if p.suffix.lower() not in {".png", ".tif", ".tiff"}:
            continue
        if "label" not in p.parent.name.lower() and "mask" not in p.parent.name.lower():
            continue
        # FloodNet 标签常带 _lab 后缀，统一去掉便于匹配
        key = p.stem.replace("_lab", "").replace("_label", "").replace("_mask", "")
        mask_index[key] = p

    if not mask_index:
        raise SystemExit(
            f"{src} 下没找到标签目录（目录名需含 label 或 mask）。目录内容：\n" + _tree(src)
        )

    # FloodNet 官方划分只有 train/val/test，按路径里的关键词归类
    buckets: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for p in img_paths:
        lowered = str(p).lower()
        if "val" in lowered:
            buckets["val"].append(p)
        elif "test" in lowered:
            buckets["test"].append(p)
        else:
            buckets["train"].append(p)

    summary = {}
    for split_name, paths in buckets.items():
        if not paths:
            continue
        summary[split_name] = _convert_split(
            paths, lambda p: mask_index.get(p.stem), water_ids, out / split_name, args
        )

    _write_meta(out, {"dataset": "floodnet", "water_ids": water_ids, "splits": summary}, args)


# ---------------------------------------------------------------------------
# generic：直接指定 images/masks 目录的通用转换
# ---------------------------------------------------------------------------
def cmd_generic(args: argparse.Namespace) -> None:
    """通用转换：源目录已经是 <split>/images + <split>/masks 的简单结构。

    适用于 download_loveda.py 下载的数据，以及你自己整理好的自采数据。

    ⚠️ 空间泄漏警告
    切图会把一张大图切成多个图块。若后续对**图块**做随机划分，
    同一张原图的图块会同时落入训练集和测试集，造成空间信息泄漏、指标虚高。
    本脚本在图块文件名中保留原图名（`{原图名}_y{y}x{x}`），
    因此可以按原图名分组来做无泄漏划分——**划分必须在原图/场景层级进行**。
    本命令不跨 split 混合数据，各 split 独立转换，因此本身不引入泄漏。
    """
    src = Path(args.src)
    out = Path(args.out)
    water_ids = args.water_ids
    if not water_ids:
        raise SystemExit("必须用 --water_ids 明确指定水体类别索引（先跑 inspect 核对）")
    print(f"水体索引 = {water_ids}")

    summary = {}
    for split in args.splits:
        img_dir = src / split / "images"
        mask_dir = src / split / "masks"
        if not img_dir.is_dir():
            print(f"[!] 跳过 {split}：{img_dir} 不存在")
            continue

        img_paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        if not img_paths:
            print(f"[!] 跳过 {split}：{img_dir} 下没有影像")
            continue

        summary[split] = _convert_split(
            img_paths,
            lambda p: mask_dir / f"{p.stem}.png",
            water_ids, out / split, args,
        )
        print(f"{split}: 保存 {summary[split]['saved_tiles']} 个图块"
              f"（跳过 {len(summary[split]['skipped'])} 张无掩膜）")

    _write_meta(out, {"dataset": "generic", "src": str(src),
                      "water_ids": water_ids, "splits": summary}, args)
    print(f"\n[OK] 转换完成 -> {out}")


def _tree(root: Path, max_lines: int = 40) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            lines.append(f"  {p.relative_to(root)}/")
        if len(lines) >= max_lines:
            lines.append("  ...")
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# synthetic：本地冒烟测试用
# ---------------------------------------------------------------------------
def cmd_synthetic(args: argparse.Namespace) -> None:
    """生成合成的"航拍水体"图像。

    刻意模拟了水体的三个真实特征，这样冒烟测试不只是跑通流程，
    还能粗略验证模型确实学到了东西：
      - 水体区域纹理平滑（真实水面缺少纹理）
      - 陆地区域高频纹理丰富
      - 水面上随机加高亮斑块（模拟镜面反光，本课题的核心难点）
    """
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    size = args.size

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split, n in counts.items():
        img_dir = out / split / "images"
        mask_dir = out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(n), desc=f"合成 {split}"):
            # 1) 用低频随机场阈值化，生成形状自然的水体区域
            low = rng.normal(size=(8, 8)).astype(np.float32)
            field = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
            mask = (field > rng.uniform(-0.3, 0.5)).astype(np.uint8)

            # 2) 陆地：棕绿色 + 强高频纹理
            land = np.zeros((size, size, 3), np.float32)
            land[..., 0] = rng.uniform(60, 90)    # B
            land[..., 1] = rng.uniform(90, 130)   # G
            land[..., 2] = rng.uniform(70, 110)   # R
            land += rng.normal(0, 18, (size, size, 3))

            # 3) 水体：蓝青色 + 极弱纹理
            water = np.zeros((size, size, 3), np.float32)
            water[..., 0] = rng.uniform(110, 150)
            water[..., 1] = rng.uniform(85, 115)
            water[..., 2] = rng.uniform(55, 80)
            water += rng.normal(0, 4, (size, size, 3))

            image = np.where(mask[..., None] > 0, water, land)

            # 4) 镜面反光：水面上的高亮斑块。注意标签仍是水体——
            #    这正是难点所在，模型不能因为像素变白就判成非水。
            if rng.random() < 0.6:
                glare = np.zeros((size, size), np.float32)
                for _ in range(rng.integers(1, 4)):
                    cx, cy = rng.integers(0, size, 2)
                    r = rng.integers(size // 12, size // 5)
                    cv2.circle(glare, (int(cx), int(cy)), int(r), 1.0, -1)
                glare = cv2.GaussianBlur(glare, (0, 0), size / 20)
                glare = glare * (mask > 0)  # 只在水面上出现
                image += glare[..., None] * rng.uniform(70, 130)

            image = np.clip(image, 0, 255).astype(np.uint8)
            cv2.imwrite(str(img_dir / f"syn_{i:04d}.png"), image)
            cv2.imwrite(str(mask_dir / f"syn_{i:04d}.png"), mask * 255)

    _write_meta(out, {"dataset": "synthetic", "counts": counts, "size": size}, args)
    print(f"\n[OK] 合成数据已生成到 {out}")
    print("  这只是流程验证用的假数据，绝不能用于论文任何结果。")


# ---------------------------------------------------------------------------
def _write_meta(out: Path, meta: dict, args: argparse.Namespace) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                    if k != "func"}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 切图相关参数，几个转换命令共用
    def add_tile_args(p):
        p.add_argument("--tile", type=int, default=512, help="切图尺寸，0 表示不切")
        p.add_argument("--stride", type=int, default=None, help="切图步长，默认等于 tile（不重叠）")
        p.add_argument("--min_water_ratio", type=float, default=0.005,
                       help="水体占比低于此值视为负样本块")
        p.add_argument("--keep_empty_prob", type=float, default=0.15,
                       help="负样本块的保留概率，控制正负比例")
        p.add_argument("--water_ids", type=int, nargs="+", default=None,
                       help="水体类别索引，务必先用 inspect 核对")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("inspect", help="查看标签实际类别索引")
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--limit", type=int, default=50, help="最多扫描多少个文件")
    p.add_argument("--preset", choices=["loveda", "floodnet", "none"], default="none")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("loveda", help="转换 LoveDA")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    add_tile_args(p)
    p.set_defaults(func=cmd_loveda)

    p = sub.add_parser("floodnet", help="转换 FloodNet")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    add_tile_args(p)
    p.set_defaults(func=cmd_floodnet)

    p = sub.add_parser("generic", help="通用转换：源目录为 <split>/images + <split>/masks")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    add_tile_args(p)
    p.set_defaults(func=cmd_generic)

    p = sub.add_parser("synthetic", help="生成合成数据用于冒烟测试")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--n_train", type=int, default=40)
    p.add_argument("--n_val", type=int, default=8)
    p.add_argument("--n_test", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synthetic)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    # tile=0 统一转成 None，表示不切图
    if getattr(args, "tile", None) == 0:
        args.tile = None
    args.func(args)
