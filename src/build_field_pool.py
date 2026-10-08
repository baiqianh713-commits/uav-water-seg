"""把抽出的自采帧整理成「候选池」标准布局，供冗余度诊断与标注使用。

为什么要这一步
--------------
`extract_video` 输出的是 `field_frames/images/*.jpg`（4K 原尺寸），
而 `selection.extract_features` / `data.list_samples` 约定的布局是
`<root>/<split>/images/*.png`。这里做三件事：

1. jpg -> png，并按约定的目录层级摆放
2. 长边缩到 1024 —— 与 RivAIrSet 的 `max_side=1024` 口径一致，
   两个数据集的冗余度数值才可比；同时标注软件打开 1024 的图快得多
3. 可选的时间抽稀（--stride）—— 候选池不一定要用满 1fps 的全部帧

关于 --stride 的取舍
--------------------
抽稀会降低帧间冗余，而冗余正是本方法的研究对象，所以**默认不抽稀**(stride=1)。
只有当候选池大到标注不起时才考虑。本次外业飞行速度仅 0.45~0.74 m/s，
1fps 下相邻帧位移不足 1 米，即使 stride=3 仍有很高重叠。

用法
----
    python -m src.build_field_pool --input data/raw/field_frames \\
        --out data/processed/field_pool --split _all --max-side 1024
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def cond_of(stem: str) -> str:
    """帧名形如 `A_50m_常光__DJI_xxx__f000120`，双下划线前是条件 ID。"""
    return stem.split("__")[0] if "__" in stem else "(未知)"


def video_of(stem: str) -> str:
    """取架次（源视频）标识。防泄漏划分要按架次，不能按帧随机。"""
    parts = stem.split("__")
    return parts[1] if len(parts) > 2 else "(未知)"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default="data/raw/field_frames")
    ap.add_argument("--out", default="data/processed/field_pool")
    ap.add_argument("--split", default="_all",
                    help="子目录名。候选池整体先放 _all，划分留给 make_splits")
    ap.add_argument("--max-side", type=int, default=1024, help="长边目标尺寸；0 表示不缩放")
    ap.add_argument("--stride", type=int, default=1,
                    help="每隔几帧取一张。默认 1（不抽稀），抽稀会削弱冗余")
    args = ap.parse_args()

    src_img = Path(args.input) / "images"
    if not src_img.is_dir():
        raise SystemExit(f"找不到 {src_img}，先运行 src.extract_video")

    frames = sorted(p for p in src_img.iterdir() if p.suffix in IMG_EXTS)
    if not frames:
        raise SystemExit(f"{src_img} 下没有图片")

    # 按架次分组后再抽稀，保证每个架次都被均匀抽到，
    # 否则全局抽稀会因架次帧数不同而产生偏差。
    by_video: dict[str, list[Path]] = defaultdict(list)
    for p in frames:
        by_video[video_of(p.stem)].append(p)

    picked: list[Path] = []
    for v in sorted(by_video):
        picked += by_video[v][:: max(1, args.stride)]
    picked.sort()

    dst = Path(args.out) / args.split / "images"
    dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for p in picked:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [!] 读不出，跳过: {p.name}")
            continue
        h, w = img.shape[:2]
        if args.max_side and max(h, w) > args.max_side:
            scale = args.max_side / max(h, w)
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / f"{p.stem}.png"), img)
        n += 1

    # 条件映射跟着一起搬，贡献点 2 的预算分配要用
    src_map = Path(args.input) / "condition_map.json"
    if src_map.exists():
        shutil.copy(src_map, Path(args.out) / "condition_map.json")

    by_cond: dict[str, int] = defaultdict(int)
    by_vid: dict[str, int] = defaultdict(int)
    for p in picked:
        by_cond[cond_of(p.stem)] += 1
        by_vid[video_of(p.stem)] += 1

    print(f"候选池建好: {dst}  共 {n} 帧"
          f"（原 {len(frames)} 帧，stride={args.stride}）")
    print(f"\n分条件:")
    for c in sorted(by_cond):
        print(f"  {c:<16} {by_cond[c]:>5} 帧")
    print(f"\n分架次（防泄漏划分的最小单位）:")
    for v in sorted(by_vid):
        print(f"  {v:<34} {by_vid[v]:>5} 帧")

    meta = {"n_frames": n, "stride": args.stride, "max_side": args.max_side,
            "by_condition": dict(by_cond), "by_video": dict(by_vid)}
    with open(Path(args.out) / "pool_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    print(f"\n下一步 —— 冗余度诊断（无需标注即可跑）:")
    print(f"  python -m src.diagnose_redundancy --root {args.out} "
          f"--split {args.split} --budgets 5 10 20 50 --n_seeds 5")


if __name__ == "__main__":
    main()
