"""把自采数据的 AnyLabeling 标注转成 images/ + masks/ 标准结构。

与 fetch_rivair_labels.convert() 的区别
--------------------------------------
RivAIrSet 的 JSON 是开着「保存图像数据」导出的，原图以 base64 内嵌在
`imageData` 字段里。你自己用 AnyLabeling 标注时这个开关默认常常是**关的**，
JSON 里 `imageData` 为 null，真图在同目录的同名 .jpg。

本脚本两种都支持：优先读内嵌 base64，没有就找同名图片文件。
其余逻辑（water 多边形 → fillPoly → 二值掩膜）与 RivAIrSet 保持一致，
保证两个数据集的标注口径相同，论文里可以直接对比。

标注约定
--------
在 AnyLabeling 里把水体多边形的标签**统一写成 `water`**（大小写不敏感）。
其他标签一律忽略。一张图里有多个水体区域就画多个多边形。
整张图没有水就**不画任何多边形，但仍要保存 JSON** —— 这样它会得到一张
全 0 掩膜，是合法的负样本；如果干脆不存 JSON，脚本会当成「未标注」跳过。

用法
----
    # 先看标注进度，不转换
    python -m src.convert_field_labels --input data/raw/field_frames --progress

    # 转换
    python -m src.convert_field_labels --input data/raw/field_frames \\
        --out data/processed/field --max-side 1024
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def load_image(js_path: Path, data: dict) -> np.ndarray | None:
    """取出这条标注对应的原图。

    两条路径：
      1) JSON 内嵌 base64（AnyLabeling 开了「保存图像数据」）
      2) 同目录同名图片文件（默认情况）
    都取不到就返回 None，调用方跳过。
    """
    b64 = data.get("imageData")
    if b64:
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return np.array(img)

    # imagePath 可能是绝对路径或带子目录，只取文件名，在 JSON 同级目录找
    cand = []
    ip = data.get("imagePath")
    if ip:
        cand.append(js_path.parent / Path(ip).name)
    cand += [js_path.with_suffix(e) for e in IMG_EXTS]

    for c in cand:
        if c.exists():
            img = cv2.imread(str(c), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def polygons_to_mask(data: dict, h: int, w: int) -> tuple[np.ndarray, int]:
    """把所有 label=='water' 的多边形填成二值掩膜。返回(掩膜, 用到的多边形数)。"""
    mask = np.zeros((h, w), np.uint8)
    used = 0
    for s in data.get("shapes", []):
        if s.get("label", "").strip().lower() != "water":
            continue
        pts = np.array(s.get("points", []), dtype=np.float64)
        if len(pts) < 3:
            # 少于 3 个点构不成多边形（可能是误点的矩形/线），跳过
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        used += 1
    return mask, used


def cond_of(stem: str) -> str:
    """帧名形如 `A_50m_常光__DJI_xxx__f000120`，双下划线前是条件 ID。"""
    return stem.split("__")[0] if "__" in stem else "(未知)"


def progress(root: Path) -> None:
    """只统计标注进度，不做转换。标注是长期工作，随时可以看还剩多少。"""
    imgs = [p for p in root.rglob("*") if p.suffix in IMG_EXTS]
    jsons = {p.with_suffix("").name for p in root.rglob("*.json")}

    by_cond: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [已标, 总数]
    for p in imgs:
        c = cond_of(p.stem)
        by_cond[c][1] += 1
        if p.stem in jsons:
            by_cond[c][0] += 1

    print(f"{'条件':<16}{'已标注':>8}{'总帧数':>8}{'进度':>8}")
    print("-" * 42)
    tot_done = tot_all = 0
    for c in sorted(by_cond):
        done, all_ = by_cond[c]
        tot_done += done
        tot_all += all_
        print(f"{c:<16}{done:>8}{all_:>8}{done / max(all_, 1):>7.0%}")
    print("-" * 42)
    print(f"{'合计':<16}{tot_done:>8}{tot_all:>8}{tot_done / max(tot_all, 1):>7.0%}")


def convert(root: Path, out_root: Path, max_side: int | None) -> None:
    js = sorted(root.rglob("*.json"))
    # extract_video 自己写的元数据文件不是标注，必须排除
    js = [p for p in js if p.name not in {"condition_map.json", "frames_meta.json"}]
    if not js:
        raise SystemExit(f"{root} 下没有找到标注 JSON")

    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ratios: dict[str, list[float]] = defaultdict(list)
    n_ok = n_skip = n_nopoly = 0

    for p in js:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            n_skip += 1
            continue

        img = load_image(p, d)
        if img is None:
            print(f"  [!] 找不到原图，跳过: {p.name}")
            n_skip += 1
            continue

        h, w = img.shape[:2]
        mask, used = polygons_to_mask(d, h, w)
        if used == 0:
            # 合法的无水负样本，但数量异常多时要警惕是不是标签写错了
            n_nopoly += 1

        if max_side and max(h, w) > max_side:
            scale = max_side / max(h, w)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            # 掩膜必须最近邻，双线性会插出 0/1 之外的值
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        stem = p.with_suffix("").name
        cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask * 255)
        ratios[cond_of(stem)].append(float(mask.mean()))
        n_ok += 1

    all_r = np.array([v for vs in ratios.values() for v in vs]) if ratios else np.array([0.0])
    print(f"\n转换完成 {n_ok} 对 -> {out_root}")
    if n_skip:
        print(f"  跳过 {n_skip} 个（解析失败或找不到原图）")
    print(f"  全图无水的样本: {n_nopoly}/{n_ok}"
          f"  —— 若这个比例异常高，检查标签是不是没写成 water")
    print(f"\n水体占比  均值={all_r.mean():.4f}  中位={np.median(all_r):.4f}"
          f"  最小={all_r.min():.4f}  最大={all_r.max():.4f}")

    print(f"\n分条件水体占比（条件间差异大会影响预算分配的解释）:")
    for c in sorted(ratios):
        a = np.array(ratios[c])
        print(f"  {c:<16} n={len(a):>4}  均值={a.mean():.4f}  中位={np.median(a):.4f}")

    print(f"\n下一步 —— 冗余度诊断:")
    print(f"  python -m src.diagnose_redundancy --root {out_root.parent} "
          f"--split {out_root.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw/field_frames",
                    help="含图片与 AnyLabeling JSON 的目录")
    ap.add_argument("--out", default="data/processed/field", help="输出根目录")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="长边缩到该尺寸；设 0 保留原尺寸")
    ap.add_argument("--progress", action="store_true", help="只看标注进度，不转换")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    if args.progress:
        progress(root)
    else:
        convert(root, Path(args.out), args.max_side or None)


if __name__ == "__main__":
    main()
