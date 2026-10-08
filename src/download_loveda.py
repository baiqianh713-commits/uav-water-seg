"""从 HuggingFace 镜像下载 LoveDA 的一个子集，用于本地跑通真实数据 baseline。

⚠️ 重要声明
-----------
本脚本用的是**第三方镜像** `chloechia/loveda`，其 train 只有 1366 对
（官方 Zenodo 发布为 2522 对 train / 1669 对 val），说明 train 是不完整子集。

因此：
  - 本脚本适合**本地验证流程、快速出 baseline**
  - **论文正式实验必须用官方发布版**：https://zenodo.org/records/5706578
    （Train.zip 4.0GB / Val.zip 2.4GB），否则数据来源无法在论文中交代清楚

带宽优化策略
------------
掩膜 PNG 只有约 0.01MB，而影像约 1.76MB（差 176 倍）。所以：
  1. 先并行下载**全部**掩膜（很快），算出每张图的水体占比
  2. 再按占比分层挑选子集，**只下载选中的影像**
这样既省带宽，又能免费得到全量水体占比统计（论文数据章节要用）。

用法
----
  python -m src.download_loveda --out data/raw/loveda_hf --n_train 150 --n_val 50
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE = "https://huggingface.co/datasets/chloechia/loveda/resolve/main/"
# 镜像里的目录名带冒号，URL 里必须转义，交给 urllib.parse.quote 处理
DIRS = {
    "train": ("urban:rural train images", "urban:rural train masks"),
    "val": ("urban:rural val images", "urban:rural val masks"),
}
WATER_ID = 4  # 已由镜像的 id2label.json 确认：4 = Water


def fetch(path: str, retries: int = 3, timeout: int = 60) -> bytes | None:
    """下载单个文件。失败重试，最终失败返回 None（不中断整批任务）。"""
    url = BASE + urllib.parse.quote(path)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def download_many(paths: list[str], dest_dir: Path, workers: int, desc: str) -> list[str]:
    """并行下载一批文件到 dest_dir。返回成功的文件名（stem）列表。

    用线程池而非进程池：瓶颈是网络延迟（I/O 等待），不是 CPU，
    线程足够且没有进程间通信开销。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    ok: list[str] = []

    def work(p: str) -> tuple[str, bool]:
        name = p.rsplit("/", 1)[-1]
        out = dest_dir / name
        if out.exists() and out.stat().st_size > 0:
            return Path(name).stem, True  # 已下载过就跳过，支持中断续传
        data = fetch(p)
        if data is None:
            return Path(name).stem, False
        out.write_bytes(data)
        return Path(name).stem, True

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, p) for p in paths]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
            stem, success = fut.result()
            if success:
                ok.append(stem)

    return sorted(ok, key=lambda s: int(s) if s.isdigit() else 0)


def list_ids(split: str) -> list[str]:
    """从 HF API 取该 split 下所有文件的 id（即不带扩展名的文件名）。"""
    api = "https://huggingface.co/api/datasets/chloechia/loveda?full=true"
    data = json.loads(urllib.request.urlopen(api, timeout=60).read())
    mask_dir = DIRS[split][1]
    ids = [
        Path(s["rfilename"]).stem
        for s in data.get("siblings", [])
        if s["rfilename"].startswith(mask_dir + "/")
    ]
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else 0)


def water_ratio_of(mask_path: Path) -> float:
    """读掩膜算水体像素占比。掩膜是类别索引图，水体为 4。"""
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return -1.0
    if m.ndim == 3:
        m = m[..., 0]
    return float((m == WATER_ID).mean())


def stratified_pick(ratios: dict[str, float], n: int, seed: int = 0) -> list[str]:
    """按水体占比分层挑选 n 张，保证覆盖从无水到大面积水体。

    为什么要分层而不是随机：随机挑选时，若数据集里大部分图不含水，
    子集可能几乎全是负样本，模型学不到东西，baseline 会失去意义。
    分层能保证正负样本都有代表性。

    注意：这只是为了**搭建一个有意义的本地 baseline 数据子集**，
    不是论文里的选帧策略。论文的选帧必须在无标注前提下进行
    （见 selection.py），而这里用了掩膜信息，只能用于构造数据集本身。
    """
    rng = np.random.default_rng(seed)
    items = [(k, v) for k, v in ratios.items() if v >= 0]

    # 分四层：无水 / 少量水 / 中等 / 大面积
    bands = {
        "none": [k for k, v in items if v < 1e-6],
        "low": [k for k, v in items if 1e-6 <= v < 0.05],
        "mid": [k for k, v in items if 0.05 <= v < 0.25],
        "high": [k for k, v in items if v >= 0.25],
    }
    print("  水体占比分层统计:")
    for name, members in bands.items():
        print(f"    {name:<6} {len(members):>5} 张")

    # 负样本（none）只取约 15%，其余给含水样本。
    # 全要负样本会导致正样本过少；完全不要则模型在陆地上会误报。
    quota = {"none": int(n * 0.15), "low": int(n * 0.25),
             "mid": int(n * 0.35), "high": n - int(n * 0.15) - int(n * 0.25) - int(n * 0.35)}

    picked: list[str] = []
    shortfall = 0
    for name, want in quota.items():
        members = bands[name]
        take = min(want, len(members))
        shortfall += want - take
        if take:
            picked.extend(rng.choice(members, size=take, replace=False).tolist())

    # 某层不够时，从其他层补齐
    if shortfall > 0:
        rest = [k for k, _ in items if k not in set(picked)]
        if rest:
            extra = rng.choice(rest, size=min(shortfall, len(rest)), replace=False)
            picked.extend(extra.tolist())

    return sorted(set(picked), key=lambda s: int(s) if s.isdigit() else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/loveda_hf")
    ap.add_argument("--n_train", type=int, default=150)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    stats = {}

    for split, n_want in [("train", args.n_train), ("val", args.n_val)]:
        if n_want <= 0:
            continue
        print(f"\n{'=' * 60}\n{split}: 目标 {n_want} 对\n{'=' * 60}")
        img_dir_remote, mask_dir_remote = DIRS[split]

        ids = list_ids(split)
        print(f"该 split 共 {len(ids)} 对")

        # --- 第一步：下载全部掩膜（很小，顺带得到全量统计）---
        mask_local = out / split / "masks"
        got = download_many(
            [f"{mask_dir_remote}/{i}.png" for i in ids],
            mask_local, args.workers, f"{split} 掩膜",
        )
        print(f"掩膜下载成功 {len(got)}/{len(ids)}")

        # --- 第二步：算水体占比，分层挑选 ---
        ratios = {stem: water_ratio_of(mask_local / f"{stem}.png") for stem in got}
        valid = np.array([v for v in ratios.values() if v >= 0])
        print(f"全量水体占比: 均值={valid.mean():.4f} 中位={np.median(valid):.4f} "
              f"最大={valid.max():.4f} 无水图={int((valid < 1e-6).sum())}/{len(valid)}")

        picked = stratified_pick(ratios, n_want, seed=args.seed)
        print(f"挑选出 {len(picked)} 张")

        # --- 第三步：只下载选中的影像 ---
        img_local = out / split / "images"
        got_img = download_many(
            [f"{img_dir_remote}/{i}.png" for i in picked],
            img_local, args.workers, f"{split} 影像",
        )
        print(f"影像下载成功 {len(got_img)}/{len(picked)}")

        # 清理：删掉没有对应影像的掩膜，保证 images/masks 一一对应
        keep = set(got_img)
        removed = 0
        for p in mask_local.glob("*.png"):
            if p.stem not in keep:
                p.unlink()
                removed += 1
        print(f"清理未配对掩膜 {removed} 个，最终 {len(keep)} 对")

        sel_ratios = {k: ratios[k] for k in got_img if k in ratios}
        arr = np.array(list(sel_ratios.values()))
        stats[split] = {
            "n_pairs": len(keep),
            "population_size": len(ids),
            "population_water_ratio_mean": float(valid.mean()),
            "subset_water_ratio_mean": float(arr.mean()) if len(arr) else 0.0,
            "subset_no_water_count": int((arr < 1e-6).sum()) if len(arr) else 0,
        }

    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "HuggingFace mirror chloechia/loveda",
        "warning": "第三方镜像，train 不完整（1366 vs 官方 2522）。论文正式实验须用官方 Zenodo 发布版。",
        "official_url": "https://zenodo.org/records/5706578",
        "water_class_id": WATER_ID,
        "splits": stats,
    }
    with open(out / "download_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 完成，元信息写入 {out / 'download_meta.json'}")
    print("提醒：这是第三方镜像子集，仅用于本地验证；论文实验请用官方发布版。")


if __name__ == "__main__":
    main()
