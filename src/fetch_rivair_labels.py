"""只下载 RivAIrSet 的**标签段**，一次拿到「图像 + 标注」。

核心发现
--------
RivAIrSet 的标注由 AnyLabeling 生成，每个 .json **内嵌了 base64 编码的原图**
（全分辨率 5280×3956）。所以：

    下载标签段 = 同时得到图像和标注

完全不需要下载 images/ 目录那 17.8 GB。每约 4.2 MB 的下载 = 1 个完整标注样本。

归档布局（已探测确认）
----------------------
    0      ~17.8GB   RIVER_WATER_DATASET/2019/images/*.JPG   (不需要)
    ~17.87GB ~22GB   RIVER_WATER_DATASET/2019/labels/*.json  (只要这段)
    之后              2020/, 2021/ ...

技术手段
--------
Zenodo 支持 HTTP Range，可直接取中间片段。但 unrar 无法解析「掐头」的归档，
所以这里：
  1. 用 rar5_slice 解析片段里的文件头，找到第一个完整头部的偏移
  2. 把原档的「签名 + 主档案头」拼到该偏移之前，构造出合法的 RAR
  3. 交给 unrar 正常解包

用法
----
  python -m src.fetch_rivair_labels --mb 900          # 下载约 900MB（约 210 个样本）
  python -m src.fetch_rivair_labels --extract         # 解包
  python -m src.fetch_rivair_labels --convert         # 转成 images/ + masks/
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

URL = "https://zenodo.org/records/17284942/files/RIVER_WATER_DATASET.rar"
GB = 1024 ** 3
# 2019 标签段起点（已通过二分探测确认：17.7GB 仍是 images，17.9GB 已是 labels）
LABELS_START = int(17.87 * GB)

OUT = Path("data/raw/rivair_labels")
SLICE = OUT / "labels_slice.bin"
SPLICED = OUT / "spliced.rar"
EXTRACT_DIR = OUT / "extracted"
UNRAR = "C:/Program Files/WinRAR/UnRAR.exe"


def fetch_range(start: int, end: int, retries: int = 4) -> bytes | None:
    for i in range(retries):
        try:
            req = urllib.request.Request(
                URL, headers={"Range": f"bytes={start}-{end}",
                              "User-Agent": "research-script"})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:
            if i == retries - 1:
                print(f"    [!] {start}-{end} 失败: {type(e).__name__}")
                return None
            time.sleep(3 * (i + 1))
    return None


def download(target_mb: int, chunk_mb: int = 16) -> None:
    """从标签段起点开始下载，支持断点续传。"""
    OUT.mkdir(parents=True, exist_ok=True)
    have = SLICE.stat().st_size if SLICE.exists() else 0
    target = target_mb * 1024 * 1024
    if have >= target:
        print(f"已有 {have / 1e6:.0f} MB，达到目标")
        return

    print(f"标签段起点 {LABELS_START / GB:.2f} GB，目标 {target_mb} MB，已有 {have / 1e6:.0f} MB")
    chunk = chunk_mb * 1024 * 1024
    t0, base = time.time(), have
    with open(SLICE, "r+b" if SLICE.exists() else "wb") as f:
        while have < target:
            s = LABELS_START + have
            e = LABELS_START + min(have + chunk, target) - 1
            data = fetch_range(s, e)
            if data is None:
                print("    中断，重跑本命令即可续传")
                break
            # 必须 seek 到精确位置再写，不能用 "ab" 追加模式。
            # 追加模式下若多个进程并发（守护脚本重启造成），
            # 会把同一块数据重复写入，导致文件中出现重复段而损坏归档。
            f.seek(have)
            f.write(data)
            f.flush()
            have = f.tell()
            el = time.time() - t0
            sp = (have - base) / 1e6 / max(el, 1e-9)
            print(f"  {have / 1e6:6.0f}/{target / 1e6:.0f} MB  {sp:.2f} MB/s  "
                  f"剩余约 {(target - have) / 1e6 / max(sp, 1e-9) / 60:.0f} 分钟")
    print(f"当前 {SLICE.stat().st_size / 1e6:.0f} MB")


def verify(truncate: bool = False) -> int:
    """校验片段完整性，返回第一个损坏位置的字节偏移（无损坏返回 -1）。

    原理：RAR5 里每个文件头紧跟其数据区，下一个头的起点应当恰好等于
    上一个头的「数据区起点 + 数据长度」。若出现跳跃，说明中间多了或少了数据。

    这个检查是必需的 —— 曾因下载用追加模式 + 多进程并发，
    导致同一个 16MB 块被重复写入，unrar 解到那里就 CRC 失败并停止。
    """
    from .rar5_slice import find_entries

    if not SLICE.exists():
        raise SystemExit("还没下载，先运行 --mb")
    data = SLICE.read_bytes()
    ents = sorted(find_entries(data, name_pattern=rb"[A-Za-z0-9_./\\-]+\.json"),
                  key=lambda e: e.header_offset)
    print(f"片段 {len(data) / 1e6:.0f} MB，解析出 {len(ents)} 个文件头")

    first_bad = -1
    n_bad = 0
    for i in range(len(ents) - 1):
        a, b = ents[i], ents[i + 1]
        expect = a.data_offset + a.data_size
        gap = b.header_offset - expect
        if abs(gap) > 4096:               # 容忍少量对齐填充
            n_bad += 1
            if first_bad < 0:
                first_bad = a.header_offset
                name_a = Path(a.name).name
                name_b = Path(b.name).name
                print(f"  [!] 首个断裂: {name_a} -> {name_b}  "
                      f"偏差 {gap:+,} 字节 @ {expect / 1048576:.0f}MB")

    if n_bad == 0:
        print("  [OK] 结构连续，无损坏")
        return -1

    print(f"  [!] 共 {n_bad} 处断裂 —— 从 {first_bad / 1048576:.0f}MB 起数据不可用")
    if truncate:
        with open(SLICE, "r+b") as f:
            f.truncate(first_bad)
        print(f"  已截断至 {first_bad / 1048576:.0f}MB，重跑 --mb 继续下载")
    return first_bad


def splice_and_extract() -> None:
    """拼接成合法 RAR 并解包。"""
    from .rar5_slice import find_entries, read_vint

    if not SLICE.exists():
        raise SystemExit("还没下载，先运行 --mb")

    data = SLICE.read_bytes()
    print(f"片段 {len(data) / 1e6:.0f} MB，扫描文件头...")
    entries = find_entries(data, name_pattern=rb"[A-Za-z0-9_./\\-]+\.json")
    if not entries:
        raise SystemExit("未在片段中找到可解析的文件头")
    first = min(entries, key=lambda e: e.header_offset)
    print(f"解析出 {len(entries)} 个条目，首个 = {Path(first.name).name}")

    # 取原档的「签名 + 主档案头」。这段很短，单独抓一次即可。
    head = fetch_range(0, 8191)
    if head is None:
        raise SystemExit("无法获取归档头")
    assert head[:8] == b"Rar!\x1a\x07\x01\x00", "不是 RAR5 归档"
    pos = 8 + 4                      # 签名 + 主头 CRC32
    hs, pos = read_vint(head, pos)
    main_end = pos + hs

    SPLICED.write_bytes(head[:main_end] + data[first.header_offset:])
    print(f"已拼接 {SPLICED.stat().st_size / 1e6:.0f} MB -> {SPLICED}")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    # 末尾必然截断，unrar 会报错，但前面完整的文件正常解出
    subprocess.run([UNRAR, "x", "-o+", "-inul", str(SPLICED), str(EXTRACT_DIR) + "/"],
                   capture_output=True)
    js = sorted(EXTRACT_DIR.rglob("*.json"))
    print(f"解出 {len(js)} 个标注文件")
    if js:
        print(f"  首个 {js[0].name}   末个 {js[-1].name}")


def convert(out_root: Path, max_side: int | None = 1024) -> None:
    """把 AnyLabeling JSON 转成 images/ + masks/ 的标准结构。

    每个 JSON 内嵌 base64 原图，同时含 'water' 多边形。
    多边形按 imageWidth/Height 的像素坐标给出，直接填充成二值掩膜。

    max_side 把长边缩到该尺寸再存：5280×3956 的原图训练时必然要切图或缩放，
    先缩小能大幅节省磁盘和后续 IO。设为 None 则保留原尺寸。
    """
    js = sorted(EXTRACT_DIR.rglob("*.json"))
    if not js:
        raise SystemExit(f"{EXTRACT_DIR} 下没有 json，先运行 --extract")

    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    stats = []
    for p in tqdm(js, desc="转换"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        b64 = d.get("imageData")
        if not b64:
            continue

        img = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
        h, w = img.shape[:2]

        # 多边形 -> 二值掩膜
        mask = np.zeros((h, w), np.uint8)
        for s in d.get("shapes", []):
            if s.get("label", "").lower() != "water":
                continue
            pts = np.array(s["points"], dtype=np.float64)
            if len(pts) < 3:
                continue
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)

        if max_side and max(h, w) > max_side:
            scale = max_side / max(h, w)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            # 掩膜必须用最近邻，插值会产生 0/1 之外的值
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        stem = Path(d.get("imagePath", p.stem)).stem
        cv2.imwrite(str(img_dir / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask * 255)
        stats.append(float(mask.mean()))

    arr = np.array(stats) if stats else np.array([0.0])
    print(f"\n转换完成 {len(stats)} 对 -> {out_root}")
    print(f"  水体占比 均值={arr.mean():.4f} 中位={np.median(arr):.4f} "
          f"最大={arr.max():.4f} 最小={arr.min():.4f}")
    print(f"  无水图 {int((arr < 1e-6).sum())}/{len(arr)}")
    print(f"\n下一步 —— 冗余度诊断（这是关键判断）:")
    print(f"  python -m src.diagnose_redundancy --root {out_root.parent} "
          f"--split {out_root.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=None, help="下载多少 MB（累计，可续传）")
    ap.add_argument("--chunk_mb", type=int, default=16)
    ap.add_argument("--verify", action="store_true", help="校验片段结构连续性")
    ap.add_argument("--fix", action="store_true", help="校验并截断损坏部分")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--out", default="data/processed/rivair/train")
    ap.add_argument("--max_side", type=int, default=1024, help="转换时长边缩放到，0=不缩放")
    args = ap.parse_args()

    if args.mb:
        download(args.mb, args.chunk_mb)
    if args.verify or args.fix:
        verify(truncate=args.fix)
    if args.extract:
        # 解包前先校验，避免在损坏数据上浪费时间
        if verify(truncate=False) >= 0:
            raise SystemExit("片段有损坏，先运行 --fix 截断后重新下载")
        splice_and_extract()
    if args.convert:
        convert(Path(args.out), args.max_side or None)
    if not any([args.mb, args.verify, args.fix, args.extract, args.convert]):
        ap.print_help()


if __name__ == "__main__":
    main()
