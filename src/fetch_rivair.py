"""分段下载 RivAIrSet 并提取其中的影像/标签。

背景
----
RivAIrSet (La Salandra et al., Data in Brief 2025) 是 RivAIr 系统论文配套的公开数据集：
  Zenodo: https://doi.org/10.5281/zenodo.17284942
  许可: CC-BY-4.0（完全开放，**仅需引用**）
  内容: 7630 张无人机影像 + JSON 多边形标注，6 个架次(2019-2024)，
        同一段 3km 河道，正下视，GSD 约 1-1.6cm

问题：整个数据集打包成**单个 101 GB 的 .rar**，实测下载约 1 MB/s，
全量需约 30 小时，不现实。

解决：Zenodo 支持 HTTP Range，且 RAR5 是**顺序存储**的 ——
下载文件开头的一段，就能得到最前面若干个文件。而它们恰好是
同一架次的**连续航拍帧**，正是本研究需要的高冗余数据。

⚠️ 需要系统安装 WinRAR / unrar 才能解包。

用法
----
  # 下载前 2GB（约 200 张连续帧），断点续传
  python -m src.fetch_rivair --gb 2

  # 解包已下载的部分
  python -m src.fetch_rivair --extract

  # 探测归档结构，定位 labels 所在偏移
  python -m src.fetch_rivair --probe 18 19 20
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/17284942/files/RIVER_WATER_DATASET.rar"
OUT_DIR = Path("data/raw/rivair")
PART = OUT_DIR / "part.rar"
GB = 1024 ** 3

# RAR5 头部里文件名是 UTF-8 明文，可直接正则匹配
NAME_PAT = re.compile(
    rb"RIVER_WATER_DATASET[\\/][0-9]{4}[\\/](?:images|labels)[\\/][A-Za-z0-9_]+\.(?:JPG|json)"
)

UNRAR_CANDIDATES = [
    "C:/Program Files/WinRAR/UnRAR.exe",
    "C:/Program Files/WinRAR/Rar.exe",
    "C:/Program Files (x86)/WinRAR/UnRAR.exe",
    "unrar",
]


def find_unrar() -> str | None:
    for c in UNRAR_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def fetch_range(start: int, end: int, timeout: int = 120, retries: int = 3) -> bytes | None:
    """取 [start, end] 字节。Zenodo 偶发断连，失败重试。"""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                URL, headers={"Range": f"bytes={start}-{end}",
                              "User-Agent": "research-script"}
            )
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:
            if attempt == retries - 1:
                print(f"    [!] 取 {start}-{end} 失败: {type(e).__name__}")
                return None
            time.sleep(3 * (attempt + 1))  # 退避重试
    return None


def download(target_bytes: int, chunk_mb: int = 32) -> None:
    """分块下载到 PART，支持断点续传（按已有文件大小继续）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    have = PART.stat().st_size if PART.exists() else 0
    if have >= target_bytes:
        print(f"已有 {have / GB:.2f} GB，达到目标 {target_bytes / GB:.2f} GB，无需下载")
        return

    print(f"目标 {target_bytes / GB:.2f} GB，已有 {have / GB:.2f} GB，继续下载...")
    chunk = chunk_mb * 1024 * 1024
    t0, start_have = time.time(), have

    with open(PART, "ab") as f:
        while have < target_bytes:
            end = min(have + chunk, target_bytes) - 1
            data = fetch_range(have, end)
            if data is None:
                print("    连续失败，稍后重跑本命令即可续传")
                break
            f.write(data)
            f.flush()
            have += len(data)
            elapsed = time.time() - t0
            speed = (have - start_have) / 1e6 / max(elapsed, 1e-9)
            remain = (target_bytes - have) / 1e6 / max(speed, 1e-9)
            print(f"  {have / GB:6.2f}/{target_bytes / GB:.2f} GB  "
                  f"{speed:.2f} MB/s  剩余约 {remain / 60:.0f} 分钟")

    print(f"本次结束，当前 {PART.stat().st_size / GB:.2f} GB")


def extract() -> None:
    """解包。部分归档末尾必然残缺，unrar 会报错但前面的完整文件仍能解出。"""
    unrar = find_unrar()
    if not unrar:
        raise SystemExit("未找到 unrar/WinRAR。请安装 WinRAR 或 7-Zip 后重试。")
    if not PART.exists():
        raise SystemExit(f"{PART} 不存在，先运行 --gb 下载")

    print(f"用 {unrar} 解包（末尾残缺文件的报错可忽略）...")
    # -o+ 覆盖同名；-inul 静默；末尾截断会返回非零退出码，属预期
    subprocess.run([unrar, "x", "-o+", "-inul", str(PART), str(OUT_DIR) + "/"],
                   capture_output=True)

    imgs = sorted(OUT_DIR.rglob("*.JPG"))
    jsons = sorted(OUT_DIR.rglob("*.json"))
    print(f"解出影像 {len(imgs)} 张，标注 {len(jsons)} 个")
    if imgs:
        print(f"  首个: {imgs[0].name}   末个: {imgs[-1].name}")
    if not jsons:
        print("  [!] 尚未解出标注文件 —— labels 目录排在 images 之后，")
        print("      需下载更大范围，或用 --probe 定位其偏移。")


def probe(offsets_gb: list[float], window_mb: int = 12) -> None:
    """在指定偏移处取一个窗口，看里面有哪些文件名。

    窗口必须大于单个文件的平均大小（约 9.5MB），否则可能整窗都落在
    某个文件的数据区内，一个头部都碰不到。
    """
    print(f"探测窗口 {window_mb} MB（单文件约 9.5MB，窗口需更大才能命中头部）\n")
    for g in offsets_gb:
        off = int(g * GB)
        data = fetch_range(off, off + window_mb * 1024 * 1024 - 1)
        if data is None:
            print(f"  {g:>6.1f} GB: 请求失败")
            continue
        hits = [h.decode("utf-8", "ignore") for h in NAME_PAT.findall(data)]
        if hits:
            # 只关心目录层级，不必列出全部文件名
            dirs = sorted({h.rsplit("/", 1)[0].rsplit("\\", 1)[0] for h in hits})
            print(f"  {g:>6.1f} GB: {len(hits)} 条  目录={dirs}")
            print(f"            首={hits[0].split('/')[-1].split(chr(92))[-1]}  "
                  f"末={hits[-1].split('/')[-1].split(chr(92))[-1]}")
        else:
            print(f"  {g:>6.1f} GB: 未命中文件名（窗口可能整个落在某文件数据区内）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gb", type=float, default=None, help="下载到多少 GB（累计，可续传）")
    ap.add_argument("--chunk_mb", type=int, default=32)
    ap.add_argument("--extract", action="store_true", help="解包已下载部分")
    ap.add_argument("--probe", type=float, nargs="+", default=None,
                    help="探测这些 GB 偏移处的文件名")
    ap.add_argument("--window_mb", type=int, default=12, help="探测窗口大小")
    args = ap.parse_args()

    if args.probe:
        probe(args.probe, args.window_mb)
        return
    if args.gb:
        download(int(args.gb * GB), args.chunk_mb)
    if args.extract:
        extract()
    if not any([args.gb, args.extract, args.probe]):
        ap.print_help()


if __name__ == "__main__":
    main()
