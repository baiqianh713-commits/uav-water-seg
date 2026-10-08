"""无人机视频抽帧 + SRT 遥测解析。

为什么视频比照片更适合本研究
----------------------------
本文方法的核心是「冗余感知选帧」—— 从大量高度相似的帧里挑出值得标注的。
视频天然满足这一点：30fps 下相邻帧几乎完全相同，正是方法的目标场景。
照片模式反而要刻意制造重复。

⚠️ 关于 SRT 字幕文件
--------------------
大疆视频的飞行遥测（GPS、相对高度、云台角度）**不在 MP4 里**，
而在同名的 `.SRT` 字幕文件中。该开关在 DJI Fly 里默认常常是**关闭**的：

    DJI Fly → 设置 → 拍摄 → 「视频字幕 / Video Caption」→ 打开

没有 SRT 就没有任何高度和 GPS 信息，贡献点 2（多条件预算分配）无法进行。
**起飞前先录 10 秒，确认 SD 卡里出现 .SRT 文件。**

目录约定
--------
    data/raw/field/
        C1_湟水河直段_50m_90d_晴_上午/
            DJI_0001.MP4
            DJI_0001.SRT
        C2_.../

一个条件一个文件夹，文件夹名即条件 ID —— 脚本据此自动生成
`condition_map.json`（贡献点 2 需要），无需另外手工记录对应关系。

用法
----
  # 先看视频信息与 SRT 是否可用，不抽帧
  python -m src.extract_video --input data/raw/field --inspect

  # 按秒抽帧（默认每 1 秒 1 帧）
  python -m src.extract_video --input data/raw/field --out data/raw/field_frames --fps 1.0

  # 高冗余模式：每 0.5 秒 1 帧，帧间更相似
  python -m src.extract_video --input data/raw/field --out data/raw/field_frames --fps 2.0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV"}

# 大疆 SRT 每条字幕的正文形如（坐标为占位示例，非真实架次数据）：
#   [latitude: 00.000000] [longitude: 000.000000] [rel_alt: 50.200 abs_alt: 2250.300]
#   [gb_yaw: 12.3 gb_pitch: -90.0 gb_roll: 0.0]
# 不同固件版本字段名略有差异，这里用宽松匹配，取不到就留空。
SRT_PATTERNS = {
    "纬度": r"\[?latitude\s*[:=]\s*([-\d.]+)",
    "经度": r"\[?long(?:i)?tude\s*[:=]\s*([-\d.]+)",
    "相对高度AGL_m": r"rel_alt\s*[:=]\s*([-\d.]+)",
    "绝对高度_m": r"abs_alt\s*[:=]\s*([-\d.]+)",
    "云台俯仰角_deg": r"gb_pitch\s*[:=]\s*([-\d.]+)",
}
# 时间轴形如 00:00:01,000 --> 00:00:02,000
TIME_PAT = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(path: Path) -> list[dict]:
    """解析 SRT，返回按时间排序的遥测记录。

    每条记录含 start_sec 与可提取到的遥测字段。
    固件差异较大，取不到的字段留 None，不报错。
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    records: list[dict] = []
    # 字幕块之间以空行分隔
    for block in re.split(r"\n\s*\n", text):
        m = TIME_PAT.search(block)
        if not m:
            continue
        h, mi, s, ms = (int(x) for x in m.groups()[:4])
        rec: dict = {"start_sec": h * 3600 + mi * 60 + s + ms / 1000.0}
        for key, pat in SRT_PATTERNS.items():
            mm = re.search(pat, block, re.IGNORECASE)
            rec[key] = float(mm.group(1)) if mm else None
        records.append(rec)

    records.sort(key=lambda r: r["start_sec"])
    return records


def telemetry_at(records: list[dict], t: float) -> dict:
    """取时间 t 处的遥测。用最近的一条记录（SRT 通常 1Hz，足够）。"""
    if not records:
        return {k: None for k in SRT_PATTERNS}
    best = min(records, key=lambda r: abs(r["start_sec"] - t))
    return {k: best.get(k) for k in SRT_PATTERNS}


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in VIDEO_EXTS)


def inspect(root: Path) -> None:
    """只检查，不抽帧。飞完第一时间跑这个确认数据可用。"""
    vids = find_videos(root)
    if not vids:
        raise SystemExit(f"{root} 下没有找到 MP4/MOV 文件")

    print(f"找到 {len(vids)} 个视频\n")
    total_frames_est = 0
    missing_srt = []

    for v in vids:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"  [!] 无法打开: {v.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        dur = n / fps if fps else 0

        srt = v.with_suffix(".SRT")
        if not srt.exists():
            srt = v.with_suffix(".srt")
        recs = parse_srt(srt)
        if not recs:
            missing_srt.append(v.name)

        cond = v.parent.name if v.parent != root else "(根目录)"
        print(f"  {v.name}")
        print(f"     条件文件夹 : {cond}")
        print(f"     分辨率     : {w}x{h}   {fps:.1f} fps   {n} 帧   {dur / 60:.1f} 分钟")
        if recs:
            alts = [r["相对高度AGL_m"] for r in recs if r["相对高度AGL_m"] is not None]
            pit = [r["云台俯仰角_deg"] for r in recs if r["云台俯仰角_deg"] is not None]
            print(f"     SRT        : {len(recs)} 条遥测 [OK]")
            if alts:
                print(f"       飞行高度 : {min(alts):.1f} ~ {max(alts):.1f} m")
            else:
                print(f"       [!] SRT 中无 rel_alt 字段（固件差异），高度需人工补")
            if pit:
                print(f"       云台角度 : {min(pit):.0f} ~ {max(pit):.0f} 度")
        else:
            print(f"     SRT        : [!] 缺失或无法解析")
        print(f"     按 1fps 可抽 : 约 {int(dur)} 帧")
        total_frames_est += int(dur)
        print()

    print("=" * 56)
    print(f"按 1 fps 抽帧，预计总计约 {total_frames_est} 帧")
    if missing_srt:
        print(f"\n[!] 以下 {len(missing_srt)} 个视频缺少可用 SRT：")
        for n in missing_srt[:10]:
            print(f"      {n}")
        print("    → 这些视频没有高度/GPS 信息。")
        print("    → 检查 DJI Fly：设置 → 拍摄 → 「视频字幕」是否打开。")
        print("    → 已拍的只能靠人工飞行日志补记高度。")
    else:
        print("\n[OK] 所有视频都有可用的 SRT 遥测")


def extract(root: Path, out: Path, target_fps: float, jpeg_quality: int) -> None:
    vids = find_videos(root)
    if not vids:
        raise SystemExit(f"{root} 下没有找到 MP4/MOV 文件")

    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)

    condition_map: dict[str, str] = {}
    meta_rows: list[dict] = []

    for v in vids:
        # 条件 ID 取自所在文件夹名；视频直接放在根目录时退回用文件名
        cond = v.parent.name if v.parent != root else v.stem

        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"[!] 无法打开 {v.name}，跳过")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        srt = v.with_suffix(".SRT")
        if not srt.exists():
            srt = v.with_suffix(".srt")
        recs = parse_srt(srt)

        # 每隔 step 帧取一张。step<1 时退化为逐帧
        step = max(1, int(round(fps / max(target_fps, 1e-6))))
        print(f"{v.name}  [{cond}]  {fps:.1f}fps  {n_total} 帧  每 {step} 帧取 1 张")

        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                t = idx / fps
                # 帧名里带条件与原始帧号，便于溯源，也便于场景级划分
                stem = f"{cond}__{v.stem}__f{idx:06d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                tel = telemetry_at(recs, t)
                condition_map[stem] = cond
                meta_rows.append({
                    "帧名": stem, "条件": cond, "源视频": v.name,
                    "原始帧号": idx, "时间_s": round(t, 2), **tel,
                })
                saved += 1
            idx += 1
        cap.release()
        print(f"    -> 抽出 {saved} 帧")

    # condition_map.json 供 allocation.py 直接使用（贡献点 2）
    with open(out / "condition_map.json", "w", encoding="utf-8") as f:
        json.dump(condition_map, f, indent=1, ensure_ascii=False)
    with open(out / "frames_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, indent=1, ensure_ascii=False)

    conds = sorted(set(condition_map.values()))
    print(f"\n{'=' * 56}")
    print(f"共抽出 {len(meta_rows)} 帧，{len(conds)} 个条件")
    for c in conds:
        k = sum(1 for v_ in condition_map.values() if v_ == c)
        print(f"  {c:<40} {k:>5} 帧")

    with_alt = sum(1 for r in meta_rows if r.get("相对高度AGL_m") is not None)
    print(f"\n有飞行高度的帧: {with_alt}/{len(meta_rows)}"
          f" ({with_alt / max(len(meta_rows), 1):.0%})")
    if with_alt == 0:
        print("  [!] 全部缺失 —— SRT 未开启或无 rel_alt 字段，高度需人工补")

    print(f"\n输出: {out}")
    print(f"  images/              抽出的帧")
    print(f"  condition_map.json   条件映射（贡献点 2 直接可用）")
    print(f"  frames_meta.json     每帧的遥测明细")
    print(f"\n下一步：跑冗余度诊断，确认帧间冗余是否足够")
    print(f"  python -m src.diagnose_redundancy --root {out} --split .")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/raw/field", help="视频根目录")
    ap.add_argument("--out", default="data/raw/field_frames", help="抽帧输出目录")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="每秒抽几帧。1.0=每秒1帧；2.0=每0.5秒1帧（冗余更高）")
    ap.add_argument("--quality", type=int, default=95, help="JPEG 质量")
    ap.add_argument("--inspect", action="store_true", help="只检查视频与 SRT，不抽帧")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    if args.inspect:
        inspect(root)
    else:
        extract(root, Path(args.out), args.fps, args.quality)


if __name__ == "__main__":
    main()
