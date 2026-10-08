"""无人机照片 EXIF/XMP 批量提取 —— 为论文数据集描述章节整理拍摄条件。

用法
----
  # 基本用法，输出 CSV
  python -m src.extract_exif --input "D:/照片/20260920_湟水河" --out flight_log.csv

  # 输出 Excel，并预填场景与天气（省得事后逐行填）
  python -m src.extract_exif --input "D:/照片/20260920_湟水河" --out flight.xlsx \
      --scene 河流 --weather 晴

  # 递归处理多个子文件夹（每个子文件夹一个架次）
  python -m src.extract_exif --input "D:/照片" --recursive --out all_flights.csv


⚠️ 关于"飞行高度"——这一点很容易搞错
--------------------------------------
EXIF 标准字段 `GPSAltitude` 记录的是**海拔高度**（相对海平面），
不是飞行高度。在西宁（海拔约 2200m）飞 50m，该字段会写约 2250，而不是 50。

真正的飞行高度（相对起飞点，即 AGL）在大疆写入的 XMP 元数据
`drone-dji:RelativeAltitude` 字段里。本脚本两者都提取：

  GPS海拔_m      <- EXIF GPSAltitude，海拔，高原地区会是很大的数
  相对高度AGL_m  <- XMP RelativeAltitude，**论文里该用这个**

若 `相对高度AGL_m` 整列为空，说明照片不是大疆机型或 XMP 被剥离
（例如经过某些软件重新压缩），此时只能依赖你的人工飞行日志。

同时提取云台俯仰角 `GimbalPitchDegree`，即拍摄角度：
  -90 = 正下视（垂直 90°）    -45 = 倾斜 45°
这正好对应采集清单里的角度分层，可自动填充，不必手记。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image, ExifTags

# EXIF 标签名 -> 数字 ID 的反查表
_TAG_ID = {name: tid for tid, name in ExifTags.TAGS.items()}
_GPS_ID = {name: tid for tid, name in ExifTags.GPSTAGS.items()}

IMAGE_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

# 输出列顺序。天气/场景留空供事后手工补充。
COLUMNS = [
    "文件名",
    "拍摄时间",
    "纬度",
    "经度",
    "GPS海拔_m",
    "相对高度AGL_m",
    "云台俯仰角_deg",
    "宽",
    "高",
    "相机型号",
    "天气",
    "场景",
]


def _to_float(value) -> float | None:
    """EXIF 数值可能是 IFDRational、tuple(分子,分母) 或普通数字，统一转 float。"""
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            num, den = value
            return float(num) / float(den) if den else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _dms_to_decimal(dms, ref: str | None) -> float | None:
    """把 EXIF 的度/分/秒格式转成十进制度。

    ref 为 'S' 或 'W' 时取负值（南纬/西经）。
    """
    if not dms or len(dms) != 3:
        return None
    d, m, s = (_to_float(x) for x in dms)
    if d is None or m is None or s is None:
        return None
    dec = d + m / 60.0 + s / 3600.0
    if ref and str(ref).upper().strip() in ("S", "W"):
        dec = -dec
    return dec


def read_exif(path: Path) -> dict:
    """读取单张照片的 EXIF。任何字段缺失都返回 None，不抛异常。"""
    out = {
        "拍摄时间": None, "纬度": None, "经度": None, "GPS海拔_m": None,
        "宽": None, "高": None, "相机型号": None,
    }
    try:
        with Image.open(path) as img:
            out["宽"], out["高"] = img.size
            exif = img.getexif()
            if not exif:
                return out

            # 拍摄时间优先用 DateTimeOriginal，它在 ExifIFD 子目录里
            dto = None
            try:
                sub = exif.get_ifd(_TAG_ID.get("ExifOffset", 0x8769))
                dto = sub.get(_TAG_ID.get("DateTimeOriginal"))
            except Exception:
                pass
            # 退回顶层 DateTime（文件修改时间，精度略差但聊胜于无）
            if not dto:
                dto = exif.get(_TAG_ID.get("DateTime"))
            if dto:
                # EXIF 格式为 "2026:09:20 08:31:05"，转成更通用的写法
                out["拍摄时间"] = str(dto).strip().replace(":", "-", 2)

            model = exif.get(_TAG_ID.get("Model"))
            make = exif.get(_TAG_ID.get("Make"))
            if model or make:
                out["相机型号"] = " ".join(
                    str(x).strip() for x in (make, model) if x
                ).strip()

            # GPS 信息在独立的 GPS IFD 里
            try:
                gps = exif.get_ifd(_TAG_ID.get("GPSInfo", 0x8825))
            except Exception:
                gps = None
            if gps:
                lat = _dms_to_decimal(
                    gps.get(_GPS_ID.get("GPSLatitude")),
                    gps.get(_GPS_ID.get("GPSLatitudeRef")),
                )
                lon = _dms_to_decimal(
                    gps.get(_GPS_ID.get("GPSLongitude")),
                    gps.get(_GPS_ID.get("GPSLongitudeRef")),
                )
                alt = _to_float(gps.get(_GPS_ID.get("GPSAltitude")))
                # GPSAltitudeRef == 1 表示低于海平面
                if alt is not None:
                    ref = gps.get(_GPS_ID.get("GPSAltitudeRef"))
                    try:
                        if ref is not None and int(
                            ref[0] if isinstance(ref, bytes) else ref
                        ) == 1:
                            alt = -alt
                    except (TypeError, ValueError, IndexError):
                        pass
                out["纬度"], out["经度"], out["GPS海拔_m"] = lat, lon, alt
    except Exception as e:
        # 损坏文件或非图像：整行留空，继续处理下一张
        print(f"  [!] 读取 EXIF 失败，跳过该字段: {path.name} ({type(e).__name__})")

    return out


# XMP 里同一字段有两种写法：属性式和标签式，都要匹配
def _xmp_value(xmp: str, field: str) -> str | None:
    for pattern in (
        rf'drone-dji:{field}\s*=\s*"([^"]*)"',      # 属性式
        rf"<drone-dji:{field}>([^<]*)</drone-dji:{field}>",  # 标签式
    ):
        m = re.search(pattern, xmp, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def read_xmp(path: Path, max_bytes: int = 256 * 1024) -> dict:
    """从文件头部提取大疆 XMP 字段。

    只读前 256KB：XMP 包总是写在 JPEG 文件开头附近，
    没必要把 5MB 的整张照片读进内存。
    """
    out = {"相对高度AGL_m": None, "云台俯仰角_deg": None}
    try:
        with open(path, "rb") as f:
            head = f.read(max_bytes)
        start = head.find(b"<x:xmpmeta")
        if start < 0:
            return out
        end = head.find(b"</x:xmpmeta>", start)
        blob = head[start : end + 12] if end > 0 else head[start:]
        xmp = blob.decode("utf-8", errors="ignore")

        rel = _xmp_value(xmp, "RelativeAltitude")
        pitch = _xmp_value(xmp, "GimbalPitchDegree")
        # 值形如 "+50.20"，float() 能正确处理前导加号
        for key, raw in (("相对高度AGL_m", rel), ("云台俯仰角_deg", pitch)):
            if raw:
                try:
                    out[key] = float(raw)
                except ValueError:
                    pass
    except Exception:
        # XMP 缺失是常见情况（非大疆机型），静默返回空值
        pass
    return out


def collect(input_dir: Path, recursive: bool) -> list[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix in IMAGE_EXTS)


def write_csv(rows: list[dict], out_path: Path) -> None:
    # utf-8-sig：带 BOM，Excel 双击打开中文才不会乱码
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


def write_xlsx(rows: list[dict], out_path: Path) -> None:
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("需要 openpyxl 才能输出 xlsx：pip install openpyxl\n"
                         "或把 --out 改成 .csv 后缀。")
    from openpyxl.styles import Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "飞行记录"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in rows:
        ws.append([("" if r.get(k) is None else r.get(k)) for k in COLUMNS])

    # 冻结首行并按内容粗调列宽，几百行时翻看方便得多
    ws.freeze_panes = "A2"
    for i, name in enumerate(COLUMNS, start=1):
        width = max(len(name) * 2, 12)
        if name in ("文件名", "拍摄时间", "相机型号"):
            width = 22
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

    wb.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, help="照片文件夹")
    ap.add_argument("--out", default="flight_log.csv", help="输出 .csv 或 .xlsx")
    ap.add_argument("--recursive", action="store_true", help="递归处理子文件夹")
    ap.add_argument("--scene", default="", help="预填“场景”列，如 河流/湖泊/水塘/湿地")
    ap.add_argument("--weather", default="", help="预填“天气”列，如 晴/阴/多云")
    args = ap.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"文件夹不存在: {input_dir}")

    paths = collect(input_dir, args.recursive)
    if not paths:
        raise SystemExit(f"{input_dir} 下没有找到 jpg/jpeg 文件"
                         + ("（已递归查找）" if args.recursive else "，试试加 --recursive"))

    print(f"找到 {len(paths)} 张照片，开始提取...\n")

    rows: list[dict] = []
    for i, p in enumerate(paths, 1):
        row = {"文件名": p.name}
        row.update(read_exif(p))
        row.update(read_xmp(p))
        row["天气"] = args.weather
        row["场景"] = args.scene
        rows.append(row)
        if i % 50 == 0 or i == len(paths):
            print(f"  已处理 {i}/{len(paths)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".xlsx":
        write_xlsx(rows, out_path)
    else:
        write_csv(rows, out_path)

    # ---------------- 完整度统计 ----------------
    total = len(rows)

    def missing(key: str) -> int:
        return sum(1 for r in rows if r.get(key) in (None, ""))

    no_gps = sum(1 for r in rows if r.get("纬度") is None or r.get("经度") is None)
    no_abs_alt = missing("GPS海拔_m")
    no_rel_alt = missing("相对高度AGL_m")
    no_time = missing("拍摄时间")
    no_pitch = missing("云台俯仰角_deg")

    print(f"\n{'=' * 56}")
    print(f"处理完成：共 {total} 张")
    print(f"{'=' * 56}")
    print(f"  缺少 GPS 坐标       : {no_gps:>5} 张 ({no_gps / total:.1%})")
    print(f"  缺少 GPS 海拔       : {no_abs_alt:>5} 张 ({no_abs_alt / total:.1%})")
    print(f"  缺少 相对高度(AGL)  : {no_rel_alt:>5} 张 ({no_rel_alt / total:.1%})")
    print(f"  缺少 拍摄时间       : {no_time:>5} 张 ({no_time / total:.1%})")
    print(f"  缺少 云台俯仰角     : {no_pitch:>5} 张 ({no_pitch / total:.1%})")

    # 高度字段的可用性提示 —— 这是论文里最关键的一列
    if no_rel_alt == total:
        print("\n  [!] 相对高度(AGL) 整列为空。可能原因：")
        print("      1) 不是大疆机型，或固件未写入 XMP")
        print("      2) 照片经过压缩/转存，XMP 被剥离（务必用原始文件）")
        print("      -> 论文中的飞行高度只能依赖你的人工飞行日志。")
        if no_abs_alt < total:
            print("      注意：GPS海拔 是**海拔**不是飞行高度，高原地区会是 2000+，不能直接用。")
    elif no_rel_alt > 0:
        print(f"\n  [~] {no_rel_alt} 张缺少相对高度，需人工补填。")
    else:
        vals = [r["相对高度AGL_m"] for r in rows if r.get("相对高度AGL_m") is not None]
        print(f"\n  [OK] 相对高度(AGL) 完整，范围 {min(vals):.1f} ~ {max(vals):.1f} m")

    if no_pitch < total:
        pitches = [r["云台俯仰角_deg"] for r in rows if r.get("云台俯仰角_deg") is not None]
        near_nadir = sum(1 for v in pitches if v <= -80)
        print(f"  [OK] 云台俯仰角范围 {min(pitches):.0f} ~ {max(pitches):.0f} 度"
              f"（约 -90=正下视，-45=倾斜45度）；正下视约 {near_nadir} 张")

    print(f"\n表格已保存: {out_path}")
    print("  “天气”“场景”两列供你事后手工补充"
          + ("（已按 --weather/--scene 预填）" if (args.weather or args.scene) else ""))


if __name__ == "__main__":
    main()
