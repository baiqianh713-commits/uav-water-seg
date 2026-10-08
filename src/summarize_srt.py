"""从大疆 SRT 字幕重建每个架次的**客观**采集条件。

为什么需要这一步
----------------
外业时人工记的文件夹名/飞行日志经常与实际不符（高度记错、时段记错）。
而 SRT 是飞控逐帧写入的，是客观记录。论文里的条件表必须以 SRT 为准，
否则「多条件预算分配」这个贡献点的自变量本身就是错的。

本脚本只读不写原始数据，输出一张条件表（CSV + JSON），供：
  - 论文 4.1 节「数据采集」表格直接引用
  - 人工确认后重命名条件文件夹

SRT 字段说明（本机固件实际格式；坐标与海拔为占位示例，非真实架次数据）
----------------------------------------------------------------------
    [iso: 100] [shutter: 1/1000.0] [fnum: 1.8] [ev: 0] [color_md: default]
    [focal_len: 24.00] [latitude: 00.000000] [longitude: 000.000000]
    [rel_alt: 50.000 abs_alt: 0000.000] [ct: 4974]

    rel_alt  相对起飞点高度(AGL)，**这才是飞行高度**
    abs_alt  海拔，不能当飞行高度用
    ct       色温(K)
    注意：本固件**不写云台角度 gb_pitch**，角度无法从 SRT 获得。

时间戳单独占一行，形如 `2026-09-29 16:34:33.806`。

用法
----
    python -m src.summarize_srt --input data/raw/field
    python -m src.summarize_srt --input data/raw/field --out docs/field-conditions
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

# 逐帧遥测字段。用宽松匹配，取不到就留 None，不报错。
FIELD_PATTERNS = {
    "lat": r"latitude\s*[:=]\s*([-\d.]+)",
    "lon": r"long(?:i)?tude\s*[:=]\s*([-\d.]+)",
    "rel_alt": r"rel_alt\s*[:=]\s*([-\d.]+)",
    "abs_alt": r"abs_alt\s*[:=]\s*([-\d.]+)",
    "iso": r"\biso\s*[:=]\s*([\d.]+)",
    "ev": r"\bev\s*[:=]\s*([-\d.]+)",
    "ct": r"\bct\s*[:=]\s*([\d.]+)",
    "focal_len": r"focal_len\s*[:=]\s*([\d.]+)",
}
# 快门是 1/1000.0 这种分数形式，单独处理成秒
SHUTTER_PAT = re.compile(r"shutter\s*[:=]\s*1/([\d.]+)")
TIMESTAMP_PAT = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.\d+)?")


def parse_srt(path: Path) -> list[dict]:
    """把 SRT 解析成逐帧记录列表。字幕块之间以空行分隔。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    records: list[dict] = []

    for block in re.split(r"\n\s*\n", text):
        if not block.strip():
            continue
        rec: dict = {}

        m = TIMESTAMP_PAT.search(block)
        if m:
            rec["time"] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")

        for key, pat in FIELD_PATTERNS.items():
            mm = re.search(pat, block, re.IGNORECASE)
            rec[key] = float(mm.group(1)) if mm else None

        ms = SHUTTER_PAT.search(block)
        # 存成分母：1/1000 记 1000。数值越大快门越快，间接反映现场有多亮。
        rec["shutter_denom"] = float(ms.group(1)) if ms else None

        if rec:
            records.append(rec)

    return records


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两个经纬度之间的地表距离（米）。用于算航迹长度与场景间距。"""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _stats(values: list[float]) -> dict:
    """最小/最大/均值。空列表返回 None，避免除零。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "mean": round(sum(vals) / len(vals), 3),
    }


def summarize_one(srt: Path, root: Path) -> dict:
    """汇总单个架次。"""
    recs = parse_srt(srt)
    if not recs:
        return {"文件夹": srt.parent.name, "源文件": srt.name, "错误": "SRT 无法解析"}

    times = [r["time"] for r in recs if r.get("time")]
    lats = [r["lat"] for r in recs if r.get("lat") is not None]
    lons = [r["lon"] for r in recs if r.get("lon") is not None]

    # 航迹长度：逐帧累加相邻点距离。60fps 下单帧位移很小，累加即近似航迹。
    track_m = 0.0
    for i in range(1, len(lats)):
        track_m += haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])

    alt = _stats([r["rel_alt"] for r in recs])
    dur_s = (max(times) - min(times)).total_seconds() if len(times) > 1 else 0.0

    return {
        "文件夹": srt.parent.name,
        "源文件": srt.with_suffix(".MP4").name,
        "起始时间": min(times).strftime("%Y-%m-%d %H:%M:%S") if times else None,
        "结束时间": max(times).strftime("%H:%M:%S") if times else None,
        "时长_s": round(dur_s, 1),
        "遥测帧数": len(recs),
        "飞行高度AGL_m": alt["mean"],
        "高度范围_m": f"{alt['min']}~{alt['max']}",
        "中心纬度": round(sum(lats) / len(lats), 6) if lats else None,
        "中心经度": round(sum(lons) / len(lons), 6) if lons else None,
        "航迹长度_m": round(track_m, 1),
        "平均速度_mps": round(track_m / dur_s, 2) if dur_s > 0 else None,
        "ISO均值": _stats([r["iso"] for r in recs])["mean"],
        "快门分母均值": _stats([r["shutter_denom"] for r in recs])["mean"],
        "色温K均值": _stats([r["ct"] for r in recs])["mean"],
        "焦距mm": _stats([r["focal_len"] for r in recs])["mean"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default="data/raw/field", help="含条件子文件夹的根目录")
    ap.add_argument("--out", default="docs/field-conditions",
                    help="输出前缀，会写 .json 与 .csv 两个文件")
    args = ap.parse_args()

    root = Path(args.input)
    # Windows 文件系统大小写不敏感，两个 glob 会匹配到同一批文件。
    # 用 resolve() 后的真实路径去重，否则每个架次会被统计两遍。
    srts = sorted({p.resolve() for p in
                   list(root.rglob("*.SRT")) + list(root.rglob("*.srt"))})
    if not srts:
        raise SystemExit(f"{root} 下没有找到 SRT 文件")

    rows = [summarize_one(s, root) for s in srts]
    # 按实际起飞时间排序 —— 这才是真实的架次顺序，与文件夹编号无关
    rows.sort(key=lambda r: r.get("起始时间") or "")

    # 逐架次明细
    for r in rows:
        print(f"\n【{r['文件夹']}】  {r['源文件']}")
        if "错误" in r:
            print(f"   [!] {r['错误']}")
            continue
        print(f"   实际时间   : {r['起始时间']} ~ {r['结束时间']}  ({r['时长_s']}s)")
        print(f"   实际高度   : {r['飞行高度AGL_m']} m   (范围 {r['高度范围_m']})")
        print(f"   中心坐标   : {r['中心纬度']}, {r['中心经度']}")
        print(f"   航迹长度   : {r['航迹长度_m']} m   平均速度 {r['平均速度_mps']} m/s")
        print(f"   曝光       : ISO {r['ISO均值']}  快门 1/{r['快门分母均值']:.0f}"
              f"  色温 {r['色温K均值']:.0f}K")

    # 场景间距矩阵：判拍的是同一地点
    print(f"\n{'=' * 64}")
    print("架次间中心点距离（米）—— 距离近说明是同一地点，不构成独立场景")
    valid = [r for r in rows if r.get("中心纬度") is not None]
    names = [r["文件夹"].split("_")[0] for r in valid]
    print("        " + "".join(f"{n:>9}" for n in names))
    for i, ri in enumerate(valid):
        line = f"{names[i]:<8}"
        for j, rj in enumerate(valid):
            d = haversine_m(ri["中心纬度"], ri["中心经度"],
                            rj["中心纬度"], rj["中心经度"])
            line += f"{d:>9.0f}"
        print(line)

    # 高度分组：看真实高度有几个可区分的水平
    print(f"\n{'=' * 64}")
    print("真实高度分组（按 5m 容差归并）")
    groups: dict[float, list[str]] = {}
    for idx, r in enumerate(valid):
        a = r["飞行高度AGL_m"]
        key = next((k for k in groups if abs(k - a) < 5), a)
        groups.setdefault(key, []).append(names[idx])
    for a, ns in sorted(groups.items()):
        print(f"   约 {a:.0f} m : {', '.join(ns)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    # CSV 用 utf-8-sig，Excel 打开才不乱码
    import csv
    keys = [k for k in rows[0] if k != "错误"]
    with open(out.with_suffix(".csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n已写出: {out.with_suffix('.json')}")
    print(f"        {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
