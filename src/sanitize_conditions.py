"""生成 field-conditions 的**可公开版**：去掉绝对位置，保留科学证据。

为什么不能直接删掉这个文件
--------------------------
`docs/field-conditions.*` 同时装着两类东西：

  🔴 必须去掉的：中心纬度 / 中心经度（6 位小数 ≈ 0.1 m 精度）、
     文件夹名里的水体名与地点（「湟水河直段」「湿地」）

  ✅ 必须留下的：它是论文两条结论的**原始出处**——
     1. 「时段变量不存在」：文件夹名标称「上午/正午」，
        而实测时间戳全部落在 36 分钟内（16:34–17:10）
     2. 「条件需客观重建」：标称「100m」的架次实测飞行高度 49.589 m
     3. 「5 个架次中心点两两间距不超过 201 m」：这是**相对**距离，
        不泄露位置，且是上面第 1 条的佐证。
        注意口径：两两最大间距 201.4 m，距质心最远 136.3 m。
        正文写「两两间距 ≤201 m」，不要写「落在 201 m 半径内」
        ——后者虽不假，但把半径说大了 65 m。

所以做法是脱敏而不是删除：绝对坐标换算成「距五架次质心的距离」，
地名换成论文里用的 A/B/C 条件组代号。

用法
----
    PYTHONIOENCODING=utf-8 python src/sanitize_conditions.py
输出 docs/field-conditions-public.csv，由 make_release.py 打包。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/field-conditions.csv"
OUT = ROOT / "docs/field-conditions-public.csv"

# 原文件夹名 → 论文中的条件组代号。
# 映射关系与 data/processed/field 的文件名前缀一致（那边已经是 A/B/C）。
GROUP_MAP = {
    "C1_湟水河直段_50m_90d_晴_上午": "A_50m_normal",
    "C4_湟水河直段_100m_90d_晴_上午": "A_50m_normal",
    "C2_湟水河直段_50m_90d_晴_正午": "B_30m_normal",
    "C5_湿地_30m_90d_晴_上午": "B_30m_normal",
    "C3_湟水河弯道_50m_90d_晴_上午": "C_50m_dark",
}

# 要丢弃的列（绝对位置）
DROP_COLS = {"中心纬度", "中心经度", "源文件"}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点间大圆距离（米）。半径取 WGS84 平均值。"""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main() -> None:
    rows = list(csv.DictReader(SRC.read_text(encoding="utf-8-sig").splitlines()))

    lats = [float(r["中心纬度"]) for r in rows]
    lons = [float(r["中心经度"]) for r in rows]
    # 质心（架次少且集中，按算术平均即可，不必做球面质心）
    clat, clon = sum(lats) / len(lats), sum(lons) / len(lons)

    dists = [haversine_m(clat, clon, la, lo) for la, lo in zip(lats, lons)]
    print("各架次距质心距离（m）：", ", ".join(f"{d:.1f}" for d in dists))
    print(f"最大两两间距：{max(haversine_m(lats[i], lons[i], lats[j], lons[j]) for i in range(len(rows)) for j in range(len(rows))):.1f} m")

    out_cols = [c for c in rows[0] if c not in DROP_COLS]
    out_cols = ["条件组"] + out_cols + ["距质心距离_m"]

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for r, d in zip(rows, dists):
            o = {c: r[c] for c in out_cols if c in r}
            o["条件组"] = GROUP_MAP.get(r["文件夹"], "?")
            o["文件夹"] = r["文件夹"].split("_")[0]   # 只留 C1..C5，去掉地名
            o["距质心距离_m"] = f"{d:.1f}"
            w.writerow(o)

    print(f"\n✅ 已生成 {OUT}")
    print("   去掉：中心经纬度、源文件名、文件夹名中的地名")
    print("   保留：时间戳、实测飞行高度、航迹长度、相机参数、相对距离")


if __name__ == "__main__":
    main()
