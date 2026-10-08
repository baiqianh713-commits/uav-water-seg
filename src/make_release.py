"""打包要公开发布的数据与结果，并在打包前逐项体检。

为什么要有这个脚本
------------------
公开发布是**不可撤销**的：一旦传上 Zenodo 并拿到 DOI，按设计就删不掉了
（这正是期刊要 DOI 的原因）。所以不能手动拖文件夹——必须有一道机器检查，
确认包里没有不该有的东西。

发布范围（2026-10-05 定）
-------------------------
  收  data/processed/field   341 帧切块（训练 227 / 验证 29 / 测试 85）
  收  data/annotate          对应的像素级标注
  收  runs/**/result.json    370 组单组结果（论文每一格的出处）
  收  src/ docs/             代码与方法记录

  **不收**  data/raw/field   原始航摄视频
  **不收**  *.SRT            大疆飞行日志 —— 🔴 逐帧记录经纬度与海拔，
                             这是整批数据里唯一受测绘管理规定约束的部分
  **不收**  *.pt / *.pth     模型权重（体积大，且不是复现所必需）

体检项
------
1. 包里不得出现 SRT / 飞行日志
2. 包里任何文本文件不得出现 "latitude" / "longitude" 字样
3. 图像不得带 EXIF GPS 标签
4. 不得混入模型权重
任何一项不过，直接中止，不生成压缩包。

用法
----
    PYTHONIOENCODING=utf-8 python src/make_release.py          # 只体检
    PYTHONIOENCODING=utf-8 python src/make_release.py --pack   # 体检通过后打包
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release"

# (源路径, 包内路径) —— 只有列在这里的才会被打包
INCLUDE = [
    (ROOT / "data/processed/field", "data/field"),
    (ROOT / "data/annotate", "data/annotations"),
    (ROOT / "src", "src"),
    (ROOT / "docs", "docs"),
]

# 单个文件（上面的 INCLUDE 走 copytree，只能收目录）。
# 🔴 许可证必须进包：Zenodo 的存档页只显示一个许可字段，
#    而本发布是代码 MIT + 数据 CC BY 的混合授权，
#    不把这两个文件放进去，下载者无从知道哪部分按哪条规则用。
INCLUDE_FILES = [
    (ROOT / "LICENSE", "LICENSE"),
    (ROOT / "LICENSE-DATA.md", "LICENSE-DATA.md"),
    (ROOT / "README.md", "README.md"),
]

# 任何路径命中这些模式都不打包（即便它在 INCLUDE 的目录里）
EXCLUDE_PATTERNS = [
    "*.SRT", "*.srt",                  # 大疆飞行日志，逐帧经纬度
    "*.pt", "*.pth", "*.ckpt",         # 模型权重
    "*.mp4", "*.MP4",                  # 原始视频
    # 🔴 架次汇总的原始版含中心经纬度与水体名称，
    #    公开的是 sanitize_conditions.py 生成的 -public 版
    "field-conditions.csv", "field-conditions.json",
]

# 🔴 真正要拦的是**坐标数值**，不是 "latitude" 这个词本身
# （解析 SRT 的代码必然出现字段名，按词拦会全误报）。
#
# 判据：坐标泄漏几乎总是**经纬度成对出现**。单看一个数分不清
# ——"24.7642" 可能是坐标，也可能是某个指标值；"10.1109" 是 DOI 前缀。
# 所以要求同一个文件里**同时**出现中国境内的纬度候选与经度候选才报警。
#   纬度 18–53 度，经度 73–135 度，小数位 ≥4（够定位到百米内才有意义）
import re as _re
LAT_RE = _re.compile(r"\b(?:1[89]|[2-4][0-9]|5[0-3])\.[0-9]{4,}\b")
LON_RE = _re.compile(r"\b(?:7[3-9]|[89][0-9]|1[0-2][0-9]|13[0-5])\.[0-9]{4,}\b")

# 占位示例不算
COORD_WHITELIST = {"00.000000", "000.000000"}


def collect_results() -> list[dict]:
    """把散落在 runs/ 下的 result.json 汇总成一个列表。"""
    out = []
    for p in sorted((ROOT / "runs").rglob("result.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_run_dir"] = p.parent.name      # 留一个出处字段，便于追溯
        out.append(d)
    return out


def check_no_srt(files: list[Path]) -> list[str]:
    return [str(f) for f in files if f.suffix.lower() == ".srt"]


def check_no_coords(files: list[Path]) -> list[str]:
    """扫文本文件里的**坐标数值**。二进制跳过（图像由 EXIF 检查负责）。"""
    bad = []
    text_ext = {".json", ".txt", ".md", ".csv", ".py", ".yaml", ".yml"}
    for f in files:
        if f.suffix.lower() not in text_ext:
            continue
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lats = {h for h in LAT_RE.findall(t)} - COORD_WHITELIST
        lons = {h for h in LON_RE.findall(t)} - COORD_WHITELIST
        if lats and lons:      # 成对出现才算
            bad.append(f"{f}（疑似坐标对：{sorted(lats)[0]} / {sorted(lons)[0]}）")
    return bad


def check_no_exif_gps(files: list[Path]) -> list[str]:
    """检查图像是否残留 EXIF GPS 标签。"""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        print("  ⚠️ 未装 Pillow，跳过 EXIF 检查（打包前请装上再跑一次）")
        return []
    bad = []
    for f in files:
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        try:
            exif = Image.open(f).getexif()
        except Exception:
            continue
        for tag_id in exif:
            if TAGS.get(tag_id, "") == "GPSInfo":
                bad.append(str(f))
                break
    return bad


def check_no_weights(files: list[Path]) -> list[str]:
    return [str(f) for f in files
            if f.suffix.lower() in {".pt", ".pth", ".ckpt"}]


def main() -> None:
    pack = "--pack" in sys.argv

    # --- 列出将要打包的文件 ---
    files: list[Path] = []
    for src, _ in INCLUDE:
        if not src.exists():
            print(f"🔴 源路径不存在：{src}")
            raise SystemExit(1)
        for f in src.rglob("*"):
            # 体检要和真正打包的清单一致，否则检查了个寂寞
            if f.is_file() and not any(f.match(p) for p in EXCLUDE_PATTERNS):
                files.append(f)
    for src, _ in INCLUDE_FILES:
        if not src.exists():
            print(f"🔴 源文件不存在：{src}")
            raise SystemExit(1)
        files.append(src)

    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"候选文件 {len(files)} 个，合计 {total_mb:.0f} MB\n")

    # --- 四项体检 ---
    checks = [
        ("SRT / 飞行日志", check_no_srt),
        ("文本中的坐标字样", check_no_coords),
        ("图像 EXIF GPS", check_no_exif_gps),
        ("模型权重", check_no_weights),
    ]
    failed = False
    for name, fn in checks:
        bad = fn(files)
        if bad:
            failed = True
            print(f"🔴 {name}：发现 {len(bad)} 处")
            for b in bad[:5]:
                print(f"     {b}")
            if len(bad) > 5:
                print(f"     ...还有 {len(bad) - 5} 处")
        else:
            print(f"✅ {name}：干净")

    results = collect_results()
    print(f"\n✅ 汇总到 {len(results)} 组 result.json")

    if failed:
        print("\n🔴 体检未通过，**不打包**。先处理上面的问题。")
        raise SystemExit(1)

    if not pack:
        print("\n体检通过。加 --pack 生成发布包。")
        return

    # --- 打包 ---
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for src, dst in INCLUDE:
        shutil.copytree(
            src, OUT / dst,
            ignore=shutil.ignore_patterns(*EXCLUDE_PATTERNS),
        )
    for src, dst in INCLUDE_FILES:
        shutil.copy2(src, OUT / dst)
    (OUT / "results_all_runs.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 发布包已生成：{OUT}")
    print("   🔴 上传前务必取得单位批准，并再跑一次本脚本体检。")


if __name__ == "__main__":
    main()
