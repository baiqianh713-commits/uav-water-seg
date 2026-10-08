"""标注质检：把掩膜叠回原图，肉眼确认边界对不对。

为什么需要
----------
标注一致性比标注精度更影响实验结论 —— 所有策略共用同一套标注，
系统性偏差会被抵消，而随机的不一致会变成噪声淹没策略间差异
（LoveDA 那次就是被方差淹没的，见 docs/experiment-log.md）。

所以要在标注早期、以及每标完一批时，抽查叠加图确认：
  - 边界有没有系统性偏内/偏外
  - 有没有整帧漏标
  - 水下暗影有没有被误挖成"非水"
  - 栈道/芦苇丛有没有被误包进来

用法
----
    # 直接在标注目录上跑（读图 + AnyLabeling JSON）
    python -m src.qc_overlay --input data/annotate --out runs/qc

    # 只抽查若干张
    python -m src.qc_overlay --input data/annotate --out runs/qc --limit 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]


def find_image(js: Path, data: dict) -> Path | None:
    """找 JSON 对应的原图：先按 imagePath，再按同名不同扩展名。"""
    cands = []
    if data.get("imagePath"):
        cands.append(js.parent / Path(data["imagePath"]).name)
    cands += [js.with_suffix(e) for e in IMG_EXTS]
    return next((c for c in cands if c.exists()), None)


def edge_bias(img: np.ndarray, mask: np.ndarray, ring_px: int = 20) -> float:
    """估计"贴着边界、本该算水却被漏掉"的面积，占已标水体的比例。

    做法：把掩膜向外扩 ring_px 得到一圈环带，环带里凡是既不像植被
    也不像栈道的像素，就当作被漏标的水（多数是芦苇投在水面上的阴影带）。

    为什么盯这一项：漏掉的是**阴影里的水**，而 C 条件整组是暗光拍摄。
    在 A/B 上把暗处判成非水，偏差会顺着光照这个自变量传进 C 组结果，
    变成与贡献点 2 相关的混杂——这类偏差不会像普通系统偏差那样被抵消。

    阈值按本次外业场景（城市公园木栈道 + 芦苇 + 浑浊水面）定，换场景要重调。
    返回值是粗估的上界量级，用于发现趋势，不是精确指标。
    """
    ring = (cv2.dilate(mask, np.ones((ring_px * 2 + 1,) * 2, np.uint8)) - mask).astype(bool)
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    reed = (g - np.maximum(b, r)) > 12   # 芦苇与草：绿通道明显占优
    deck = (r - b) > 18                  # 木栈道：红棕色，红远大于蓝
    missed = ring & ~reed & ~deck
    return float(missed.sum()) / max(1, int(mask.sum()))


def water_like_ratio(img: np.ndarray) -> float:
    """粗估画面里"看着像水"的面积占比，用来抓空标注。

    不是分割器，只排除两类最显眼的非水地物（植被、木栈道），
    剩下的灰绿低饱和区域当作水。宁可高估——它只用于提醒"这帧你可能忘了标"。
    """
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    reed = (g - np.maximum(b, r)) > 12
    deck = (r - b) > 18
    bright = img.max(axis=2).astype(np.int16) > 235   # 过曝的路面/反光白斑
    return float((~reed & ~deck & ~bright).mean())


def render(img: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """半透明青色填充 + 黄色描边。

    填充让人看清"模型会把哪片区域当成水"，
    描边让人看清边界具体落在哪一像素上 —— 两者缺一不可。
    """
    out = img.copy()
    color = np.zeros_like(img)
    color[..., 1] = 255  # 青绿：水体在原图里偏绿，用高饱和色才分得开
    color[..., 2] = 255
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/annotate", help="含图与 AnyLabeling JSON 的目录")
    ap.add_argument("--out", default="runs/qc", help="叠加图输出目录")
    ap.add_argument("--alpha", type=float, default=0.35, help="填充透明度")
    ap.add_argument("--limit", type=int, default=0, help="最多渲染几张，0 表示全部")
    args = ap.parse_args()

    root = Path(args.input)
    js = sorted(p for p in root.rglob("*.json")
                if p.name not in {"condition_map.json", "frames_meta.json"})
    if not js:
        raise SystemExit(f"{root} 下没有标注 JSON")
    if args.limit:
        js = js[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'帧名':<52}{'水体占比':>10}{'多边形':>8}{'边界漏标':>10}")
    print("-" * 82)

    biases: list[float] = []
    suspect_empty: list[str] = []
    for p in js:
        d = json.loads(p.read_text(encoding="utf-8"))
        ip = find_image(p, d)
        if ip is None:
            print(f"  [!] 找不到原图: {p.name}")
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [!] 读不出原图: {ip.name}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        n_poly = 0
        for s in d.get("shapes", []):
            if s.get("label", "").strip().lower() != "water":
                continue
            pts = np.array(s.get("points", []), dtype=np.float64)
            if len(pts) < 3:
                continue
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            n_poly += 1

        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"),
                    render(img, mask, args.alpha),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        if n_poly == 0:
            # 空标注最危险：它会变成"整图皆背景"的样本，主动教模型"水=背景"，
            # 比漏标更糟。多半是画完没按 F 就翻页，自动保存写了个空 JSON。
            wl = water_like_ratio(img)
            flag = "  <- 疑似漏标！" if wl > 0.15 else ""
            print(f"{p.stem:<52}{0.0:>10.4f}{0:>8}{'—':>9}"
                  f"  (像水区域 {wl:.0%}){flag}")
            if wl > 0.15:
                suspect_empty.append(p.stem)
            continue

        bias = edge_bias(img, mask)
        biases.append(bias)
        print(f"{p.stem:<52}{mask.mean():>10.4f}{n_poly:>8}{bias:>9.1%}")

    if suspect_empty:
        print(f"\n[!] {len(suspect_empty)} 帧没有任何多边形，但画面里大片像水——"
              f"几乎肯定是忘了按 F 完成对象就翻页了，必须重标：")
        for s in suspect_empty:
            print(f"      {s}")

    if biases:
        med = float(np.median(biases))
        print(f"\n边界漏标中位数 {med:.1%}", end="  ")
        if med > 0.03:
            print("[!] 偏内——边界可能画在了芦苇阴影带内侧。"
                  "应画在芦苇秆根部，阴影带算水。")
        else:
            print("[+] 正常")

    print(f"\n叠加图已写入 {out.resolve()}")
    print("检查要点：边界有无系统性偏内/偏外；水下暗影与芦苇阴影带是否被误挖；"
          "栈道与成片芦苇是否被误包。")


if __name__ == "__main__":
    main()
