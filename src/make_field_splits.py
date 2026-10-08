"""自采数据的防泄漏划分：按架次内**时间段**切分，边界留缓冲带。

为什么不能用 make_splits.py
---------------------------
`make_splits.py` 按场景随机划分，前提是场景数量足够多。本次外业只有 **5 个架次**，
且条件 C（暗光）只有 1 个架次 —— 随机分会让整个 C 条件落进某一个集合，
test 里就没有暗光样本，贡献点 2 直接做不成。

为什么必须留缓冲带
------------------
相邻帧是近重复（本数据最近邻距离中位数 0.0766，81.5% 的帧存在近重复）。
若 train 的最后一帧和 val 的第一帧只差 3 秒，模型在验证时看到的
几乎就是训练过的画面，指标虚高。这与 `make_splits` 里讲的图块泄漏是同一类问题，
只是沿时间轴发生。

    架次内帧序： [-------- train --------][gap][-- val --][gap][---- test ----]
                                          ↑ 丢弃         ↑ 丢弃

缓冲带上的帧被标记为 `drop`，**不需要标注**，能省下标注工时。

划分协议（论文 4.x 节应照此描述）
---------------------------------
每个架次按帧号升序切成 train/val/test 三段，段间丢弃 `--buffer` 帧作为缓冲；
所有架次、所有采集条件都同时出现在三个集合中，
保证 test 覆盖全部条件，同时相邻帧不跨集合。

用法
----
    # 1) 先算划分（只看结果，不动文件），并导出待标注清单
    python -m src.make_field_splits --pool data/processed/field_pool_s3/_all/images \\
        --out data/processed/field_splits.json

    # 2) 标注 + convert_field_labels 之后，把图和掩膜按划分摆好
    python -m src.make_field_splits --apply \\
        --splits data/processed/field_splits.json \\
        --src data/processed/field_flat --dst data/processed/field
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_stem(stem: str) -> tuple[str, str, int]:
    """帧名 `A_50m_normal__DJI_..._0281_D__f000120` -> (条件, 架次, 原始帧号)。"""
    parts = stem.split("__")
    if len(parts) < 3:
        return "(未知)", "(未知)", 0
    try:
        idx = int(parts[2].lstrip("f"))
    except ValueError:
        idx = 0
    return parts[0], parts[1], idx


def plan(pool_dir: Path, ratios: tuple[float, float, float], buffer: int) -> dict:
    """算出每一帧属于 train / val / test / drop。"""
    stems = sorted(p.stem for p in pool_dir.glob("*.png"))
    if not stems:
        raise SystemExit(f"{pool_dir} 下没有 png")

    by_video: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for s in stems:
        _, vid, idx = parse_stem(s)
        by_video[vid].append((idx, s))

    assign: dict[str, str] = {}
    r_tr, r_va, _ = ratios

    for vid in sorted(by_video):
        seq = [s for _, s in sorted(by_video[vid])]  # 按原始帧号升序 = 时间序
        n = len(seq)
        # 两个切点。缓冲带从切点向两侧各扩 buffer//2，保证 train/val/test 不相邻。
        cut1 = int(round(n * r_tr))
        cut2 = int(round(n * (r_tr + r_va)))
        half = max(0, buffer // 2)

        for i, s in enumerate(seq):
            if i < cut1 - half:
                assign[s] = "train"
            elif i < cut1 + half:
                assign[s] = "drop"
            elif i < cut2 - half:
                assign[s] = "val"
            elif i < cut2 + half:
                assign[s] = "drop"
            else:
                assign[s] = "test"

    return assign


def report(assign: dict[str, str]) -> None:
    """按 split × 条件 打印交叉表 —— 要确认每个条件在三个集合里都有样本。"""
    cross: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_video: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s, sp in assign.items():
        cond, vid, _ = parse_stem(s)
        cross[cond][sp] += 1
        per_video[vid][sp] += 1

    order = ["train", "val", "test", "drop"]
    print(f"\n{'条件':<18}" + "".join(f"{o:>8}" for o in order) + f"{'合计':>8}")
    print("-" * 60)
    for c in sorted(cross):
        row = cross[c]
        print(f"{c:<18}" + "".join(f"{row[o]:>8}" for o in order)
              + f"{sum(row.values()):>8}")
    tot = defaultdict(int)
    for row in cross.values():
        for o in order:
            tot[o] += row[o]
    print("-" * 60)
    print(f"{'合计':<18}" + "".join(f"{tot[o]:>8}" for o in order)
          + f"{sum(tot.values()):>8}")

    print(f"\n{'架次':<34}" + "".join(f"{o:>8}" for o in order))
    print("-" * 66)
    for v in sorted(per_video):
        row = per_video[v]
        print(f"{v:<34}" + "".join(f"{row[o]:>8}" for o in order))

    need = sum(v for k, v in tot.items() if k != "drop")
    print(f"\n需要标注的帧数: {need}"
          f"（缓冲带 {tot['drop']} 帧不用标）")

    # 关键校验：test 里必须每个条件都有样本，否则贡献点 2 做不成
    missing = [c for c in cross if cross[c]["test"] == 0]
    if missing:
        print(f"\n  [!] 以下条件在 test 中没有样本，需调整比例: {', '.join(missing)}")
    else:
        print(f"  [+] 三个集合覆盖全部 {len(cross)} 个采集条件")


def apply_split(splits_file: Path, src: Path, dst: Path) -> None:
    """把已标注转换好的 images/ + masks/ 按划分复制到 train/val/test。"""
    assign = json.loads(splits_file.read_text(encoding="utf-8"))
    n = defaultdict(int)
    missing = 0

    for stem, sp in assign.items():
        if sp == "drop":
            continue
        si, sm = src / "images" / f"{stem}.png", src / "masks" / f"{stem}.png"
        if not (si.exists() and sm.exists()):
            missing += 1
            continue
        for sub, f in (("images", si), ("masks", sm)):
            d = dst / sp / sub
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, d / f"{stem}.png")
        n[sp] += 1

    print(f"已写出 -> {dst}")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:<6} {n[sp]:>5} 对")
    if missing:
        print(f"  [!] {missing} 帧尚未标注或转换，已跳过"
              f" —— 用 convert_field_labels --progress 看标注进度")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/processed/field_pool_s3/_all/images")
    ap.add_argument("--out", default="data/processed/field_splits.json")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.15, 0.25],
                    help="train val test，按每个架次内的帧数比例")
    ap.add_argument("--buffer", type=int, default=6,
                    help="每个切点处丢弃的帧数（前后各一半），防相邻帧跨集合")
    ap.add_argument("--apply", action="store_true", help="按已有划分摆放文件")
    ap.add_argument("--splits", default="data/processed/field_splits.json")
    ap.add_argument("--src", default="data/processed/field_flat")
    ap.add_argument("--dst", default="data/processed/field")
    args = ap.parse_args()

    if args.apply:
        apply_split(Path(args.splits), Path(args.src), Path(args.dst))
        return

    assign = plan(Path(args.pool), tuple(args.ratios), args.buffer)
    report(assign)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assign, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n划分已写入 {out}")

    # 待标注清单：缓冲带的帧不在其中
    todo = sorted(s for s, sp in assign.items() if sp != "drop")
    todo_f = out.with_name("annotate_todo.txt")
    todo_f.write_text("\n".join(todo), encoding="utf-8")
    print(f"待标注清单 {todo_f}（{len(todo)} 帧）")


if __name__ == "__main__":
    main()
