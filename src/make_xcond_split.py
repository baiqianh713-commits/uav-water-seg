"""跨条件划分：用常光（A+B）训练，只在暗光（C）上测试。

为什么要做这个
--------------
同分布设置下本方法与随机选帧无显著差异，根因很可能不是方法，而是**数据太容易**：
测试集与训练集是同一片水面、同一天、同一条航线，random 在 N=20 已达 0.9480，
而全量 baseline 才 0.9848 —— **总共只有 3.7 个点的空间**，任何方法最多赢这么多。

跨条件评估制造出真实的难度：模型只见过常光（ISO ~100），
却要在暗光（ISO 1969，暗 19 倍）上分割。这同时直接服务于贡献点 2 ——
"标注预算该怎么在采集条件间分配"，前提正是条件间存在泛化鸿沟。

划分协议
--------
    train : A_50m_normal + B_30m_normal 的 train 段   (93 + 88 = 181)
    val   : A + B 的 val 段                           (12 + 11 =  23)
    test  : C_50m_dark 的**全部**帧                   (69)

C 组完全不参与训练，故无泄漏；把 C 的全部 69 帧都用作测试是为了提高测试统计量，
这比只用 C_test 的 17 帧可靠得多。

⚠️ 局限：C 组仅 1 个架次，测试集内部高度相关，等效独立样本远少于 69。
   论文中须说明，不可按 69 个独立样本做显著性声称。

用法
----
    python -m src.make_xcond_split \\
        --src data/processed/field_flat \\
        --splits data/processed/field_splits.json \\
        --dst data/processed/field_xcond
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def cond_of(stem: str) -> str:
    """帧名 `A_50m_normal__DJI_..._0281_D__f000000` -> 条件 ID。"""
    return stem.split("__")[0] if "__" in stem else "(未知)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/processed/field_flat",
                    help="convert_field_labels 的输出，含 images/ 与 masks/")
    ap.add_argument("--splits", default="data/processed/field_splits.json")
    ap.add_argument("--dst", default="data/processed/field_xcond")
    ap.add_argument("--train-conds", nargs="+", default=["A_50m_normal", "B_30m_normal"],
                    help="用于训练/验证的条件")
    ap.add_argument("--test-cond", default="C_50m_dark",
                    help="仅用于测试的条件，其全部帧都进 test")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    assign = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    plan: dict[str, str] = {}
    for stem, sp in assign.items():
        c = cond_of(stem)
        if c == args.test_cond:
            # 留出条件的**全部**帧都进测试集，包括原划分里的 drop（缓冲带）。
            # 缓冲带本是为防相邻帧跨集合而设，此处 C 完全不参与训练，无此顾虑。
            plan[stem] = "test"
        elif c in args.train_conds and sp in ("train", "val"):
            plan[stem] = sp
        # 其余（A/B 的 test 与 drop 段）丢弃：它们既非训练数据，
        # 也不属于本实验要考察的留出条件

    n = defaultdict(int)
    missing = 0
    for stem, sp in plan.items():
        si, sm = src / "images" / f"{stem}.png", src / "masks" / f"{stem}.png"
        if not (si.exists() and sm.exists()):
            missing += 1
            continue
        for sub, f in (("images", si), ("masks", sm)):
            d = dst / sp / sub
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, d / f"{stem}.png")
        n[sp] += 1

    print(f"跨条件划分已写出 -> {dst}")
    print(f"  训练条件: {', '.join(args.train_conds)}")
    print(f"  留出条件: {args.test_cond}（全部帧用作测试）\n")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:<6} {n[sp]:>5} 对")
    if missing:
        print(f"  [!] {missing} 帧缺失，已跳过")

    out = dst / "xcond_splits.json"
    out.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n划分明细 {out}")
    print(f"\n下一步 —— 先建 baseline（全部 {n['train']} 张训练，确立跨条件上界）:")
    print(f"  python -m src.train --config configs/field_xcond.yaml "
          f"--budget 0 --strategy all --seed 0 --out runs/field_xcond_baseline")


if __name__ == "__main__":
    main()
