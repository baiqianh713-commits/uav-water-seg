"""汇总收敛侧（S=600）的多种子结果 —— 用于填 paper-ch5.md 5.2.5 的【待填 sd】。

要回答什么
----------
5.2.5 节的核心推论是「各预算训练至收敛后，预算间差异不超过种子噪声」。
该推论此前只有**单种子**支撑（极差 0.65 点），而固定轮数侧 N=20 的
10 种子标准差就有 0.0136 —— 0.65 点落在单种子噪声量级内，结论立不住。

本脚本把 4 个种子的结果汇总为每个预算的 mean ± sd，使该陈述可被量化检验：
若「预算间极差」小于「种子间标准差」，则「差异不超过种子噪声」成立。

数据来源
--------
  seed 0  : runs/conv_grid/n{N}_s600/result.json      （曲线族实验，10-03 02:35 完成）
  seed 1-3: runs/conv_seeds/n{N}_s600_seed{S}/result.json（补种子实验，10-03 12:32 启动）

两批实验的配置必须一致才能合并统计，所以脚本会**逐项核对** budget / 目标步数 /
strategy，不一致即报错退出，而不是悄悄把不可比的数字平均到一起。

⚠️ 本脚本不做策略比较，只描述收敛侧的预算间差异。
不得用它的输出声称某一预算优于另一预算。

运行
----
    python -m src.analyze_converged
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BUDGETS = [10, 20, 50, 227]
SEEDS = [0, 1, 2, 3]
TARGET_STEPS = 600

GRID_DIR = Path("runs/conv_grid")       # seed 0
SEEDS_DIR = Path("runs/conv_seeds")     # seed 1~3

# 固定轮数侧的 10 种子标准差，用作「种子噪声」的参照量级。
# 来源：runs/field_sweep 的 random 策略，由 src/analyze_sweep.py 产出。
FIXED_EPOCH_SD = {10: 0.0533, 20: 0.0136, 50: 0.0151}


def result_path(n: int, seed: int) -> Path:
    """seed 0 来自曲线族目录，其余来自补种子目录 —— 两批的命名规则不同。"""
    if seed == 0:
        return GRID_DIR / f"n{n}_s{TARGET_STEPS}" / "result.json"
    return SEEDS_DIR / f"n{n}_s{TARGET_STEPS}_seed{seed}" / "result.json"


def load_one(n: int, seed: int) -> dict | None:
    """读一组结果并核对配置。缺失返回 None（实验可能还在跑）。"""
    fp = result_path(n, seed)
    if not fp.exists():
        return None
    r = json.loads(fp.read_text(encoding="utf-8"))

    # 配置核对：预算与种子必须与目录名一致，否则说明目录与内容对不上。
    # N=227 是全量，脚本里用 `--strategy all --budget 0`，故 budget 记为 0。
    expect_budget = 0 if n == 227 else n
    if r.get("budget") != expect_budget:
        raise SystemExit(
            f"{fp}: budget 记为 {r.get('budget')}，按目录名应为 {expect_budget}。"
            "目录与内容不一致，须先查清。"
        )
    if r.get("seed") != seed:
        raise SystemExit(
            f"{fp}: seed 记为 {r.get('seed')}，按目录名应为 {seed}。"
        )
    return r


def main() -> None:
    table: dict[int, list[float]] = {}
    missing: list[str] = []

    for n in BUDGETS:
        vals = []
        for s in SEEDS:
            r = load_one(n, s)
            if r is None:
                missing.append(f"N={n} seed={s}")
                continue
            vals.append(r["test_iou"])
        table[n] = vals

    print(f"收敛侧多种子汇总（目标步数 S={TARGET_STEPS}）")
    print("=" * 72)
    print(f"{'预算 N':>8}{'种子数':>8}{'均值':>10}{'标准差':>10}"
          f"{'最小':>10}{'最大':>10}   固定轮数侧 sd")
    for n in BUDGETS:
        v = np.array(table[n])
        if len(v) == 0:
            print(f"{n:>8}{0:>8}        —         —         —         —")
            continue
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        ref = FIXED_EPOCH_SD.get(n)
        ref_s = f"{ref:.4f}" if ref is not None else "—（全量仅单次）"
        print(f"{n:>8}{len(v):>8}{v.mean():>10.4f}{sd:>10.4f}"
              f"{v.min():>10.4f}{v.max():>10.4f}   {ref_s}")

    # ---- 核心判据：预算间极差 vs 种子间噪声 ----
    done = {n: np.array(v) for n, v in table.items() if len(v) == len(SEEDS)}
    if len(done) == len(BUDGETS):
        means = np.array([done[n].mean() for n in BUDGETS])
        sds = np.array([done[n].std(ddof=1) for n in BUDGETS])
        spread = means.max() - means.min()
        pooled = float(np.sqrt((sds ** 2).mean()))   # 合并标准差

        print("\n" + "=" * 72)
        print(f"预算间极差（各预算均值的最大-最小） = {spread:.4f}（{spread*100:.2f} 点）")
        print(f"种子间合并标准差                     = {pooled:.4f}（{pooled*100:.2f} 点）")
        print(f"比值 极差 / 合并sd                   = {spread / pooled:.2f}")
        if spread <= pooled:
            print("\n  => 预算间极差不超过种子噪声。5.2.5 的陈述成立：")
            print("     「各预算训练至收敛后，其差异不超过种子噪声」")
        else:
            print(f"\n  => [!] 预算间极差为种子噪声的 {spread / pooled:.2f} 倍，")
            print("     不能再写成「差异不超过种子噪声」。")
            print("     应改为给出实测极差与标准差，让读者自行判断，")
            print("     并说明单调性是否仍然缺失（见下）。")

        # 单调性：固定轮数侧是单调上升的，收敛后是否仍然如此
        mono = bool(np.all(np.diff(means) > 0))
        print(f"\n  单调性（均值是否随预算严格递增）：{'是' if mono else '否'}")
        print(f"  各预算均值：" + "  ".join(f"N={n}:{done[n].mean():.4f}"
                                            for n in BUDGETS))
        if not mono:
            lo = BUDGETS[int(np.argmin(means))]
            print(f"  最低者为 N={lo} —— 单调性缺失，与固定轮数侧的单调上升形成对照。")

        print("\n可直接填入 paper-ch5.md 5.2.5 的表述：")
        parts = ", ".join(f"$N={n}$ 为 {done[n].mean():.4f} ± {done[n].std(ddof=1):.4f}"
                          for n in BUDGETS)
        print(f"  「本文另以 4 个种子在 600 步处复核，各预算的均值与标准差为"
              f"{parts}。」")
    else:
        print("\n" + "=" * 72)
        print(f"实验尚未跑完，缺 {len(missing)} 组：{', '.join(missing)}")
        print("待 scripts/converged_seeds.sh 收官后重跑本脚本。")

    print("\n[!] 本脚本只描述收敛侧的预算间差异，不作策略比较；")
    print("   其输出不得用于声称某一预算优于另一预算。")


if __name__ == "__main__":
    main()
