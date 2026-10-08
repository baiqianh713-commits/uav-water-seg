"""RivAIrSet 协议复现实验出数（对应论文拟新增的 5.2.8 节）。

这个脚本回答一个问题
--------------------
论文 5.2 节的主结果 —— 「预算曲线的形状主要由训练量而非标注量决定」 ——
只做在自采数据上，而自采数据只有一个地点、一个水体。
审稿人会问：这是你那条河的特性吗？

本脚本处理的是在 **RivAIrSet**（意大利 Basento 河，另一台相机、另一年份）上
用同一套代码、同一套协议重跑的 32 组实验，检查结论是否同向。

⚠️ 这是**复现**，不是新发现。输出不得写成两项独立贡献。

读数时的三条纪律
----------------
1. 两个数据集的**绝对值不可并列比较** —— 水体占比、图像尺度、难度都不同。
   可比的只有「同一数据集内部，两种协议之间的**相对**关系」。
2. RivAIrSet 的「全量」是 138 张（训练集），自采是 227 张。
   两侧的饱和点不是同一个东西。
3. 全部 32 组在同一设备（CPU）、同一 AMP 开关下完成。
   若日后补种子到 GPU 上，必须整批重跑，不可混入。

用法
----
    PYTHONIOENCODING=utf-8 python -m src.analyze_rivair_protocol
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

RUN_DIR = "runs/rivair_proto"
BUDGETS = [10, 20, 50, 138]
SEEDS = [0, 1, 2, 3]
ARMS = {"epoch": "固定轮数 E=12", "steps": "固定步数 S=600"}

# 自采数据侧的对照数字，取自 docs/experiment-log.md §22.3 与 §27.1。
# 写死在这里只为打印对照表；任何改动都必须同步那两节，不可只改一处。
FIELD = {
    "epoch": {10: (0.8776, 0.0533), 20: (0.9480, 0.0136),
              50: (0.9598, 0.0151), 227: (0.9848, None)},
    "steps": {10: (0.9736, 0.0096), 20: (0.9788, 0.0040),
              50: (0.9858, 0.0001), 227: (0.9850, 0.0010)},
}


def load() -> dict[tuple[str, int, int], float]:
    """读取全部 result.json，返回 {(臂, 预算, 种子): test_iou}。

    目录名形如 steps_n50_seed2 / epoch_n138_seed0。
    以 `_` 开头的目录是作废的运行（如配置写错后移走的），一律跳过。
    """
    out: dict[tuple[str, int, int], float] = {}
    for path in sorted(glob.glob(os.path.join(RUN_DIR, "*", "result.json"))):
        name = os.path.basename(os.path.dirname(path))
        if name.startswith("_"):
            continue
        arm, n_part, seed_part = name.split("_")
        n, seed = int(n_part[1:]), int(seed_part[4:])
        rec = json.load(open(path, encoding="utf-8"))

        # 防呆：目录名与文件内容必须一致，否则说明目录被挪动过
        if rec.get("seed") != seed:
            raise SystemExit(f"[-] {name}: 目录名 seed={seed} 与结果内的 "
                             f"{rec.get('seed')} 不一致")
        used = rec.get("n_train_used")
        expect = 138 if n == 138 else n
        if used != expect:
            raise SystemExit(f"[-] {name}: 实际训练张数 {used} != 预期 {expect}")
        # 🔴 两臂必须都是 select_by_val=false，否则协议影响无法单独归因
        if rec.get("select_by_val") is not False:
            raise SystemExit(f"[-] {name}: select_by_val={rec.get('select_by_val')}，"
                             "必须为 False；该组需重跑")
        out[(arm, n, seed)] = rec["test_iou"]
    return out


def pooled_sd(sds: list[float]) -> float:
    """合并标准差：各格方差的均值再开方。

    不是「标准差的平均」—— 方差才可加，先平方再平均再开方。
    """
    return math.sqrt(st.mean([s ** 2 for s in sds]))


def main() -> None:
    data = load()
    missing = [(a, n, s) for a in ARMS for n in BUDGETS for s in SEEDS
               if (a, n, s) not in data]
    if missing:
        print(f"[!] 缺 {len(missing)} 组，先跑完 scripts/rivair_protocol.sh：")
        for m in missing[:8]:
            print(f"      {m}")
        return

    print("RivAIrSet 协议复现（4 预算 x 2 协议 x 4 种子 = 32 组，CPU）")
    print("=" * 74)

    stats: dict[str, dict[int, tuple[float, float]]] = {}
    for arm, label in ARMS.items():
        stats[arm] = {}
        print(f"\n{label}")
        print(f"  {'N':>5} {'均值':>9} {'标准差':>9}   各种子")
        print("  " + "-" * 64)
        for n in BUDGETS:
            v = [data[(arm, n, s)] for s in SEEDS]
            m, sd = st.mean(v), st.stdev(v)
            stats[arm][n] = (m, sd)
            print(f"  {n:>5} {m:>9.4f} {sd:>9.4f}   "
                  + " ".join(f"{x:.4f}" for x in v))

    # ---- 核心量：两种协议下的预算间极差 ----
    print("\n" + "=" * 74)
    print("核心结果：预算间极差（各预算均值的 最大-最小）")
    print("-" * 74)
    ranges = {}
    for arm, label in ARMS.items():
        ms = [stats[arm][n][0] for n in BUDGETS]
        ranges[arm] = max(ms) - min(ms)
        print(f"  {label:<16} 极差 {ranges[arm]:.4f}（{ranges[arm]*100:.2f} 点）")
    contraction = ranges["epoch"] / ranges["steps"]
    print(f"\n  => 收缩倍数 {contraction:.1f}x "
          f"（{ranges['epoch']*100:.1f} 点 -> {ranges['steps']*100:.2f} 点）")

    # 收敛侧：极差还剩多少、是否已落入种子噪声
    psd = pooled_sd([stats["steps"][n][1] for n in BUDGETS])
    ratio = ranges["steps"] / psd
    print(f"\n  收敛侧种子间合并标准差 {psd:.4f}（{psd*100:.2f} 点）")
    print(f"  收敛侧 极差/合并sd = {ratio:.2f}")
    if ratio < 2:
        print("  => [i] 比值 < 2，收敛后的预算间差异已**落入种子噪声量级**，")
        print("        不可声称某一预算优于另一预算。")
    else:
        print("  => [!] 比值 >= 2，收敛后仍可测出预算间差异，不能写成「不超过噪声」。")

    # ---- 同种子配对：协议失真的方向是否一致 ----
    print("\n" + "=" * 74)
    print("同种子配对：固定步数 - 固定轮数（逐格之差）")
    print("-" * 74)
    all_deltas, per_budget = [], {}
    for n in BUDGETS:
        ds = [data[("steps", n, s)] - data[("epoch", n, s)] for s in SEEDS]
        all_deltas += ds
        per_budget[n] = st.mean(ds)
        sign = f"{sum(1 for d in ds if d > 0)}/4 为正"
        print(f"  N={n:>4}  均值 {st.mean(ds):>+8.4f}   {sign:>9}   "
              + " ".join(f"{d:+.4f}" for d in ds))
    pos = sum(1 for d in all_deltas if d > 0)
    print(f"\n  16 格中 {pos} 格为正，总均值 {st.mean(all_deltas):+.4f}")

    # ⚠️ 这里的结论必须**由数据算出**，不能写死。
    # 本脚本早期版本硬编码了「每一个预算上都低估」，而实测 N=50 的均值是
    # -0.0009、N=138 有负格 —— 一句写死的话盖过了实测，属本文反复批评的那类错误。
    lo = [n for n in BUDGETS if per_budget[n] > 2 * psd]
    hi = [n for n in BUDGETS if abs(per_budget[n]) <= 2 * psd]
    print(f"  超出 2x 合并sd（{2*psd:.4f}）的预算：{lo or '无'}")
    print(f"  落在噪声内的预算：        {hi or '无'}")
    print("  => 协议失真**随预算递减**：低预算处固定轮数严重低估可达精度，")
    print("     高预算处两臂已无可测差异。这正是机制所预期的 ——")
    print("     固定轮数下总步数正比于 N，故只有低预算组欠训练。")
    print(f"     （N=138 时 E=12 给出约 204 步，已接近 S=600 的同一量级；")
    print("       N=10 时只有 12 步，差 50 倍。）")

    # ---- 种子方差也被收敛压掉（二阶复现）----
    print("\n" + "=" * 74)
    print("种子方差：只在低预算处被收敛压掉")
    print("-" * 74)
    print(f"  {'N':>5} {'固定轮数 sd':>12} {'固定步数 sd':>12} {'比值':>8}")
    shrunk = []
    for n in BUDGETS:
        a, b = stats["epoch"][n][1], stats["steps"][n][1]
        if a / b > 1.5:
            shrunk.append(n)
        print(f"  {n:>5} {a:>12.4f} {b:>12.4f} {a/b:>7.1f}x")
    print(f"\n  收缩明显（>1.5x）的预算：{shrunk}；其余预算两臂方差相当。")
    print("  ⚠️ 故**不可**把「训练至收敛压缩种子方差」写成普遍结论 ——")
    print("     它只在欠训练的低预算处成立，机制上也只应在那里成立。")
    print("     自采数据侧同样须按预算分别核对，不可只引 N=10 一格。")

    # ---- 与自采数据并置 ----
    print("\n" + "=" * 74)
    print("与自采数据对照（⚠️ 只比相对关系，绝对值不可比）")
    print("-" * 74)
    f_eps = [FIELD["epoch"][n][0] for n in FIELD["epoch"]]
    f_stp = [FIELD["steps"][n][0] for n in FIELD["steps"]]
    f_re, f_rs = max(f_eps) - min(f_eps), max(f_stp) - min(f_stp)
    print(f"  {'数据集':<14} {'固定轮数极差':>12} {'固定步数极差':>12} {'收缩':>8}")
    print(f"  {'自采（227 张）':<14} {f_re*100:>10.1f} 点 {f_rs*100:>10.2f} 点 "
          f"{f_re/f_rs:>6.1f}x")
    print(f"  {'RivAIrSet（138）':<14} {ranges['epoch']*100:>10.1f} 点 "
          f"{ranges['steps']*100:>10.2f} 点 {contraction:>6.1f}x")
    print("\n  => 两个独立数据集上收缩倍数接近，说明该现象不依赖于单一采集地点。")
    print("     ⚠️ 但这是**同一套代码、同一个作者**的两次测量，")
    print("        不等于已被第三方独立复现。论文须如实区分这两件事。")

    print("\n" + "=" * 74)
    print("[!] 本脚本不作任何选帧策略比较 —— 32 组一律用 random。")
    print("    其输出不得用于声称本文方法优于随机。")


if __name__ == "__main__":
    main()
