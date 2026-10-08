"""把一个 sweep 目录里的 result.json 汇总成论文能用的表 + 配对显著性检验。

为什么要单独写这个脚本（而不是每次临时敲 python）
--------------------------------------------------
1. **不能读 `runs/all_results.jsonl`**。那是追加日志，重跑会把同一配置记两次，
   统计出来的"种子数"是假的（RivAIr 那次 hybrid@20 被算成 36 个种子，实际只有 5 个）。
   唯一可信来源是每个实验目录下的 `result.json`，一个目录一条结果。
2. 三套网格（固定轮数 / 跨条件 / 固定步数）必须用**完全一样**的统计口径，
   否则协议之间的对比不成立。写成脚本才能保证这一点。

为什么用配对检验而不是两组独立样本的 t 检验
--------------------------------------------
每个种子同时决定了"选哪些帧"和"模型权重怎么初始化"。
同一个种子下 random 与 kcenter 的网络初始化是一样的，
所以两者之差把初始化这项噪声**消掉了** —— 这正是配对设计的意义。
若按独立样本做，初始化方差会留在分母里，把真实差异淹掉。

Wilcoxon 符号秩检验作为补充：它不假设差值服从正态分布。
注意 n=5 时它的最小可能 p 值就是 0.0625，即"5 个种子全部同向"也到不了 0.05。
所以 5 种子下看 Wilcoxon 的 p 值是否 <0.05 没有意义，要看 t 检验 + 同向比例。

用法
----
    python -m src.analyze_sweep --root runs/field_sweep
    python -m src.analyze_sweep --root runs/field_xcond_sweep --baseline runs/field_xcond_baseline
    python -m src.analyze_sweep --root runs/field_steps --metric test_boundary_f
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:                              # scipy 不是硬依赖，缺了就只报均值
    HAVE_SCIPY = False


def load_results(root: Path) -> list[dict]:
    """逐个实验目录读 result.json。一个目录一条，天然去重。"""
    out = []
    for fp in sorted(root.glob("*/result.json")):
        try:
            r = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  [!] {fp} 解析失败，已跳过（可能是实验被中断写到一半）")
            continue
        r["_dir"] = fp.parent.name
        out.append(r)
    return out


def cohens_d_paired(diff: np.ndarray) -> float:
    """配对设计的效应量：差值均值 / 差值标准差。

    注意这与两独立样本的 Cohen's d 不是一回事，论文里要写明用的是配对版
    （也叫 d_z）。配对 d 通常明显大于独立 d，因为分母里不含被配对消掉的方差。
    """
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="sweep 输出目录，如 runs/field_sweep")
    ap.add_argument("--metric", default="test_iou",
                    help="主判据。默认 test_iou —— 见实验日志第八节：boundary_f 量程大"
                         "但噪声按比例更大，判别力更差")
    ap.add_argument("--ref", default="random", help="作为对照的策略")
    ap.add_argument("--baseline", default=None,
                    help="全量训练的 baseline 目录，用于算'达 baseline 百分之多少'")
    ap.add_argument("--out", default=None, help="把汇总写成 json")
    args = ap.parse_args()

    root = Path(args.root)
    rows = load_results(root)
    if not rows:
        print(f"{root} 下没有 result.json")
        return

    m = args.metric
    # cell[(预算, 策略)][种子] = 指标值
    cell: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        if m not in r:
            continue
        cell[(r["budget"], r["strategy"])][r["seed"]] = r[m]

    budgets = sorted({b for b, _ in cell})
    strategies = sorted({s for _, s in cell}, key=lambda s: (s != args.ref, s))

    base_val = None
    if args.baseline:
        bp = Path(args.baseline) / "result.json"
        if bp.exists():
            base_val = json.loads(bp.read_text(encoding="utf-8")).get(m)

    print(f"\n{'=' * 74}")
    print(f"{root}    判据 = {m}    共 {len(rows)} 组")
    if base_val is not None:
        print(f"全量 baseline {m} = {base_val:.4f}")
    print("=" * 74)

    # ---- 表一：均值 ± 标准差 ----
    head = f"\n{'N':>5} " + " ".join(f"{s:>22}" for s in strategies)
    print(head)
    print("-" * len(head))
    for b in budgets:
        line = f"{b:>5} "
        for s in strategies:
            v = np.array(list(cell[(b, s)].values()))
            line += f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f} (n={len(v):>2}) " \
                if len(v) else f"{'—':>22} "
        print(line)

    # ---- 表二：按种子配对，与 ref 比 ----
    print(f"\n按种子配对检验（vs {args.ref}）")
    print(f"{'N':>5} {'策略':<10} {'Δ':>9} {'t-p':>7} {'W-p':>7} {'d_z':>7} {'同向':>7} {'达baseline':>10}")
    print("-" * 74)
    summary = []
    for b in budgets:
        ref = cell[(b, args.ref)]
        for s in strategies:
            if s == args.ref:
                continue
            cur = cell[(b, s)]
            seeds = sorted(set(ref) & set(cur))      # 只用两边都有的种子，才叫配对
            if len(seeds) < 2:
                continue
            d = np.array([cur[k] - ref[k] for k in seeds])
            tp = wp = float("nan")
            if HAVE_SCIPY:
                tp = float(stats.ttest_rel([cur[k] for k in seeds],
                                           [ref[k] for k in seeds]).pvalue)
                if np.any(d != 0):
                    wp = float(stats.wilcoxon(d).pvalue)
            pos = int((d > 0).sum())
            pct = (np.mean([cur[k] for k in seeds]) / base_val * 100) if base_val else float("nan")
            star = " *" if tp == tp and tp < 0.05 else ""
            print(f"{b:>5} {s:<10} {d.mean():>+9.4f} {tp:>7.3f} {wp:>7.3f} "
                  f"{cohens_d_paired(d):>7.2f} {pos:>4}/{len(d):<2} "
                  + (f"{pct:>9.1f}%" if pct == pct else f"{'—':>10}") + star)
            summary.append({"budget": b, "strategy": s, "n_seeds": len(d),
                            "delta": d.mean(), "t_p": tp, "wilcoxon_p": wp,
                            "cohens_dz": cohens_d_paired(d), "n_positive": pos,
                            "pct_of_baseline": pct if pct == pct else None})

    # ---- 变异系数：决定该用哪个指标当主判据 ----
    print(f"\n各格变异系数（标准差/均值），用于判断该指标的判别力")
    print(f"{'N':>5} " + " ".join(f"{s:>12}" for s in strategies))
    for b in budgets:
        line = f"{b:>5} "
        for s in strategies:
            v = np.array(list(cell[(b, s)].values()))
            line += f"{(v.std(ddof=1) / v.mean() * 100):>11.1f}% " if len(v) > 1 and v.mean() else f"{'—':>12} "
        print(line)

    sig = [x for x in summary if x["t_p"] == x["t_p"] and x["t_p"] < 0.05]
    print(f"\n显著格子（配对 t，p<0.05）：{len(sig)} / {len(summary)}")
    for x in sig:
        print(f"  N={x['budget']:<4} {x['strategy']:<10} Δ={x['delta']:+.4f}  p={x['t_p']:.3f}")
    if not sig:
        print("  （无）—— 不得写成'本方法优于随机'，只能写'未检出差异'")
    if not HAVE_SCIPY:
        print("\n[!] 未安装 scipy，p 值全为 nan。pip install scipy")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"root": str(root), "metric": m, "baseline": base_val,
                        "n_runs": len(rows), "comparisons": summary},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写出 {args.out}")


if __name__ == "__main__":
    main()
