"""数据集冗余度诊断 —— 跑任何训练实验之前**必须先跑这个**。

为什么需要它
------------
本研究的核心方法是"冗余感知选帧"：相邻帧高度相似，标了一帧就不该再标它的近邻。
这个方法只在**高冗余**数据上有价值。

上次在 LoveDA 上跑了 8 组实验才发现该数据集冗余度太低（跨城市独立影像），
策略间差异被种子方差完全淹没（见 docs/experiment-log.md）。
那是数小时的算力和时间浪费。这个脚本用几分钟给出同样的判断。

诊断什么
--------
1. **最近邻距离分布**：每个样本到其最近邻的距离。若大量样本的最近邻距离
   极小，说明存在近重复帧 —— 这是冗余的直接证据。
2. **有效样本数比例**：用 k-center greedy 覆盖整个数据集所需的样本数。
   若 50 个样本就能覆盖 1000 张图的特征空间，冗余度就很高。
3. **随机 vs 覆盖度选择的差异**：直接模拟两种策略选出的子集，
   比较它们的特征空间覆盖质量。这是对"方法是否有发挥空间"的最直接预判。
4. **难度先验的有效性**：各维度的分布与相关性（LoveDA 上反光维度失效过）。

判据
----
  高冗余 → 方法有发挥空间，可以继续跑实验
  低冗余 → 与 LoveDA 同样的问题，不要浪费算力，换数据

用法
----
  python -m src.diagnose_redundancy --root data/processed/floodnet --split train
  python -m src.diagnose_redundancy --root data/processed/loveda_small --split train  # 对照
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import list_samples
from .selection import difficulty_score, extract_features, kcenter_greedy


def nn_distance_stats(embed: np.ndarray) -> dict:
    """最近邻距离分布。这是冗余度最直接的度量。

    嵌入已 L2 归一化，所以距离范围大致在 [0, 2]。
    近重复帧的距离会非常接近 0。
    """
    n = embed.shape[0]
    # 分块计算距离矩阵，避免 n 很大时内存爆炸
    nn_dists = np.empty(n, dtype=np.float64)
    block = 512
    for i in range(0, n, block):
        chunk = embed[i : i + block]
        d = np.linalg.norm(chunk[:, None, :] - embed[None, :, :], axis=-1)
        # 把自己到自己的 0 距离排除
        np.fill_diagonal(d[:, i : i + chunk.shape[0]], np.inf)
        nn_dists[i : i + chunk.shape[0]] = d.min(axis=1)

    # 用分位数刻画分布。中位数很小 = 普遍存在近邻 = 高冗余
    qs = [1, 5, 10, 25, 50, 75, 90]
    return {
        "mean": float(nn_dists.mean()),
        "median": float(np.median(nn_dists)),
        "percentiles": {f"p{q}": float(np.percentile(nn_dists, q)) for q in qs},
        "raw": nn_dists,
    }


def coverage_curve(embed: np.ndarray, fractions: list[float]) -> dict:
    """用 k-center greedy 选出 k 个样本后，剩余样本到已选集合的最大距离。

    这个值下降得越快，说明少量样本就能代表整个数据集，冗余度越高。
    """
    n = embed.shape[0]
    result = {}
    for frac in fractions:
        k = max(1, int(n * frac))
        idx = kcenter_greedy(embed, k, seed=0)
        sel = embed[idx]
        # 每个样本到已选集合的最小距离，取最大值（即最差覆盖情况）
        d = np.linalg.norm(embed[:, None, :] - sel[None, :, :], axis=-1).min(axis=1)
        result[f"{frac:.0%}"] = {
            "k": k,
            "max_dist": float(d.max()),
            "mean_dist": float(d.mean()),
        }
    return result


def strategy_gap(embed: np.ndarray, budgets: list[int], n_seeds: int = 5) -> dict:
    """模拟随机选择 vs 覆盖度选择，比较子集的覆盖质量。

    这是对"方法是否有发挥空间"最直接的预判，且**不需要训练任何模型**。
    若两者覆盖质量接近，训练结果也不会有差异 —— 这正是 LoveDA 上发生的事。

    ⚠️ 用哪个指标评判很关键
    k-center greedy 优化的是 **最差情况**（minimax：最小化最大覆盖距离），
    不是平均情况。若用"平均覆盖距离"评判它，会得到 k-center 不如随机的
    错误结论 —— 随机采样倾向于落在样本密集区，平均距离自然更小，
    但会完全漏掉稀疏区域的样本。
    因此这里以 **max_dist（最差覆盖距离）** 为主判据，同时报告平均值供参考。
    """
    n = embed.shape[0]
    out = {}
    for budget in budgets:
        if budget >= n:
            continue

        def cover(idx: np.ndarray) -> tuple[float, float]:
            """返回 (最差覆盖距离, 平均覆盖距离)。"""
            d = np.linalg.norm(embed[:, None, :] - embed[idx][None, :, :], axis=-1).min(axis=1)
            return float(d.max()), float(d.mean())

        rand_max, rand_mean = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(s)
            mx, mn = cover(rng.choice(n, size=budget, replace=False))
            rand_max.append(mx)
            rand_mean.append(mn)

        kc_max, kc_mean = [], []
        for s in range(n_seeds):
            mx, mn = cover(np.array(kcenter_greedy(embed, budget, seed=s)))
            kc_max.append(mx)
            kc_mean.append(mn)

        rm, km = np.array(rand_max), np.array(kc_max)
        # 覆盖距离越小越好，improvement 为正表示 kcenter 更优
        improvement = (rm.mean() - km.mean()) / (rm.mean() + 1e-12)
        # 效应量：改进幅度相对于随机方差的倍数。
        # 关键指标 —— 若改进幅度小于方差，训练实验就分辨不出来。
        pooled_std = np.sqrt((rm.var(ddof=1) + km.var(ddof=1)) / 2) if n_seeds > 1 else 0.0
        effect = (rm.mean() - km.mean()) / pooled_std if pooled_std > 1e-12 else float("inf")

        out[str(budget)] = {
            # 主判据：最差覆盖距离
            "random_max_cover_dist": float(rm.mean()),
            "random_max_std": float(rm.std(ddof=1)) if n_seeds > 1 else 0.0,
            "kcenter_max_cover_dist": float(km.mean()),
            "kcenter_max_std": float(km.std(ddof=1)) if n_seeds > 1 else 0.0,
            "relative_improvement": float(improvement),
            "effect_size": float(effect) if np.isfinite(effect) else None,
            # 参考：平均覆盖距离（k-center 并不优化这个指标）
            "random_mean_cover_dist": float(np.mean(rand_mean)),
            "kcenter_mean_cover_dist": float(np.mean(kc_mean)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=1500,
                    help="样本过多时先随机抽样，控制 O(n^2) 距离计算的开销")
    ap.add_argument("--out", default=None, help="诊断结果 JSON 输出路径")
    args = ap.parse_args()

    stems = list_samples(args.root, args.split)
    if len(stems) > args.max_samples:
        rng = np.random.default_rng(0)
        stems = sorted(rng.choice(stems, size=args.max_samples, replace=False).tolist())
        print(f"样本过多，随机抽取 {len(stems)} 个用于诊断")

    print(f"数据集 {args.root} / {args.split}：{len(stems)} 个样本")
    print("正在提取特征...")
    embed, diffs, stems = extract_features(args.root, args.split, stems)
    print(f"特征维度 {embed.shape}\n")

    # ---- 1. 最近邻距离 ----
    print("=" * 66)
    print("1. 最近邻距离分布（嵌入已 L2 归一化，距离范围约 [0, 2]）")
    print("=" * 66)
    nn = nn_distance_stats(embed)
    print(f"  均值 {nn['mean']:.4f}   中位数 {nn['median']:.4f}")
    for q, v in nn["percentiles"].items():
        print(f"    {q:>4}: {v:.4f}")

    raw = nn["raw"]
    # 近重复的判定阈值。0.05 是经验值，对应视觉上几乎无差别的帧。
    for thr in [0.02, 0.05, 0.10]:
        frac = float((raw < thr).mean())
        print(f"  最近邻距离 < {thr:.2f} 的样本占比: {frac:.1%}")

    # ---- 2. 覆盖度曲线 ----
    print("\n" + "=" * 66)
    print("2. k-center 覆盖度（用 k 个样本代表全体时的最差/平均覆盖距离）")
    print("=" * 66)
    cov = coverage_curve(embed, [0.01, 0.02, 0.05, 0.10, 0.20])
    print(f"  {'比例':>6} {'k':>6} {'最差覆盖距离':>14} {'平均覆盖距离':>14}")
    for frac, v in cov.items():
        print(f"  {frac:>6} {v['k']:>6} {v['max_dist']:>14.4f} {v['mean_dist']:>14.4f}")

    # ---- 3. 策略差异预判（最关键）----
    print("\n" + "=" * 66)
    print("3. 随机 vs 覆盖度选择的差异预判（无需训练模型）")
    print("=" * 66)
    gap = strategy_gap(embed, args.budgets, args.n_seeds)
    print("  主判据 = 最差覆盖距离（k-center 优化的正是这个 minimax 目标）")
    print(f"  {'预算':>6} {'随机(最差)':>12} {'kc(最差)':>11} {'相对改进':>10} {'效应量':>9}"
          f" | {'随机(平均)':>12} {'kc(平均)':>11}")
    for budget, v in gap.items():
        eff = v["effect_size"]
        eff_s = f"{eff:>9.2f}" if eff is not None else f"{'inf':>9}"
        print(f"  {budget:>6} {v['random_max_cover_dist']:>12.4f} "
              f"{v['kcenter_max_cover_dist']:>11.4f} "
              f"{v['relative_improvement']:>9.1%} {eff_s}"
              f" | {v['random_mean_cover_dist']:>12.4f} {v['kcenter_mean_cover_dist']:>11.4f}")
    print("  注：覆盖距离越小越好；相对改进为正表示 kcenter 优于随机。")
    print("      平均列仅供参考 —— k-center 不优化平均距离，随机采样在该指标上")
    print("      天然占优（倾向落在密集区），用它评判会得出错误结论。")

    # ---- 4. 难度先验 ----
    print("\n" + "=" * 66)
    print("4. 难度先验各维度分布")
    print("=" * 66)
    names = ["反光(glare)", "阴影(shadow)", "纹理平坦(flatness)", "蓝绿(bluegreen)"]
    dead_dims = []
    for j, nm in enumerate(names):
        c = diffs[:, j]
        print(f"  {nm:<22} 均值={c.mean():.4f} 标准差={c.std():.4f} "
              f"范围=[{c.min():.4f}, {c.max():.4f}]")
        # 均值极小说明该维度几乎不触发（LoveDA 上反光就是这样）
        if c.mean() < 0.01 or c.std() < 0.005:
            dead_dims.append(nm)

    # ---- 总体判断 ----
    print("\n" + "=" * 66)
    print("诊断结论")
    print("=" * 66)

    near_dup = float((raw < 0.05).mean())
    nn_median = nn["median"]
    max_effect = max(
        (v["effect_size"] for v in gap.values() if v["effect_size"] is not None),
        default=0.0,
    )
    verdict_lines = []

    # 冗余度判据。参考实测值：
    #   LoveDA（跨城市独立卫星影像，已验证低冗余）  中位数 0.246，<0.05 占比 1.5%
    #   同条件重复采集的数据                        中位数 0.068，<0.05 占比 30%
    # 最近邻距离中位数比"近重复占比"更稳健，因为后者对阈值选择敏感。
    if nn_median < 0.10 or near_dup > 0.20:
        verdict_lines.append(
            f"[+] 最近邻距离中位数 {nn_median:.4f}，近重复帧占比 {near_dup:.1%} "
            "—— 冗余度高，冗余感知有发挥空间"
        )
    elif nn_median < 0.18 or near_dup > 0.05:
        verdict_lines.append(
            f"[~] 最近邻距离中位数 {nn_median:.4f}，近重复帧占比 {near_dup:.1%} "
            "—— 冗余度中等，建议加大种子数并扩大候选池"
        )
    else:
        verdict_lines.append(
            f"[-] 最近邻距离中位数 {nn_median:.4f}，近重复帧占比仅 {near_dup:.1%} "
            "—— 冗余度低（LoveDA 为 0.246 / 1.5%，属此类）"
        )

    # 效应量判据。注意这衡量的是**特征空间覆盖度**的可分辨性，
    # 不直接等于训练精度的可分辨性 —— 覆盖度改进能否转化为精度提升，
    # 仍需实际训练验证。所以效应量大只是必要条件，不是充分条件。
    if max_effect >= 3.0:
        verdict_lines.append(
            f"[+] 覆盖度效应量 {max_effect:.2f} —— 远大于随机方差，选择策略确实产生了不同的子集"
        )
    elif max_effect >= 1.0:
        verdict_lines.append(f"[~] 覆盖度效应量 {max_effect:.2f} —— 勉强可分辨，建议种子数 ≥5")
    else:
        verdict_lines.append(
            f"[-] 覆盖度效应量仅 {max_effect:.2f} —— 小于随机方差，"
            "策略选出的子集与随机无实质差别，**不建议在此数据上跑完整实验**"
        )

    if dead_dims:
        verdict_lines.append(f"[!] 以下难度维度几乎不触发，在此数据上无效: {', '.join(dead_dims)}")

    for line in verdict_lines:
        print("  " + line)

    if nn_median >= 0.18 and near_dup <= 0.05:
        print("\n  ** 冗余度低：该数据集难以体现冗余感知的价值。")
        print("     优先考虑连续航拍视频帧（自采数据），而非多地点独立影像。**")

    print("\n  注意：本诊断只衡量**特征空间**的可分辨性。覆盖度改进能否转化为")
    print("        分割精度提升，仍须实际训练验证 —— 效应量大是必要条件而非充分条件。")

    # ---- 保存 ----
    out_path = Path(args.out) if args.out else Path(args.root) / f"redundancy_{args.split}.json"
    payload = {
        "root": str(args.root), "split": args.split, "n_samples": len(stems),
        "nn_distance": {k: v for k, v in nn.items() if k != "raw"},
        "near_duplicate_fraction": {
            f"<{t}": float((raw < t).mean()) for t in [0.02, 0.05, 0.10]
        },
        "coverage": cov,
        "strategy_gap": gap,
        "difficulty_dims": {
            names[j]: {"mean": float(diffs[:, j].mean()), "std": float(diffs[:, j].std())}
            for j in range(diffs.shape[1])
        },
        "dead_dimensions": dead_dims,
        "verdict": verdict_lines,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n诊断结果已写入 {out_path}")


if __name__ == "__main__":
    main()
