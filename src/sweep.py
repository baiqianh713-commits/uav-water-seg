"""标注预算曲线实验的批量驱动脚本。

论文主图就是这个脚本产出的：横轴标注量 N，纵轴 IoU，
每条曲线一个选帧策略，带多种子的误差带。

用法
----
  # 先看要跑多少组、大概多久（不实际执行）
  python -m src.sweep --config configs/floodnet.yaml --dry_run

  # 正式跑
  python -m src.sweep --config configs/floodnet.yaml \
      --budgets 5 10 20 50 100 --strategies random hybrid --seeds 0 1 2

  # 只汇总已有结果、画图（不重跑实验）
  python -m src.sweep --config configs/floodnet.yaml --summarize_only

为什么必须多种子
----------------
小样本实验方差极大。只跑一个种子，你看到的"策略 A 比 B 好 3 个点"
很可能纯粹是运气。审稿人一定会问这一点，所以从一开始就按
均值±标准差来做，不要等被拒了再补。
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run_one(config: str, budget: int | None, strategy: str, seed: int,
            out_root: str, extra_opts: list[str], skip_existing: bool,
            alloc_strategy: str | None = None) -> tuple[str, bool]:
    """跑一个实验。返回 (实验目录, 是否实际执行)。"""
    tag = f"{Path(config).stem}"
    if alloc_strategy:
        tag += f"_A-{alloc_strategy}"
    tag += f"_{strategy}_n{budget or 'all'}_s{seed}"
    out_dir = Path(out_root) / tag

    # 已有完整结果就跳过。租卡中断后重跑整个 sweep 时能省下大量时间和钱。
    if skip_existing and (out_dir / "result.json").exists():
        print(f"[跳过] {tag}（已有 result.json）")
        return str(out_dir), False

    cmd = [
        sys.executable, "-m", "src.train",
        "--config", config,
        "--strategy", strategy,
        "--seed", str(seed),
        "--out", str(out_dir),
    ]
    # 只有确实存在 last.pt 时才传 --resume。
    # 之前无条件传入，配合「实验后清理权重」会出问题：
    # last.pt 被删但目录残留，训练被跳过而 best.pt 又不存在。
    if (out_dir / "last.pt").exists():
        cmd.append("--resume")
    if budget is not None:
        cmd += ["--budget", str(budget)]
    if alloc_strategy:
        cmd += ["--alloc_strategy", alloc_strategy]
    if extra_opts:
        cmd += ["--opts"] + extra_opts

    print(f"[运行] {tag}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[失败] {tag} 退出码={result.returncode}")
    return str(out_dir), True


def summarize(out_root: str, metric: str = "test_iou") -> list[dict]:
    """扫描所有 result.json，按 (策略, 预算) 聚合多种子结果。"""
    rows = []
    for path in sorted(Path(out_root).glob("*/result.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    if not rows:
        print(f"{out_root} 下没有 result.json，先跑实验。")
        return []

    # 按 (分配策略, 选帧策略, 预算) 分组。
    # 分配策略为 None 表示不分层的单池实验，与分层实验区分开统计。
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("alloc_strategy"), r.get("strategy", "?"), r.get("budget"))
        groups.setdefault(key, []).append(r)

    has_alloc = any(k[0] for k in groups)
    alloc_col = f"{'分配策略':<14}" if has_alloc else ""
    print(f"\n{alloc_col}{'选帧策略':<12}{'预算N':>7}{'种子数':>7}"
          f"{metric + ' 均值':>14}{'标准差':>9}{'最小':>8}{'最大':>8}")
    print("-" * (70 + (14 if has_alloc else 0)))

    summary = []
    for (alloc_strategy, strategy, budget), items in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0] or "", kv[0][1], kv[0][2] if kv[0][2] is not None else 10**9),
    ):
        values = [r[metric] for r in items if metric in r]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        entry = {
            "strategy": strategy,
            "alloc_strategy": alloc_strategy,
            # 画图时用的标签：有分配策略时展示两层组合
            "label": f"{alloc_strategy}+{strategy}" if alloc_strategy else strategy,
            "budget": budget,
            "n_seeds": len(arr),
            "mean": float(arr.mean()),
            # ddof=1 是样本标准差。种子数 <2 时无法估计，记为 0 并在图上标注。
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "values": arr.tolist(),
        }
        summary.append(entry)
        n_label = str(budget) if budget is not None else "全部"
        warn = "  (种子数<3，方差不可靠)" if len(arr) < 3 else ""
        alloc_cell = f"{(alloc_strategy or '-'):<14}" if has_alloc else ""
        print(f"{alloc_cell}{strategy:<12}{n_label:>7}{len(arr):>7}"
              f"{entry['mean']:>14.4f}{entry['std']:>9.4f}"
              f"{entry['min']:>8.4f}{entry['max']:>8.4f}{warn}")

    out_path = Path(out_root) / f"summary_{metric}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已写入 {out_path}")
    return summary


def plot_curve(summary: list[dict], out_path: str, metric: str) -> None:
    """画标注预算曲线。这是论文主图。"""
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境（云服务器）必须用非交互后端
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    by_strategy: dict[str, list[dict]] = {}
    upper_bound = None
    for e in summary:
        if e["budget"] is None:
            # 全量标注组画成水平虚线，作为性能上界参考
            upper_bound = e
            continue
        # label 在有分配策略时是 "分配+选帧" 的组合，否则就是选帧策略名
        by_strategy.setdefault(e.get("label") or e["strategy"], []).append(e)

    for strategy, entries in sorted(by_strategy.items()):
        entries.sort(key=lambda e: e["budget"])
        x = [e["budget"] for e in entries]
        y = np.array([e["mean"] for e in entries])
        sd = np.array([e["std"] for e in entries])
        ax.plot(x, y, marker="o", label=strategy, linewidth=1.8, markersize=5)
        ax.fill_between(x, y - sd, y + sd, alpha=0.15)  # ±1 标准差误差带

    if upper_bound is not None:
        ax.axhline(upper_bound["mean"], linestyle="--", color="gray", linewidth=1.2,
                   label=f"Full supervision ({upper_bound['mean']:.3f})")

    # 图上一律用英文标签：投稿图表本来就必须是英文，
    # 而且云服务器通常没装中文字体，用中文会显示成方框。
    ax.set_xscale("log")  # 预算通常按倍数取值，对数轴更能看清趋势
    ax.set_xlabel("Number of annotated images, N (log scale)")
    ax.set_ylabel({"test_iou": "Test IoU", "test_f1": "Test F1",
                   "test_boundary_f": "Test Boundary F-score"}.get(metric, metric))
    ax.set_title("Annotation budget vs. segmentation accuracy")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(out_path)
    print(f"图已保存到 {out_path}")


def _acquire_lock(out_root: Path) -> Path | None:
    """确保同一 out_root 下只有一个 sweep 在跑。

    多个实例并发会把日志写串，还会同时训练同一组实验、互相覆盖输出。
    锁文件里记 PID；若持有者已不存在（上次异常退出），自动接管。
    """
    import os
    lock = out_root / ".sweep.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)          # 探测进程是否存活（不发真实信号）
            print(f"[!] 已有 sweep 在运行 (PID {pid})，本次退出。")
            print(f"    若确认它已死，删除 {lock} 后重试。")
            return None
        except (ValueError, OSError):
            print(f"[i] 发现陈旧锁文件，接管")
    out_root.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--strategies", nargs="+", default=["random", "hybrid"],
                        help="条件内选帧策略")
    parser.add_argument("--alloc_strategies", nargs="+", default=None,
                        help="条件间预算分配策略。不指定则不分层（单池选择）。"
                             "可选：proportional/uniform/sqrt/diversity/difficulty/div_diff")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--include_full", action="store_true",
                        help="额外跑一组全量标注，作为性能上界")
    parser.add_argument("--out_root", default="runs/sweep")
    parser.add_argument("--metric", default="test_iou")
    parser.add_argument("--opts", nargs="*", default=[], help="传给 train.py 的配置覆盖")
    parser.add_argument("--dry_run", action="store_true", help="只列出要跑的实验，不执行")
    parser.add_argument("--summarize_only", action="store_true", help="只汇总画图，不跑实验")
    parser.add_argument("--rerun", action="store_true", help="强制重跑已完成的实验")
    args = parser.parse_args()

    # 分配策略维度：未指定时用 [None] 占位，表示不分层
    alloc_list = args.alloc_strategies or [None]
    jobs: list[tuple] = [
        (a, s, b, sd)
        for a, s, b, sd in itertools.product(
            alloc_list, args.strategies, args.budgets, args.seeds
        )
    ]
    if args.include_full:
        # 全量组与选帧/分配策略均无关，只需按种子跑
        jobs += [(None, "all", None, sd) for sd in args.seeds]

    if args.dry_run:
        print(f"共 {len(jobs)} 组实验：")
        for a, s, b, sd in jobs:
            alloc_txt = f"分配={a:<13} " if a else ""
            print(f"  {alloc_txt}选帧={s:<12} 预算={str(b or '全部'):<6} 种子={sd}")
        print("\n注意：总耗时 = 单组耗时 × 组数。租卡前先单独跑一组测准时间，"
              "再乘以组数估算费用。")
        return

    lock = None
    if not args.summarize_only:
        lock = _acquire_lock(Path(args.out_root))
        if lock is None:
            return
    try:
      if not args.summarize_only:
        for alloc_strategy, strategy, budget, seed in jobs:
            run_one(args.config, budget, strategy, seed,
                    args.out_root, args.opts, skip_existing=not args.rerun,
                    alloc_strategy=alloc_strategy)
    finally:
        if lock is not None and lock.exists():
            lock.unlink()

    summary = summarize(args.out_root, args.metric)
    if summary:
        plot_curve(summary, str(Path(args.out_root) / f"budget_curve_{args.metric}.png"), args.metric)


if __name__ == "__main__":
    main()
