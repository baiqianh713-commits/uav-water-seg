"""画论文 5.2.8 节的配图：两个数据集上的两协议预算曲线（2x2 子图）。

这张图要说明什么
----------------
上行（观察到的现象）：同一批标注预算，换一种训练协议，「标注量—精度」曲线
    的形状就变了。固定轮数 E=12 画出一条陡峭上升的曲线；固定步数 S=600 把
    同一批预算画成一条近乎平坦的线。两个数据集上都如此。

下行（造成现象的算术）：两种协议各自固定住了一个量、放跑了另一个量，
    而放跑的那个量正好相反：
      - 固定轮数：每图被看次数恒为约 12，总梯度步数**正比于 N**
                  （N=10 只有 12 步，全量有 336 步）；
      - 固定步数：总梯度步数恒为约 600，每图被看次数**反比于 N**
                  （N=10 每图被看 480 次，全量只有 21.7 次）。
    这是算术，不是经验规律 —— 所以上行的现象在两个数据集上都必然出现，
    只有**幅度**依赖数据集。

🔴 读这张图时的四条纪律（删掉任何一条，图就会被过度解读）
--------------------------------------------------------
1. 两个数据集的**纵轴绝对值不可横向比较**。水体占比、图像尺度、测试集大小
   都不同。可比的只有「同一个数据集内部，两条协议曲线之间的相对关系」。
2. 这是**同一套代码、同一个作者**跑的两次测量，排除的是数据特异性，
   **不是**实现特异性。不得写成「已获独立复现」。
3. 图中 32+组 RivAIrSet 实验与自采侧的全部实验**一律用 random 选帧**。
   这张图不包含任何策略比较，不得用来声称本文方法优于随机。
4. 自采侧两条曲线的种子数不同（固定轮数 10 个种子，固定步数 4 个种子），
   且固定轮数侧的 N=227 只跑过一次（无标准差）。图上按实际情况标注，
   不得把单次结果画成带误差带的均值。

步数从哪来
----------
🔴 本研究在「每轮几步」上已经错过三次（floor 还是 ceil、drop_last 开没开）。
所以这里**不自己推**：每轮步数一律从各次运行的 train.log 里抓程序自己打印的
那一行 ——

    固定步数模式：目标 600 步，每轮 1 步 -> 训练 600 轮（实际 600 步）

脚本再拿它跟 train.py 的实现（floor(N/batch)，但 N<=batch 时为 1）对一遍，
对不上就直接报错退出，不出图。

运行
----
    PYTHONIOENCODING=utf-8 python -m src.plot_protocol_replication

输出 runs/fig5_8_protocol_replication.png（300 dpi）与同名 .pdf（矢量）。
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器环境也能出图，必须在 import pyplot 之前设
import matplotlib.pyplot as plt

# 这张图最终要缩到页宽（GRSL 还是双栏）。画布小、字号大，缩完才读得清。
plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})

# ---------------------------------------------------------------- 常量

OUT_DIR = Path("runs")
BATCH = 8            # 两个数据集的 train.batch_size 都是 8，脚本会逐个核对
EPOCHS_FIXED = 12    # 固定轮数协议的 E

# 两条协议曲线的配色与样式，两个子图共用，便于横向看
STY_EPOCH = dict(color="crimson",   marker="s", ls="-",  lw=1.8, ms=6)
STY_STEPS = dict(color="#1f77b4",   marker="o", ls="-",  lw=1.8, ms=6)


# ---------------------------------------------------------------- 读数据

def read_iou(path: Path, *, expect_n: int, expect_seed: int) -> float:
    """读一个 result.json 的 test_iou，并核对它确实是我们以为的那一组。

    防的是「目录被挪动过 / 改名后内容没跟着改」这类错误 —— 这类错误不会
    报异常，只会悄悄把不可比的数字平均到一起。
    """
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("seed") != expect_seed:
        raise SystemExit(f"{path}: 内容里 seed={r.get('seed')}，按目录名应为 {expect_seed}")
    if r.get("n_train_used") != expect_n:
        raise SystemExit(f"{path}: 实训 {r.get('n_train_used')} 张，按目录名应为 {expect_n} 张")
    # 🔴 两臂都必须 select_by_val=False，否则「协议」之外又多了一个变量，
    #    精度差异无法单独归因到训练量上。
    if r.get("select_by_val") is not False:
        raise SystemExit(f"{path}: select_by_val={r.get('select_by_val')}，必须为 False")
    return r["test_iou"]


def read_batch_size(run_dir: Path) -> int:
    """从该次运行自己存下的 config.yaml 里读 batch_size（不信任全局常量）。"""
    txt = (run_dir / "config.yaml").read_text(encoding="utf-8")
    m = re.search(r"^  batch_size: (\d+)$", txt, re.M)
    if not m:
        raise SystemExit(f"{run_dir}/config.yaml 里找不到 train.batch_size")
    return int(m.group(1))


def read_step_line(run_dir: Path) -> tuple[int, int, int, int]:
    """抓固定步数运行的 train.log 里那行程序输出，返回四个数。

    返回 (目标步数, 每轮步数, 训练轮数, 实际步数)。
    这是本脚本里**唯一**的步数来源 —— 不手工推算。
    """
    txt = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"目标 (\d+) 步，每轮 (\d+) 步 -> 训练 (\d+) 轮（实际 (\d+) 步）", txt)
    if not m:
        raise SystemExit(f"{run_dir}/train.log 里找不到『固定步数模式』那一行")
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def expected_spe(n: int, batch: int) -> int:
    """按 train.py 的实现算每轮步数，只用来跟日志**对账**，不用来出图。

    train.py 给 DataLoader 设的是 drop_last=(len(train_ds) > batch)：
    训练集比一个 batch 大时丢掉最后不满一批的零头，所以是 floor 不是 ceil；
    训练集不比一个 batch 大时不丢，整批算一步。
    """
    return 1 if n <= batch else n // batch


# ---------------------------------------------------------------- 两个数据集的配置

def field_runs() -> dict:
    """自采数据（Basento 以外，国内单一场地，训练集全量 227 张）。

    固定轮数臂：runs/field_sweep/field_random_n{N}_s{0..9}，10 个种子；
                N=227 没有扫描，只有 runs/field_baseline 一次（无 sd）。
    固定步数臂：seed 0 在 runs/conv_grid/n{N}_s600（注意这里的 _s600 是**步数**
                不是种子号），seed 1~3 在 runs/conv_seeds/n{N}_s600_seed{S}。

    ⚠️ 两批的目录命名规则不同，所以种子号一律显式写出，不从目录名猜。
    """
    budgets = [10, 20, 50, 227]
    epoch_runs: dict[int, list[tuple[Path, int]]] = {}
    for n in budgets:
        if n == 227:
            epoch_runs[n] = [(Path("runs/field_baseline"), 0)]
        else:
            epoch_runs[n] = [(Path(f"runs/field_sweep/field_random_n{n}_s{s}"), s)
                             for s in range(10)]
    steps_runs = {
        n: [(Path(f"runs/conv_grid/n{n}_s600"), 0)]
           + [(Path(f"runs/conv_seeds/n{n}_s600_seed{s}"), s) for s in (1, 2, 3)]
        for n in budgets
    }
    return dict(key="field", title="(a) Field data (own UAV survey, 227 train frames)",
                sub_title="(c) Field data: what each protocol holds fixed,\n"
                          "and what it lets run with $N$",
                budgets=budgets, full=227,
                epoch_runs=epoch_runs, steps_runs=steps_runs)


def rivair_runs() -> dict:
    """RivAIrSet（意大利 Basento 河，公开数据集，训练集全量 138 张）。

    两臂都在 runs/rivair_proto 下，各 4 个种子，命名规则统一。
    """
    budgets = [10, 20, 50, 138]
    return dict(key="rivair", title="(b) RivAIrSet (public dataset, 138 train frames)",
                sub_title="(d) RivAIrSet: the same arithmetic, on a dataset\n"
                          "whose full training set is 138 rather than 227",
                budgets=budgets, full=138,
                epoch_runs={n: [(Path(f"runs/rivair_proto/epoch_n{n}_seed{s}"), s)
                                for s in range(4)] for n in budgets},
                steps_runs={n: [(Path(f"runs/rivair_proto/steps_n{n}_seed{s}"), s)
                                for s in range(4)] for n in budgets})


def collect(ds: dict) -> dict:
    """把一个数据集的两臂结果读齐，顺带把步数对账做掉。"""
    out = {"epoch": {}, "steps": {}, "spe": {}, "steps_actual": {}, "batch": None}

    for arm in ("epoch", "steps"):
        for n, dirs in ds[f"{arm}_runs"].items():
            vals = []
            for d, seed in dirs:
                fp = d / "result.json"
                if not fp.exists():
                    raise SystemExit(f"缺 {fp} —— 这一组没跑完，先补齐再出图")
                vals.append(read_iou(fp, expect_n=n, expect_seed=seed))
            out[arm][n] = vals

    # ---- batch_size 核对：两臂所有运行必须一致，否则「每图被看次数」没法并排
    batches = {read_batch_size(d) for dirs in ds["steps_runs"].values() for d, _ in dirs}
    if len(batches) != 1:
        raise SystemExit(f"{ds['key']}: 各次运行的 batch_size 不一致 {batches}")
    out["batch"] = batches.pop()
    if out["batch"] != BATCH:
        raise SystemExit(f"{ds['key']}: batch_size={out['batch']}，脚本常量写的是 {BATCH}")

    # ---- 步数：从日志抓，再跟 train.py 的实现对账
    for n, dirs in ds["steps_runs"].items():
        target, spe, n_ep, actual = read_step_line(dirs[0][0])
        exp = expected_spe(n, out["batch"])
        if spe != exp:
            raise SystemExit(
                f"{ds['key']} N={n}: 日志说每轮 {spe} 步，按 train.py 的实现应为 {exp} 步。"
                "两者不符说明实现或日志有一处改过，必须查清再出图。")
        out["spe"][n] = spe
        out["steps_actual"][n] = actual
    return out


# ---------------------------------------------------------------- 统计小工具

def mean_sd(vals: list[float]) -> tuple[float, float | None]:
    """均值与样本标准差；只有一个值时标准差返回 None（不画误差带）。"""
    return (vals[0], None) if len(vals) < 2 else (st.mean(vals), st.stdev(vals))


def spread(stats: dict[int, tuple[float, float | None]], budgets: list[int]) -> float:
    """预算间极差 = 各预算均值的 最大 - 最小。这是本图的核心量。"""
    ms = [stats[n][0] for n in budgets]
    return max(ms) - min(ms)


# ---------------------------------------------------------------- 画图

def panel_curve(ax, ds: dict, data: dict) -> tuple[float, float]:
    """上行：一个数据集的两条协议曲线，返回两臂的极差（供标注与打印）。"""
    budgets = ds["budgets"]
    stats = {arm: {n: mean_sd(data[arm][n]) for n in budgets} for arm in ("epoch", "steps")}

    for arm, sty in (("epoch", STY_EPOCH), ("steps", STY_STEPS)):
        ys = [stats[arm][n][0] for n in budgets]
        ax.plot(budgets, ys, zorder=3, **sty)

        # 误差带只画在**有多个种子**的那几个预算上。
        # 单次运行没有标准差，若按 sd=0 一起画进去，带子会在那里收成一条线，
        # 看上去像「该预算处方差极小」—— 那是视觉上的失实陈述。
        multi = [n for n in budgets if stats[arm][n][1] is not None]
        if multi:
            # 带子只能画在连续的一段上；若多种子的预算不连续，宁可报错也不乱连
            i0, i1 = budgets.index(multi[0]), budgets.index(multi[-1])
            if budgets[i0:i1 + 1] != multi:
                raise SystemExit(f"{arm} 臂的多种子预算不连续：{multi}，误差带画法需另行决定")
            ax.fill_between(multi,
                            [stats[arm][n][0] - stats[arm][n][1] for n in multi],
                            [stats[arm][n][0] + stats[arm][n][1] for n in multi],
                            color=sty["color"], alpha=0.16, lw=0, zorder=2)
        # 单次运行的点：改成空心标记并明写 single run，和多种子点区别开
        for n in budgets:
            if stats[arm][n][1] is None:
                ax.plot([n], [stats[arm][n][0]], marker=sty["marker"], ms=sty["ms"] + 3,
                        mfc="white", mec=sty["color"], mew=1.8, ls="none", zorder=4)
                ax.annotate("single run\n(no sd)", (n, stats[arm][n][0]),
                            textcoords="offset points", xytext=(-9, -4),
                            fontsize=8, color=sty["color"], ha="right", va="top")

    # 每臂每点由几个种子平均而来 —— 两臂种子数不同（自采侧 10 vs 4），
    # 不写出来读者会默认两条带子的可信度相同。放进下方的数字框，不塞图例。
    def seed_note(arm: str) -> str:
        counts = sorted({len(data[arm][n]) for n in budgets})
        return "/".join(str(c) for c in counts)

    r_epoch, r_steps = spread(stats["epoch"], budgets), spread(stats["steps"], budgets)
    # 把两臂的极差直接写在图上 —— 这是全图唯一要读者记住的数
    ax.annotate(
        f"spread over budgets  seeds/pt\n"
        f"  E={EPOCHS_FIXED:<5}{r_epoch*100:5.1f} pt    {seed_note('epoch'):>4}\n"
        f"  S=600  {r_steps*100:5.2f} pt    {seed_note('steps'):>4}\n"
        f"  contraction {r_epoch/r_steps:4.1f}x",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=8.2, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", lw=0.7, alpha=0.93))

    # 纵轴范围：把误差带整条包进来，上限不超过 1.0（IoU 不可能大于 1，
    # 轴画到 1.04 会让人以为还有上升空间）。
    # ⚠️ 两个子图的纵轴**各自独立** —— 两个数据集的绝对 IoU 本来就不可横向比较，
    #    强行统一纵轴反而会诱导读者去比不可比的东西。
    band = [stats[a][n][0] + sign * (stats[a][n][1] or 0)
            for a in ("epoch", "steps") for n in budgets for sign in (-1, 1)]
    lo, hi = min(band), max(band)
    pad = (hi - lo) * 0.10
    ax.set_ylim(lo - pad, min(1.0, hi + pad))

    # 两条曲线直接贴标签，不用图例 —— 只有两条线时直接标注比图例好读，
    # 也省下了为图例预留的那块空白（留空白就得把纵轴顶到 1.0 以上）。
    # 锚点要躲开曲线本身：固定轮数那条是陡升的，标在最低预算处、向右下排，
    # 曲线就从文字上方跑掉了；固定步数那条近乎水平，标在第二个预算正上方即可。
    for arm, sty, text, idx, off, ha in (
        ("epoch", STY_EPOCH, f"Fixed-epoch (E={EPOCHS_FIXED})", 0, (10, -14), "left"),
        ("steps", STY_STEPS, "Fixed-step (S=600)", 1, (6, 12), "center"),
    ):
        anchor = budgets[idx]
        ax.annotate(text, (anchor, stats[arm][anchor][0]),
                    textcoords="offset points", xytext=off, ha=ha, va="top" if off[1] < 0 else "bottom",
                    fontsize=9.5, color=sty["color"], fontweight="medium")

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{n}" + ("\n(full)" if n == ds["full"] else "") for n in budgets])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Annotation budget $N$ (log scale)")
    ax.set_ylabel("Test IoU")
    ax.grid(alpha=0.25, ls="-", lw=0.5)
    ax.set_title(ds["title"], fontsize=10.5)
    return r_epoch, r_steps


def panel_budget(ax, ds: dict, data: dict, *, show_ylabel: bool) -> None:
    """下行：同一组预算下，两种协议各自固定了什么、放跑了什么。

    四条线都是**计数**（步数、次数），所以可以共用一根对数纵轴：
      实线 = 总梯度步数        虚线 = 每张图平均被看的次数
    """
    budgets = ds["budgets"]
    batch = data["batch"]

    # 固定轮数臂：实际步数 = 每轮步数 x E。每轮步数取自日志（见 collect 的对账）。
    ep_steps = [data["spe"][n] * EPOCHS_FIXED for n in budgets]
    # 每图平均被看次数 = 总样本取用数 / 标注张数 = 步数 x batch / N。
    # drop_last 丢掉的零头已经体现在「每轮步数」里，所以这个式子是精确的。
    ep_seen = [s * batch / n for s, n in zip(ep_steps, budgets)]

    # 固定步数臂：实际步数取自日志（目标 600，但 ceil 到整轮后常略多于 600）
    st_steps = [data["steps_actual"][n] for n in budgets]
    st_seen = [s * batch / n for s, n in zip(st_steps, budgets)]

    ax.plot(budgets, ep_steps, **{**STY_EPOCH, "ls": "-"},
            label="Fixed-epoch: total steps  ($\\propto N$)", zorder=3)
    ax.plot(budgets, ep_seen, **{**STY_EPOCH, "ls": "--", "marker": "s", "ms": 4},
            label="Fixed-epoch: times each image seen (const.)", zorder=3)
    ax.plot(budgets, st_steps, **{**STY_STEPS, "ls": "-"},
            label="Fixed-step: total steps (const.)", zorder=3)
    ax.plot(budgets, st_seen, **{**STY_STEPS, "ls": "--", "marker": "o", "ms": 4},
            label="Fixed-step: times each image seen  ($\\propto 1/N$)", zorder=3)

    # 两端各标一个数，读者不用去查纵轴刻度
    for xs, ys, color, va in ((budgets, ep_steps, STY_EPOCH["color"], "bottom"),
                              (budgets, st_seen, STY_STEPS["color"], "top")):
        for i in (0, len(xs) - 1):
            ax.annotate(f"{ys[i]:.0f}" if ys[i] >= 10 else f"{ys[i]:.1f}",
                        (xs[i], ys[i]), textcoords="offset points",
                        xytext=(0, 7 if va == "bottom" else -13),
                        fontsize=8.6, color=color, ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{n}" + ("\n(full)" if n == ds["full"] else "") for n in budgets])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Annotation budget $N$ (log scale)")
    if show_ylabel:
        ax.set_ylabel("Count (log scale)")
    ax.grid(alpha=0.25, ls="-", lw=0.5)
    ax.set_title(ds["sub_title"], fontsize=10.5)
    # 纵轴顶部留出空白，把图例放进那片空白里 —— 否则图例会正好压住两条线的
    # 交叉点，而那个交叉点恰恰是「两种协议让相反的量随 N 跑起来」最直观的地方。
    # 图例排成**一列**：横排两列时整个框比子图还宽，会甩出坐标区两侧。
    ax.set_ylim(6.5, 6000)
    ax.legend(loc="upper left", fontsize=8.0, ncol=1, framealpha=0.95,
              handlelength=2.0, labelspacing=0.32, borderpad=0.45)


def main() -> None:
    datasets = [field_runs(), rivair_runs()]
    loaded = [collect(ds) for ds in datasets]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8))

    spreads = []
    for col, (ds, data) in enumerate(zip(datasets, loaded)):
        spreads.append(panel_curve(axes[0][col], ds, data))
        panel_budget(axes[1][col], ds, data, show_ylabel=(col == 0))

    fig.tight_layout()
    # png 看稿用，pdf 投稿用，svg 给 Typst 排版用（Typst 对 svg 的支持最稳）
    for ext in ("png", "pdf", "svg"):
        fp = OUT_DIR / f"fig5_8_protocol_replication.{ext}"
        fig.savefig(fp, dpi=300)
        print(f"已写出 {fp}")

    # ---- 把图上的数原样打印一遍，方便跟正文表格逐个核对 ----
    print("\n核心数字（请与 paper-ch5.md 5.2.8 及两稿正文逐个核对）")
    print("=" * 66)
    for ds, (re_, rs_) in zip(datasets, spreads):
        print(f"  {ds['key']:<8} 固定轮数极差 {re_*100:5.1f} 点   "
              f"固定步数极差 {rs_*100:5.2f} 点   收缩 {re_/rs_:4.1f}x")
    print("\n  🔴 两个数据集的收缩倍数接近，说明该现象不依赖单一采集地点；")
    print("     但这是同一套代码、同一个作者的两次测量，**不是**第三方独立复现。")
    print("  🔴 全部实验一律 random 选帧，本图不含任何策略比较。")


if __name__ == "__main__":
    main()
