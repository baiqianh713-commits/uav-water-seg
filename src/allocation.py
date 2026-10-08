"""多条件标注预算分配 —— 贡献点 2。

问题设定
--------
无人机影像不是同质的一堆图，而是由若干**采集条件**构成的分层数据。
以 RivAIr 数据集（La Salandra et al., 2025, Table 1）为例：

    2019-06  50m  中高流量  晴   2,025 张
    2020-07  50m  低流量    晴   1,981 张
    2021-10  50m  中高流量  阴   1,574 张
    2022-03  75m  中高流量  晴     697 张
    2023-05  75m  高流量    阴   1,046 张
    2024-06  75m  中低流量  晴     307 张

六个条件（流态 × 天气 × 高度），张数从 307 到 2,025 相差近 7 倍。

给定总标注预算 N，应当如何在这些条件之间分配？这个问题此前无人研究，
而它直接回应 RivAIr 论文 §4.4 明确写出的未来方向 ——
"multiple condition-specific YOLOv8 models, each finely trained on datasets
representative of distinct flow regimes and weather conditions"：
每种条件各训一个模型，就意味着每种条件各需一套标注。

两层结构
--------
本模块只负责**条件间**的预算分配（分多少给每个条件）；
条件**内部**选哪几帧仍由 selection.py 负责。二者组合成完整策略：

    总预算 N  --[allocation]-->  各条件配额 {c: n_c}  --[selection]-->  具体帧

分配策略
--------
  proportional : 按各条件的候选池大小成比例。工程上最朴素的做法。
  uniform      : 每个条件等额。最常见的"公平"直觉。
  sqrt         : 按池大小的平方根分配。在成比例与等额之间折中，
                 是分层抽样中的经典做法（Neyman 分配的简化形式）。
  diversity    : 按条件内部的**外观多样性**分配 —— 越同质的条件给越少。
                 这是本文提出的方法，直觉是：一个条件内部如果高度冗余，
                 少量样本就足以代表它。
  difficulty   : 按条件的平均水体难度分配。
  div_diff     : 多样性与难度的几何加权组合（本文方法的完整形式）。

⚠️ 学术诚实提醒
uniform / proportional / sqrt 都是分层抽样的标准做法，不能声称原创。
可辩护的创新在于：① 把"条件间预算分配"这个问题**提出来**并系统化研究；
② 用无监督的条件内多样性/水体难度来驱动分配。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .selection import extract_features, difficulty_score, select_frames


# ---------------------------------------------------------------------------
# 条件分组
# ---------------------------------------------------------------------------
def group_by_condition(
    stems: list[str],
    pattern: str | None = None,
    mapping: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """把样本按采集条件分组。

    两种指定方式：
      pattern : 正则，第 1 组作为条件 ID。适合文件名里编码了条件的情况，
                例如 `2019-06_alt50_medhigh_sunny_0001.png` 用
                `^(\\d{4}-\\d{2})` 可按架次分组。
      mapping : 显式的 {样本名: 条件ID} 字典，从 JSON 读入。
                适合条件信息在外部元数据里的情况（RivAIr 就是这样）。

    两者都不给时，全部样本归为单一条件 "all"，
    此时分配退化为平凡情形，便于与单条件实验对比。
    """
    groups: dict[str, list[str]] = defaultdict(list)

    if mapping is not None:
        for s in stems:
            groups[mapping.get(s, "unknown")].append(s)
    elif pattern is not None:
        rx = re.compile(pattern)
        for s in stems:
            m = rx.match(s)
            groups[m.group(1) if m else "unknown"].append(s)
    else:
        groups["all"] = list(stems)

    return {k: sorted(v) for k, v in sorted(groups.items())}


def load_condition_map(path: str | Path) -> dict[str, str]:
    """从 JSON 读取 {样本名: 条件ID} 映射。

    支持两种格式：
      {"stem1": "cond_a", "stem2": "cond_b", ...}
      {"cond_a": ["stem1", "stem2"], "cond_b": [...]}   （自动反转）
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {}
    # 值是列表 -> 条件到样本的映射，需要反转
    if isinstance(next(iter(data.values())), list):
        return {stem: cond for cond, stems in data.items() for stem in stems}
    return data


# ---------------------------------------------------------------------------
# 条件级统计量（全部无监督，不使用任何标签）
# ---------------------------------------------------------------------------
def condition_stats(
    root: str | Path,
    split: str,
    groups: dict[str, list[str]],
    difficulty_weights: dict | None = None,
) -> dict[str, dict]:
    """计算每个条件的规模、内部多样性与平均难度。

    多样性定义为条件内部样本嵌入的**平均成对距离**。
    值越小说明该条件内部越同质（冗余越高），少量样本即可代表它。

    为什么用平均成对距离而不是方差：嵌入已 L2 归一化，
    平均成对距离与该条件在特征空间占据的"体积"更直接相关，
    且量纲与 selection.py 里 k-center 使用的距离一致，便于解释。

    ⚠️ 所有统计量只用图像本身，不用掩膜 —— 冷启动时没有任何标注可用。
    """
    stats: dict[str, dict] = {}

    for cond, cond_stems in groups.items():
        if not cond_stems:
            continue
        embed, diffs, kept = extract_features(root, split, cond_stems)
        score = difficulty_score(diffs, difficulty_weights)

        # 平均成对距离。样本多时抽样估计，避免 O(n^2) 爆炸
        n = embed.shape[0]
        if n < 2:
            diversity = 0.0
        else:
            if n > 300:
                rng = np.random.default_rng(0)
                sub = embed[rng.choice(n, 300, replace=False)]
            else:
                sub = embed
            # 成对距离矩阵的上三角均值
            d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            iu = np.triu_indices(d.shape[0], k=1)
            diversity = float(d[iu].mean())

        stats[cond] = {
            "pool_size": len(kept),
            "diversity": diversity,
            "difficulty_mean": float(score.mean()),
            "difficulty_std": float(score.std()),
            "stems": kept,
        }

    return stats


# ---------------------------------------------------------------------------
# 分配策略
# ---------------------------------------------------------------------------
def _largest_remainder(weights: np.ndarray, total: int, caps: np.ndarray) -> np.ndarray:
    """按权重把 total 个配额分成整数，并遵守每个条件的池上限。

    用最大余额法（largest remainder）而非简单取整：
    直接 round 会导致配额之和不等于 total，而预算必须精确用完，
    否则不同策略的实际标注量不同，对比就不公平了。

    超出池上限的部分会重新分配给还有余量的条件，迭代直到稳定。
    """
    weights = np.maximum(weights.astype(np.float64), 0.0)
    caps = caps.astype(np.int64)
    k = len(weights)
    total = int(min(total, caps.sum()))  # 预算不能超过所有池子总量

    alloc = np.zeros(k, dtype=np.int64)
    active = np.ones(k, dtype=bool)  # 还能再分的条件
    remaining = total

    # 迭代：分配 -> 削顶 -> 把溢出量分给剩余条件
    for _ in range(k + 2):
        if remaining <= 0 or not active.any():
            break
        w = np.where(active, weights, 0.0)
        if w.sum() <= 0:
            # 权重全为 0，退化为在活跃条件上平均分配
            w = active.astype(np.float64)
        share = w / w.sum() * remaining

        base = np.floor(share).astype(np.int64)
        # 剩余名额按小数部分从大到小分配
        left = remaining - base.sum()
        if left > 0:
            frac = share - base
            frac[~active] = -1.0
            for i in np.argsort(-frac)[:left]:
                base[i] += 1

        alloc += base
        # 削顶：超过池上限的截断
        over = np.maximum(alloc - caps, 0)
        alloc = np.minimum(alloc, caps)
        active &= alloc < caps
        remaining = int(over.sum())

    return alloc


def allocate(
    stats: dict[str, dict],
    total_budget: int,
    strategy: str = "proportional",
    div_diff_alpha: float = 0.5,
    min_per_condition: int = 1,
) -> dict[str, int]:
    """把总预算分配到各条件。返回 {条件ID: 配额}。

    min_per_condition 保证每个条件至少分到这么多样本（受其池容量限制）。
    为什么需要：纯比例分配下，小条件可能被分到 0
    （例如池 [1000, 1000, 5]、N=100 时第三个条件配额为 0），
    那么该采集条件下模型完全没有训练数据，实际部署时在该条件下会失效。
    设为 0 可关闭此保障（用于消融实验，验证保障本身是否有价值）。
    """
    conds = sorted(stats)
    pools = np.array([stats[c]["pool_size"] for c in conds], dtype=np.float64)
    divs = np.array([stats[c]["diversity"] for c in conds], dtype=np.float64)
    diffs = np.array([stats[c]["difficulty_mean"] for c in conds], dtype=np.float64)

    def _norm(x: np.ndarray) -> np.ndarray:
        """归一化到 [0.05, 1]。下界非 0，避免某条件权重恒为 0 而被完全排除
        （每个采集条件至少应有代表，否则该条件下模型完全没见过数据）。"""
        lo, hi = x.min(), x.max()
        if hi <= lo:
            return np.full_like(x, 0.5)
        return 0.05 + 0.95 * (x - lo) / (hi - lo)

    if strategy == "proportional":
        w = pools
    elif strategy == "uniform":
        w = np.ones_like(pools)
    elif strategy == "sqrt":
        w = np.sqrt(pools)
    elif strategy == "diversity":
        w = _norm(divs)
    elif strategy == "difficulty":
        w = _norm(diffs)
    elif strategy == "div_diff":
        # 几何加权，与 selection.py 的 kcenter_greedy 保持同一套参数化逻辑：
        # alpha=0 纯多样性，alpha=1 纯难度，中间连续插值。
        a = float(np.clip(div_diff_alpha, 0.0, 1.0))
        w = np.power(_norm(divs), 1.0 - a) * np.power(_norm(diffs), a)
    else:
        raise ValueError(f"未知分配策略：{strategy}")

    caps = pools.astype(np.int64)

    # 先预留最小配额，剩余预算再按权重分配。
    # 这样保证每个采集条件都有训练样本，同时不破坏策略间的相对差异。
    floor = np.minimum(np.full(len(conds), max(0, int(min_per_condition)), dtype=np.int64), caps)
    if floor.sum() > total_budget:
        # 预算连最小配额都不够：按权重决定哪些条件能拿到这 1 个名额，
        # 避免总是偏向字母序靠前的条件
        alloc = np.zeros(len(conds), dtype=np.int64)
        order = np.argsort(-w)
        for i in order[:total_budget]:
            alloc[i] = min(1, caps[i])
        return {c: int(n) for c, n in zip(conds, alloc)}

    rest = _largest_remainder(w, total_budget - int(floor.sum()), caps - floor)
    alloc = floor + rest
    return {c: int(n) for c, n in zip(conds, alloc)}


# ---------------------------------------------------------------------------
# 组合入口：分配 + 条件内选帧
# ---------------------------------------------------------------------------
def select_with_allocation(
    root: str | Path,
    split: str,
    total_budget: int,
    alloc_strategy: str = "proportional",
    frame_strategy: str = "hybrid",
    seed: int = 0,
    all_stems: list[str] | None = None,
    condition_pattern: str | None = None,
    condition_map: dict[str, str] | None = None,
    difficulty_weights: dict | None = None,
    hybrid_alpha: float = 0.5,
    div_diff_alpha: float = 0.5,
    return_detail: bool = False,
):
    """完整的两层选择：先在条件间分配预算，再在条件内选帧。

    返回选中的样本名列表；return_detail=True 时额外返回分配明细，
    用于论文里报告"每个条件分到了多少、为什么"。
    """
    from .data import list_samples

    stems = all_stems if all_stems is not None else list_samples(root, split)
    groups = group_by_condition(stems, condition_pattern, condition_map)

    stats = condition_stats(root, split, groups, difficulty_weights)
    quotas = allocate(stats, total_budget, alloc_strategy, div_diff_alpha)

    picked: list[str] = []
    detail: dict[str, dict] = {}

    for cond in sorted(quotas):
        n_c = quotas[cond]
        if n_c <= 0:
            detail[cond] = {"quota": 0, "picked": [], **{k: v for k, v in stats[cond].items() if k != "stems"}}
            continue
        # 每个条件用不同但可复现的种子，避免所有条件选出相同位置的样本
        sub_seed = seed * 1000 + abs(hash(cond)) % 997
        chosen = select_frames(
            root, split, n_c,
            strategy=frame_strategy, seed=sub_seed,
            all_stems=stats[cond]["stems"],
            difficulty_weights=difficulty_weights,
            hybrid_alpha=hybrid_alpha,
        )
        picked.extend(chosen)
        detail[cond] = {
            "quota": n_c, "picked": chosen,
            **{k: v for k, v in stats[cond].items() if k != "stems"},
        }

    picked = sorted(set(picked))
    if return_detail:
        return picked, detail
    return picked


# ---------------------------------------------------------------------------
# 诊断：打印各策略的分配结果对比
# ---------------------------------------------------------------------------
def compare_allocations(
    root: str | Path,
    split: str,
    total_budget: int,
    condition_pattern: str | None = None,
    condition_map: dict[str, str] | None = None,
    difficulty_weights: dict | None = None,
) -> dict:
    """并排展示各分配策略的结果。用于快速判断策略间是否真有区别。

    若所有策略给出几乎相同的配额，说明该数据集的条件间差异太小，
    这个贡献点在此数据上无法验证 —— 早发现比跑完实验才发现好。
    """
    from .data import list_samples

    stems = list_samples(root, split)
    groups = group_by_condition(stems, condition_pattern, condition_map)
    stats = condition_stats(root, split, groups, difficulty_weights)

    strategies = ["proportional", "uniform", "sqrt", "diversity", "difficulty", "div_diff"]
    table = {s: allocate(stats, total_budget, s) for s in strategies}

    conds = sorted(stats)
    print(f"总预算 N={total_budget}，共 {len(conds)} 个条件\n")
    print(f"{'条件':<20}{'池大小':>8}{'多样性':>9}{'难度':>8}   " +
          "".join(f"{s[:9]:>10}" for s in strategies))
    print("-" * (45 + 10 * len(strategies)))
    for c in conds:
        st = stats[c]
        row = f"{c:<20}{st['pool_size']:>8}{st['diversity']:>9.4f}{st['difficulty_mean']:>8.4f}   "
        row += "".join(f"{table[s][c]:>10}" for s in strategies)
        print(row)
    print("-" * (45 + 10 * len(strategies)))
    print(f"{'合计':<20}{sum(st['pool_size'] for st in stats.values()):>8}{'':>9}{'':>8}   " +
          "".join(f"{sum(table[s].values()):>10}" for s in strategies))

    # 策略间差异度：若配额向量几乎相同，这个贡献点在该数据上做不出来
    print("\n策略间配额差异（L1 距离，0=完全相同）:")
    vecs = {s: np.array([table[s][c] for c in conds], dtype=float) for s in strategies}
    max_d = 0.0
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1:]:
            d = float(np.abs(vecs[s1] - vecs[s2]).sum())
            max_d = max(max_d, d)
            if d > 0:
                print(f"  {s1:<13} vs {s2:<13}: {d:>7.0f}")
    if max_d < 0.02 * total_budget * len(conds):
        print("\n[!] 各策略配额高度相似 —— 该数据集条件间差异太小，"
              "预算分配这个贡献点在此数据上难以验证。")

    return {"stats": {k: {kk: vv for kk, vv in v.items() if kk != 'stems'}
                      for k, v in stats.items()}, "allocations": table}
