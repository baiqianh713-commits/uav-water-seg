"""标注帧选择策略 —— 本论文的核心创新点所在。

问题设定（"冷启动"标注预算）
----------------------------
你有一堆**完全无标注**的无人机影像/视频帧，标注预算只有 N 张。
选哪 N 张去标注？注意此时还没有任何标注，所以不能用任何需要标签的信息，
也不能用测试集——一旦用了，整篇论文作废。

因此这里所有策略都只依赖**图像本身**（无监督特征），这既是方法上的约束，
也恰好是实际应用中的真实情形：拿到一段航拍视频，要先决定标哪几帧。

策略
----
  random    : 随机选。这是主要对手（baseline），必须打败它。
  uniform   : 按时间/文件名等间隔选。无人机视频的朴素做法。
  kcenter   : k-center greedy 最大化特征空间覆盖度，即"冗余感知"——
              相邻帧高度相似，选了一帧就不该再选它的近邻。
  stratified: 按水体难度代理指标分层均匀采样。
  hybrid    : 冗余感知 + 水体困难样本导向的组合（本文提出的方法）。

⚠️ 论文写作提醒：kcenter 和分层采样都是已有的通用主动学习方法，
不能声称是自己的创新。可辩护的创新点在于**水体专属的难度先验**
（镜面反光 / 阴影混淆 / 无纹理）以及它们与冗余感知的结合方式。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 无监督特征提取
# ---------------------------------------------------------------------------
def extract_features(
    root: str | Path,
    split: str,
    stems: list[str],
    thumb_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """为每张图提取两组特征。

    返回
    ----
    embed      : (N, D) 外观嵌入，用于衡量"两帧是否冗余"
    difficulty : (N, 4) 水体难度代理指标 [反光, 阴影, 纹理平坦度, 蓝绿倾向]
    stems      : 与上面两个数组行对应的样本名

    为什么手工特征而不是深度特征：
    1. 冷启动时没有目标域训好的模型，用 ImageNet 特征反而可能引入域偏差。
    2. 手工特征可解释，论文里能讲清楚每一维的物理含义。
    3. 纯 CPU 就能算，不占用租来的 GPU 时间。
    后续可以加一组 ImageNet/DINO 特征做对比实验，看哪种嵌入选出的帧更好。
    """
    img_dir = Path(root) / split / "images"
    embeds, diffs, kept = [], [], []

    for stem in stems:
        path = img_dir / f"{stem}.png"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        small = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # --- 外观嵌入：颜色直方图 + 缩略图灰度 ---
        # 颜色直方图捕捉场景整体色调，缩略图保留粗略空间布局。
        # 两者拼接后，视觉相近的帧在欧氏距离下会靠得很近。
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        thumb = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).flatten() / 255.0
        embeds.append(np.concatenate([hist, thumb]).astype(np.float32))

        # --- 难度代理指标（针对水体的三个物理难点）---
        # 1. 镜面反光：高亮度 + 低饱和度的像素占比。
        #    水面反光会把水像素"洗白"，模型容易误判为非水。
        glare = float(((v > 200) & (s < 60)).mean())

        # 2. 阴影：低亮度像素占比。
        #    建筑/植被阴影和深水在 RGB 上极易混淆，是误报主要来源。
        shadow = float((v < 55).mean())

        # 3. 纹理平坦度：Laplacian 方差的倒数。
        #    水面无纹理是它最稳定的视觉线索，但平坦的路面、屋顶同样无纹理，
        #    所以纹理平坦区域多的图，恰恰是难以区分的图。
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        flatness = 1.0 / (1.0 + lap_var / 100.0)

        # 4. 蓝绿倾向：偏蓝青色调的像素占比，粗略表征可能含水。
        #    OpenCV 的 H 通道范围是 0-180，蓝青大致在 85-130。
        bluegreen = float(((h > 85) & (h < 130) & (s > 40)).mean())

        diffs.append([glare, shadow, flatness, bluegreen])
        kept.append(stem)

    if not kept:
        raise RuntimeError(f"{img_dir} 下没有可读取的图像")

    embed = np.stack(embeds)
    # L2 归一化，让欧氏距离等价于余弦距离，避免某几维数值大就主导距离
    embed = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-9)

    return embed, np.array(diffs, dtype=np.float32), kept


# ---------------------------------------------------------------------------
# 各策略
# ---------------------------------------------------------------------------
def select_random(stems: list[str], n: int, seed: int = 0) -> list[str]:
    """随机选 N 张。主要对照组。"""
    rng = np.random.default_rng(seed)
    n = min(n, len(stems))
    idx = rng.choice(len(stems), size=n, replace=False)
    return [stems[i] for i in sorted(idx)]


def select_uniform(stems: list[str], n: int) -> list[str]:
    """按排序后的位置等间隔选。

    对连续航拍视频帧来说，文件名顺序≈时间顺序，
    等间隔采样是工程上最常见的朴素做法，值得作为对照。
    """
    n = min(n, len(stems))
    if n == 0:
        return []
    idx = np.linspace(0, len(stems) - 1, n).round().astype(int)
    return [stems[i] for i in sorted(set(idx.tolist()))]


def kcenter_greedy(
    embed: np.ndarray,
    n: int,
    difficulty: np.ndarray | None = None,
    alpha: float = 0.0,
    seed: int = 0,
) -> list[int]:
    """覆盖度 + 难度的贪心选择。本文方法的核心算法。

    基础形式（alpha=0）是标准的 k-center greedy：每次选距离已选集合最远的样本，
    使选出的 N 个样本尽可能覆盖整个特征空间，自动避开互相冗余的近重复帧
    ——这正是无人机视频最需要的（相邻帧几乎一样，标了一帧不该再标它的邻居）。

    加入难度后用**几何加权**组合两个准则：

        效用 = 最小距离^(1-alpha) × 难度^alpha

    为什么用几何加权而不是 `距离 × (1 + alpha·难度)`：
    后者在数学上是失效的参数化。因为 argmax 对权重的正比例缩放不变，
    当 alpha 增大时 `1 + alpha·难度` 趋近 `alpha·难度`，选择结果很快饱和，
    而且距离项永远存在，无论 alpha 多大都到不了"纯难度导向"。
    几何加权下 alpha 是真正的插值系数：
        alpha = 0   -> 纯覆盖度（严格等价于 k-center）
        alpha = 1   -> 近似纯难度导向（见下方关于起始点的说明）
        0 < alpha < 1 -> 两者连续折中
    这样消融实验才能扫出有意义的 alpha 曲线。

    注意：起始点是**随机**选的，不是按难度选的。所以 alpha=1 时结果是
    "随机起始点 + 难度前 N-1"，与严格的"难度前 N"差一个样本。
    这是刻意保留的——随机起始点让不同 seed 产出不同子集，
    才能报均值±标准差。若需严格的纯难度基线，用 strategy="difficulty"。
    """
    rng = np.random.default_rng(seed)
    num = embed.shape[0]
    n = min(n, num)
    if n == 0:
        return []

    alpha = float(np.clip(alpha, 0.0, 1.0))

    if difficulty is None or alpha == 0.0:
        diff_term = None
    else:
        # 归一化到 [0.05, 1]：下界不取 0，否则难度最低的样本
        # 在 alpha>0 时效用恒为 0，会被永久排除在候选之外
        d = np.asarray(difficulty, dtype=np.float64)
        lo, hi = d.min(), d.max()
        d = (d - lo) / (hi - lo) if hi > lo else np.full_like(d, 0.5)
        diff_term = np.power(0.05 + 0.95 * d, alpha)

    # alpha=1 时距离项的指数为 0，退化成纯难度排序；
    # 此处直接走贪心也能得到同样结果，不必特殊处理。

    # 第一个点随机选，保证多种子下结果有变化，可以报均值±标准差
    first = int(rng.integers(num))
    selected = [first]

    # 维护每个样本到已选集合的最小距离，避免每轮重算全部距离矩阵
    min_dist = np.linalg.norm(embed - embed[first], axis=1)

    for _ in range(n - 1):
        # 距离项加 eps 再取幂，防止 0^负数 或 0 距离吞掉整个乘积
        dist_term = np.power(min_dist + 1e-12, 1.0 - alpha)
        utility = dist_term if diff_term is None else dist_term * diff_term
        utility = utility.copy()
        utility[selected] = -np.inf  # 已选的不再重复选
        nxt = int(np.argmax(utility))
        selected.append(nxt)
        # 增量更新最小距离
        min_dist = np.minimum(min_dist, np.linalg.norm(embed - embed[nxt], axis=1))

    return sorted(selected)


def select_stratified(scores: np.ndarray, n: int, n_bins: int = 5, seed: int = 0) -> list[int]:
    """按一维得分分层，每层等额抽样。

    保证选出的样本在难度上覆盖从易到难，而不是全挤在某个区间。
    """
    rng = np.random.default_rng(seed)
    num = len(scores)
    n = min(n, num)
    if n == 0:
        return []

    # 用分位数划分，保证每层样本数大致相等（等宽划分会导致某些层为空）
    edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)

    selected: list[int] = []
    per_bin = max(1, n // n_bins)
    for b in range(n_bins):
        members = np.where(bins == b)[0]
        if len(members) == 0:
            continue
        take = min(per_bin, len(members))
        selected.extend(rng.choice(members, size=take, replace=False).tolist())

    # 数量不足时从剩余样本里随机补齐
    if len(selected) < n:
        remaining = np.setdiff1d(np.arange(num), np.array(selected, dtype=int))
        if len(remaining):
            extra = rng.choice(remaining, size=min(n - len(selected), len(remaining)), replace=False)
            selected.extend(extra.tolist())

    return sorted(selected[:n])


def difficulty_score(diffs: np.ndarray, weights: dict | None = None) -> np.ndarray:
    """把 4 维难度指标合成一个标量分数，归一化到 [0, 1]。

    权重是超参数，需要在验证集上调，论文里要做敏感性分析
    （审稿人一定会问这几个权重怎么定的）。
    """
    w = {"glare": 1.0, "shadow": 1.0, "flatness": 0.5, "bluegreen": 0.5}
    if weights:
        w.update(weights)

    # 各维先做 min-max 归一化，否则量纲不同无法相加
    normed = np.zeros_like(diffs)
    for j in range(diffs.shape[1]):
        col = diffs[:, j]
        lo, hi = col.min(), col.max()
        normed[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5

    score = (
        w["glare"] * normed[:, 0]
        + w["shadow"] * normed[:, 1]
        + w["flatness"] * normed[:, 2]
        + w["bluegreen"] * normed[:, 3]
    )
    total = sum(abs(v) for v in w.values())
    return score / (total + 1e-9)


def select_frames(
    root: str | Path,
    split: str,
    n: int,
    strategy: str = "random",
    seed: int = 0,
    all_stems: list[str] | None = None,
    difficulty_weights: dict | None = None,
    hybrid_alpha: float = 0.5,
) -> list[str]:
    """统一入口。返回被选中去"标注"的样本名列表。

    hybrid_alpha 控制冗余感知与难度导向的配比：
      0 → 纯覆盖度（等价于 kcenter）
      1 → 纯难度导向
    """
    from .data import list_samples

    stems = all_stems if all_stems is not None else list_samples(root, split)

    if strategy == "random":
        return select_random(stems, n, seed)
    if strategy == "uniform":
        return select_uniform(stems, n)
    if strategy == "all":
        return stems

    # 以下策略需要图像特征
    embed, diffs, stems = extract_features(root, split, stems)

    if strategy == "kcenter":
        idx = kcenter_greedy(embed, n, seed=seed)
    elif strategy == "stratified":
        idx = select_stratified(difficulty_score(diffs, difficulty_weights), n, seed=seed)
    elif strategy == "difficulty":
        # 纯粹选最难的 N 张。预期表现不好（会全挤在极端反光帧上，
        # 缺乏多样性），但作为消融实验很有价值——能证明"光选难的不够，
        # 必须同时保证覆盖度"。
        idx = np.argsort(-difficulty_score(diffs, difficulty_weights))[:n].tolist()
    elif strategy == "hybrid":
        # 本文方法：覆盖度与水体难度的几何加权组合，hybrid_alpha 为插值系数
        score = difficulty_score(diffs, difficulty_weights)
        idx = kcenter_greedy(embed, n, difficulty=score, alpha=hybrid_alpha, seed=seed)
    else:
        raise ValueError(f"未知选帧策略：{strategy}")

    return [stems[i] for i in idx]
