# 与两篇高重合度文献的差异化论述

本文档用于撰写论文的 Related Work 与 Introduction 中的"研究空白"段落。
所有事实均来自 2026-09 对原文/HTML 全文/代码仓库的核查，标注了置信度。

> ⚠️ 标记为「待核实」的条目我无法从公开渠道确认（MDPI 正文返回 403）。
> **投稿前必须自己下原文逐条核对**，这些是论述的承重点，错了会被审稿人直接击穿。

---

## 一、两篇文献的准确画像

### 文献 A — Chen & Tong, arXiv:2507.10084 (IJRS)

*A Transfer Learning-Based Method for Water Body Segmentation in Remote Sensing Imagery: A Case Study of the Zhada Tulin Area*

| 项目 | 事实 | 置信度 |
|---|---|---|
| 平台 | **纯卫星，无任何无人机数据** | 已核实（全文 HTML） |
| 源域 | 自建 "Dataset A"：GF-7（0.8m 全色 / 3.2m 多光谱），207 景 → 3875 个 512² 图块 | 已核实 |
| 目标域 | "Dataset B"：GF-2，20 景 → **180 个图块**，西藏札达土林 | 已核实 |
| 划分 | **随机 9:1 训练/验证，无独立测试集** | 已核实 |
| 模型 | SegFormer MiT-B5 | 已核实 |
| 关键结果 | 水体 IoU：直接迁移 25.50% → 微调后 **64.84%** | 已核实 |
| 对照 | SegFormer 从头训 37.47%；U-Net 从头训 48.82% | 已核实 |
| 选样策略 | **无。随机划分** | 已核实 |
| 标注预算研究 | **无。单一操作点** | 已核实 |
| 多种子/方差 | **无。单次结果** | 已核实 |

**作者自陈的局限（§4.2）**：承认本研究"主要验证了经典的预训练-微调范式"（primarily
validated the classic pre-train-fine-tune paradigm），并在未来工作中提出应采用更强的
域适应方法"以更少标注缩小域间隙"。

> 📌 **这句自陈是你最好的切入点** —— 作者本人指出了这个方向的未完成状态。

**一个必须注意的细节**：该文选择 GF-2 作为目标域的理由是其"分辨率与 GF-7 一致"
(resolution consistent with GF-7)，即**刻意保持了尺度一致性**。所以它处理的域间隙
本质上是*辐射/地域*差异，而非*平台/尺度*差异。

---

### 文献 B — Liao et al., Appl. Sci. 2025, 15, 6292

*A Lightweight Network for Water Body Segmentation in Agricultural Remote Sensing Using Learnable Kalman Filters and Attention Mechanisms* (LKF-DCANet)
代码：https://github.com/Estrellading/LKF-DCANet

| 项目 | 事实 | 置信度 |
|---|---|---|
| 核心贡献 | **轻量化网络架构**：CADCN 编码器 + CATM/可学习卡尔曼滤波解码器，0.22M 参数 | 已核实 |
| 辅助手段 | 特征级知识蒸馏（MSE 对齐中间层） | 已核实 |
| 源域 | GLH-Water 公开数据集 | 已核实 |
| 目标域 | **自建无人机数据集**：四川东风渠灌区（30.52°N, 104.23°E），DJI Mavic 3E，约 3.52 km²，含渠道/农田水面/人工塘 | 已核实 |
| 标注比例 | 微调仅用 **10% 标注**，再对余下 90% 推理生成伪标签 | 已核实 |
| 结果 | 公开数据集 IoU 85.95%；自建无人机数据集 IoU **96.28%** | 已核实 |
| 10% 如何选出 | **仓库用 `voc_annotation.py`（VOC 模板脚本，按惯例是 `random.sample` 随机划分）** | **待核实** —— 需读该脚本源码确认 |
| 标注预算研究 | README 与摘要均无标注比例消融（5%/20%/50%） | **待核实**（正文 403） |
| 多种子/方差 | 仓库 README 全为单一数值，无种子、无置信区间 | 已核实（仓库层面） |
| 数据集是否公开 | **是**，百度网盘（提取码 2025），权重亦公开 | 已核实 |

**一个必须自己判断的疑点**：多处表述为"对余下 90% 无标注数据推理，生成伪标签
**用于评估**"（generating pseudo-labels for evaluation）。若确实是拿自己生成的伪标签
当作评估基准，则该评估是循环论证，96.28% 这个数值也就不可作为真实精度看待。
**但这可能是摘要转述的歧义**，正文也许有独立的人工标注测试集。务必自己确认。

---

## 二、重合度的准确判断（修正我此前的结论）

我此前把文献 A 标为"高重合"、文献 B 标为"几乎完全重合"，**这个判断过重了**。核实后：

| | 研究**场景**重合 | 研究**贡献**重合 |
|---|---|---|
| 文献 A | **低** —— 卫星→卫星，刻意保持分辨率一致，无无人机 | 低 —— 贡献是两阶段迁移范式的区域验证 |
| 文献 B | **高** —— 无人机 + 水体 + 10% 标注 + 迁移 + 四川灌区 | **低** —— 贡献是轻量化架构 + 知识蒸馏 |

**准确的表述是：文献 B 与你的场景高度重合，但与你的贡献并不重合。**

它的 "10% 标注" 是用来*展示*所提出架构在低标注下仍可用，是一个演示性操作点；
而不是把标注量当作*被研究的自变量*。两者的区别，就是"我的模型在 10% 标注下能到
96%"和"同样 10% 的预算，标哪 10% 能最大化收益"之间的区别。

---

## 三、四个可辩护的研究空白

### 空白 1：两篇都在回答"怎么训"，没人回答"标哪些"

这是最核心、最干净的差异化。

- 文献 A：随机 9:1 划分（已核实）
- 文献 B：`voc_annotation.py` 随机划分（待核实，但 VOC 模板惯例如此）

两者都**默认标注子集是给定的**，优化的是在该子集上的训练方式（两阶段迁移 / 轻量化
架构 + 蒸馏）。而"哪些样本值得标注"这个问题在两篇中都未被提出。

**正交性论证**（这是让审稿人接受你工作的关键）：
你的选帧策略与它们的训练方法**不竞争、可叠加**。选帧决定*输入哪些数据*，
迁移/蒸馏决定*如何利用这些数据*。你的方法可以直接接在 LKF-DCANet 的 pipeline 前端。
建议在实验中真的做这一组（用你的策略选出的 10% 替换其随机 10%），
这会是最有说服力的一张表。

### 空白 2：没有标注预算曲线，"10%" 是个未经检验的数字

文献 B 报告了单一的 10% 操作点。由此无从得知：
- 5% 是否已经足够？（若足够，则 10% 浪费了一半标注成本）
- 精度在何处饱和？
- 随机选 10% 的**方差**有多大？（单次结果无法回答）

你的贡献 1（预算曲线 + 饱和点 + 多种子误差带）提供的是这两篇都不具备的信息。
注意：预算曲线本身撑不起论文（这是消融性质），但作为**承载策略对比的载体**是必要的。

### 空白 3：真正的跨平台尺度间隙无人量化

| | 源域 GSD | 目标域 GSD | 尺度比 |
|---|---|---|---|
| 文献 A | GF-7 0.8m | GF-2（刻意选择的同分辨率） | **≈1×** |
| 文献 B | GLH-Water（卫星） | 无人机低空 | 未量化 |
| **本文** | LoveDA 0.3m | 无人机 ≈1.5cm（FloodNet）/ 自采 | **≈20× 以上** |

文献 A 明确为了"尺度一致性"而选择 GF-2，等于主动规避了尺度间隙。
文献 B 虽跨了平台，但重点在架构轻量化，未把域间隙本身作为研究对象拆解
（GSD、视角、镜面反光各自贡献多少）。

**你的贡献**：在统一协议下，把卫星→无人机的域间隙分解为可测量的分量，
并给出每个分量下的标注预算响应。这需要你的自采数据（可控高度/角度/时段）才能做，
是别人拿公开数据做不了的部分 —— **这也是你的六旋翼的真正价值所在**。

### 空白 4：小样本实验缺乏方差报告

两篇均为单次运行的单一数值。在 N=180 图块（文献 A）或 10% 子集（文献 B）这种量级，
不同随机划分间的方差极可能达到数个 IoU 点。

⚠️ **措辞警告**：不要在 Related Work 里指控他人"不严谨"—— 审稿人有可能就是原作者。
正确做法是把它写成**你自己协议的设计动因**：

> ✅ 推荐写法："Since results in the low-label regime are known to exhibit high
> variance across data splits, we report mean ± standard deviation over K random
> seeds and evaluate on a held-out test set that is never used for selection or
> model selection."
>
> ❌ 避免写法："Prior work reports only single-run results and lacks a held-out
> test set, which makes their numbers unreliable."

同理，文献 A 无独立测试集（模型选择与结果报告同用那 18 个验证图块）这一点，
**只用来指导你自己的协议设计，不要写成对它的批评。**

---

## 四、一个意外发现：文献 B 反而支持你的难度先验

LKF-DCANet 明确说明其自建无人机数据集是为了测试：

> long/narrow water bodies, fragmented distributions, and strong background
> interference (**building occlusion, vegetation cover, topographic shadows**)

这与你 `selection.py` 中的难度先验高度吻合 —— 尤其是**阴影**（shadow）和
**细长/破碎结构**。

**这个发现有三重价值**：

1. **你的难度先验不是凭空设计的**，有独立文献支持这些确实是无人机水体分割的真实
   难点。引用它来论证先验的合理性，比自己声称"我认为反光和阴影很难"有力得多。
2. **暴露出一个精确的空白**：文献 B *识别*了这些难点，但只用它们来*描述数据集特性*，
   从未用它们来*驱动样本选择*。你做的正是这一步 —— 把已知的难点从"数据集描述"
   转化为"选样准则"。这是一个很好讲的故事。
3. **它的数据集已公开**（百度网盘，提取码 2025）。**强烈建议下载作为第二个目标域**：
   - 可直接与其 10% 随机基线做同预算对比
   - 多一个目标域能大幅增强泛化性论证
   - 省去部分自采数据的时间压力

另外，文献 A 的局限自陈（"主要验证了经典预训练-微调范式"）与未来工作
（"以更少标注缩小域间隙"）可以直接引用，作为本方向仍待推进的文献依据。

---

## 五、可直接改写使用的英文草稿

> 以下为草稿，需在你的真实结果产出后调整措辞，并补全引用格式。
> **不要原样照抄** —— 结合你最终的实验设置重写。

### Related Work 段落

> **Label-efficient water body segmentation.** Transfer learning has become the
> standard remedy for annotation scarcity in water body segmentation. Chen and
> Tong [A] propose a two-stage strategy in which SegFormer is pre-trained on a
> heterogeneous GF-7 source domain and fine-tuned on a small GF-2 target set of
> the arid Zhada Tulin area, raising water IoU from 25.50% under direct transfer
> to 64.84%. Liao et al. [B] instead target model efficiency, introducing
> LKF-DCANet — a 0.22M-parameter network combining deformable convolutional
> attention with a learnable Kalman filter — and transfer it to a self-collected
> UAV dataset over the Dongfengqu irrigation district using only 10% of the
> labels, with the remaining images pseudo-labeled.
>
> These works establish that pre-training and architectural design can
> substantially reduce the number of target-domain labels required. They are,
> however, concerned with *how to train* on a labeled subset that is assumed to
> be given: in both cases the labeled subset is obtained by random splitting, and
> the label ratio is fixed at a single operating point rather than treated as a
> variable under study. Consequently it remains unknown how much of the reported
> label efficiency is attributable to the method itself, and how much could be
> gained — at zero additional annotation cost — by choosing *which* images to
> annotate. Our work is orthogonal and complementary: we keep the training recipe
> fixed and ask how a fixed annotation budget should be allocated across an
> unlabeled UAV image pool.

### Introduction 中的空白陈述

> Existing label-efficient approaches to UAV water body segmentation optimize the
> training procedure while treating the annotated subset as given, typically drawn
> at random. Three questions therefore remain open. First, for a fixed annotation
> budget, does the *choice* of images to annotate matter, and by how much?
> Second, where does accuracy saturate as the budget grows — i.e. how few images
> actually suffice? Third, prior cross-domain studies either preserve ground
> sampling distance between source and target [A] or do not decompose the
> platform gap [B], leaving the satellite-to-UAV scale gap unquantified as a
> function of annotation budget. We address these questions with a
> redundancy-aware, water-specific frame selection strategy and a unified
> annotation-budget benchmark.

### 正交性声明（供 Discussion 使用）

> We emphasise that frame selection is orthogonal to the training-side advances of
> [A] and [B]: selection determines *which* samples enter the labeled pool, while
> staged transfer and distillation determine *how* that pool is exploited. To make
> this concrete, we substitute our selected subset for the random subset at the
> same budget in an otherwise unchanged pipeline, and observe [结果待填].

---

## 六、投稿前必做的核查清单

- [ ] 下载文献 A 的 PDF，核对表 1/表 2 数值、确认确无测试集与选样策略
- [ ] 下载文献 B 的 MDPI 正文，核对三件事：
      - [ ] 10% 子集的选取方式（是否随机）← **论述承重点**
      - [ ] 是否存在标注比例消融 ← **论述承重点**
      - [ ] 96.28% 是在人工标注测试集还是自生成伪标签上得到的
- [ ] 读 `LKF-DCANet/voc_annotation.py`，确认划分逻辑
- [ ] 下载其公开的东风渠无人机数据集，评估能否作为第二目标域
- [ ] 检索 2026 年新发表的相关工作（本文档基于 2026-09 的检索，可能已有新进展）
- [ ] 确认文献 A 的期刊版本（IJRS）与 arXiv 版本是否有实质差异，引用应以期刊版为准

---

## 附：一个必须避开的方法论陷阱

文献 A 使用 512×512 窗口、128 像素步长切图（相邻图块重叠 75%），
随后对图块做**随机** 9:1 划分。这种做法会让空间上高度重叠的相邻图块
同时落入训练集和验证集，造成**空间信息泄漏**，从而虚高指标。

**你的代码里有同样的风险** —— `prepare_data.py` 的 `--stride` 支持重叠切图。
务必遵守：

1. **按原始影像（而非图块）划分 train/val/test**，确保同一景影像的图块不跨集合
2. 或者在划分时保证集合间有空间缓冲带
3. 自采数据尤其危险：同一水体的连续航拍帧几乎相同，必须按**飞行架次/地点**划分，
   而不是按帧随机划分

这一点在你的论文里应当明确写出协议（"splits are constructed at the flight/scene
level to preclude spatial leakage between overlapping tiles"），
既是严谨性加分项，也是与前人工作的一个实质差别。
