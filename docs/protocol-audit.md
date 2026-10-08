# 2.4 节文献口径普查：抽样框、核对记录与计数

> 本文件是论文 2.4 节「该方向的报告惯例」一说的**全部证据**。
> 审稿人若质疑「凭什么是这些论文」，答案必须能在本文件里找到。
>
> 🔴 纪律（开始检索**之前**写定，不得事后修改；如需改动，
> 在文末「规则修订记录」里写明改了什么、为什么、改动前已核过几篇）：
>
> 1. **必须真读正文。** 判定一律引用正文原句并注明出处位置。
>    只看摘要不得填表。拿不到全文的记为「排除：无法获取全文」，
>    **不得按印象、按摘要、按同作者其它论文推测**。
> 2. **命中即读，不得跳过。** 按下述排序命中的论文，读不到就记为排除，
>    不得换下一篇顶上 —— 跳过等于偷偷挑了对本文有利的样本。
> 3. **分母如实写。** 最终计数写成「N 篇已核 / M 篇因无法获取全文排除」，
>    百分比的分母是 N，不是 N+M，且 M 必须一并报告。
> 4. **不点名批评。** 本表只记录客观的「报告了 / 未报告」，
>    不对任何论文的严谨性作评价；正文引用本表时写「该方向的报告惯例」。
> 5. **不得凭记忆拼 arXiv 号或 URL。** 每篇的全文地址必须先检索确认，
>    下载后**必须核对首页标题与作者**再填表。
>    （2026-10-05 立：我按记忆猜了两个 arXiv 号，`2101.02323` 猜 Cai CVPR 2021
>    实为另一篇生物医学分割主动学习，`2204.07065` 猜 Rangnekar 实为一篇数学优化
>    论文。两份已删。若不核首页就填表，这份普查会凭空出现两篇不存在的证据。）

## 一、抽样框（2026-10-05 写定，检索前）

**时间窗**：2021–2026（近 5 年）。

**三类来源，各取若干**：

| 类 | 范围 |
|---|---|
| A | 主动学习 / 样本选择的**评估方法学**类（benchmark、survey、reality-check 类） |
| B | **低标注预算语义分割**类（主动学习、半监督、少样本的标注量—精度曲线） |
| C | **遥感 / 无人机**的标注预算或样本选择类 |

**排序与取样**：每类按引用量（Google Scholar / Semantic Scholar）或会议期刊等级
排序，自上而下取。目标总量 15–20 篇，已核 4 篇计入其中。

**纳入判据**：论文须绘制或报告「标注量—精度」关系（多个标注预算下的性能），
否则与本文的论点无关，记为「排除：无预算曲线」。

## 二、三项核对口径（判定标准写死）

对每篇论文，核以下三项。**判定一律以正文可见的陈述为准**，
附录与开源代码**不计入**「正文是否报告」——本文的论点是**论文正文的报告惯例**，
代码里能翻出来不等于论文报告了（但若正文明确指向代码，另在备注说明）。

| 项 | 判「是」的条件 |
|---|---|
| ① 报告训练轮数 | 正文给出 epoch 数（固定值或各设置的值） |
| ② 报告总梯度步数 | 正文给出 iteration / step / update 的总数 |
| ③ 每图遍历次数可从正文无歧义确定 | 见下方细则 |
| ④ 给出 $N$ 与步数（或遍历次数）的算术关系 | 正文写出标注量、批大小、轮数与总步数之间的关系式或等价陈述 |
| ⑤ 量化训练量对结论的影响 | 正文给出改变训练量后结论/排序如何变化的实测 |
| ⑥ 指出或控制「跨预算训练量不等」 | 见下方细则。**这一项才是本文所称的空白所在** |

### ③ 的细则（2026-10-05 修订，理由见 §五）

判「是」的条件是：正文给出的信息足以**无歧义地**算出，在**每一个预算点上**，
每张已标注图像被梯度更新看到的次数。下列任一成立即判「是」：

- (a) 直接给出该次数；
- (b) 给出轮数，**且**该框架中「一轮 = 完整遍历一次当前标注集」
  （此时遍历次数 = 轮数）；
- (c) 同时给出总步数、批大小、标注集大小（此时遍历次数 $approx S dot B slash N$）；
- (d) 给出明确的收敛/早停判据，使该量不影响结论。

判「否」的典型情形：

- 只给「轮数」，而该框架的「一轮」是**固定步数**而非一遍数据
  （如 nnU-Net 定义一轮 = 250 次迭代）；
- 标注集被**上采样/重采样到固定规模**后再按轮训练（此时遍历次数 $= E dot M slash N$，
  需额外知道 $M$）；
- 只给「训练至收敛」而无任何可操作的判据。

🔴 **(b) 的后果必须正视**：凡「一轮即一遍标注集 + 轮数固定」的论文，
③ 一律判「是」。因此 **2.4 节不得写「该方向普遍不报告每图被遍历次数」**——
按本细则这句话多半是错的。可辩护的空白陈述只能落在 ⑥ 上。

⚠️ ③ 判「是」的 (a)(b)(c) 与 (d) **意义不同**，必须在备注里区分：
前三者是该量可被读出；(d) 是**使该量变得无关紧要**（训练到收敛则遍历次数
多少不影响结论）。(d) 恰恰是本文建议的做法，不是缺陷。计数时分开报，
不得合并成一个数字。

### ⑥ 的细则（2026-10-05 新增）

判「是」的条件是下列任一：

- 正文**指出**在其协议下不同预算点的总步数或每图遍历次数不相等
  （哪怕只是一句话，不要求量化）；
- 正文**采取了使之相等的措施**并说明了用意
  （如上采样至固定规模、按固定总步数训练、训练至收敛）。

判「否」：正文给出了训练配置，但从未提及该配置会使训练量随预算变化，
也未说明任何措施是为此而设。

⚠️ 「采取了措施但未说明用意」**判否，并在备注注明**——这正是
Lüth 2023 的情形（上采样至固定规模，但未命名为偏差来源）。
这类情形在计数时单列，不得并入「否」的主数字去夸大空白。

#### ⑥ 的细则补充：什么才算「措施」（2026-10-05 第 11 篇时写定，见 §五 修订 2）

🔴 **固定轮数与固定总步数这两种默认做法，一律不算「措施」。** 理由：

- 二者是本文所**考察的对象本身**，不是针对本问题所做的处理；
- 二者各自只让**一侧**的量相等：固定轮数使每图遍历次数相等而总步数随预算增长，
  固定总步数使总步数相等而每图遍历次数随预算递减。把任一侧算成「使训练量相等的
  措施」，等于默认另一侧的不等无关紧要 —— 而本文的全部论点恰恰是两侧都要紧。

若不加此限，**每一篇论文都会落进「有措施无说明」**（不是固定轮数就是固定步数），
这一子类就失去了区分力，而「纯否」的数字会被人为压到 0。

**算「措施」的只有**：该做法是一项**非默认的、额外的**安排，其效果是
使训练量随预算的变化**消失或变得无关**。目前仅两类：

- 明确的收敛 / 早停判据（训练量随预算怎么变都不影响结论）—— Beck 2021；
- 把标注集**上采样 / 重采样至固定规模**后再训练（为此多做了一步，
  其唯一效果就是抹平规模差）—— Lüth 2023。

（本补充只是把原细则里「措施」一词的边界写死，**未改动任何已填判定**，
核对情况见 §五 修订 2。）

## 三、核对记录

### 已核（前序会话 4 篇）—— 🔴 已按修订 1 的新判据**重新取证**，两处改判

见论文 2.4 节 `tab:protocol-audit`。四篇为 Mittal 2019、Zhan 2022、
Lüth 2023、Lüth 2025。⚠️ Mittal 2019 在 2021 年时间窗之外，
系作为该方向的奠基性对照纳入，纳入理由在此记明，计数时单列。

本次已重新下载四篇全文逐篇复核（`refs/` 下留档），结论如下。

#### [A-3] Mittal 等 2019，《Parting with Illusions about Deep Active Learning》，arXiv:1912.05361

🔴 **③ 由「否」改判「是」。原判定是错的，论文表格必须改。**

- 全文来源：arXiv:1912.05361。本地留档，12 页 / 48722 字符。首页核对：标题作者均符。
- **① 轮数：是（150 轮，正文）** —— 分类与分割两处均在正文给出：
  > 「We trained all AL methods (without SSL methods) for 150 epochs per sampling
  > cycle with a batch size of 64.」（§4 分类实验设置，第 4 页）

  > 「We use a batch size of 10 and train each cycle for 150 epochs.」（§5 分割实验设置，第 8 页）
- **② 总步数：仅半监督分支（维持原判，已复核）**
  > 「We train the semi-supervised AL methods for 50k iterations per sampling
  > cycle」（第 4 页）；分割侧「trained for 10k iterations for each cycle with a
  > batch size of 8」（第 8 页）
- **③ 每图遍历次数：是（情形 b）** —— 「150 epochs per sampling cycle」，
  其一轮即遍历一次当前标注集，故每图遍历次数恒为 150，各预算点相同。
  **这正是修订 1 所预见的情形**：报了轮数即等于报了遍历次数。
- **④ 否；⑤ 否；⑥ 否（纯否）**
- 备注：该文是「固定轮数」一侧的又一实例（150 轮固定 ⇒ 总步数 $prop N$）。
  与 Mittal 2023 为同一第一作者的两篇不同论文，计数分列。

#### [A-4] Zhan 等 2022，《A Comparative Survey of Deep Active Learning》，arXiv:2203.13450

🔴 **① 与 ③ 均改判为「正文否 / 附录是」。原判定「是（每数据集固定）」未区分正文与附录。**

- 全文来源：arXiv:2203.13450。本地留档，24 页 / 99085 字符。首页核对符。
- **① 轮数：否（正文）／ 是（附录 D.2）** —— 正文明确推给附录：
  > 「More details of experimental settings (i.e., datasets, implementations)
  > are in Section D in Appendix.」（§3.1，第 6 页）

  附录才给出逐数据集的值：
  > 「MNIST, FashionMNIST and EMNIST: number of training epochs is 20 ...
  > CIFAR10, CIFAR100: number of training epochs is 30 ... TinyImageNet:
  > number of training epochs is 40」（Appendix D.2 Implementation details，第 18 页）

  ⚠️ 边界：正文 §3.3 的轮数消融中出现过「trained 30 epochs」等数值，
  但那是**消融的自变量取值**，不是各数据集默认配置的声明。故正文判否。
- **② 总步数：否（维持）**
- **③ 每图遍历次数：否（正文）／ 是（附录，情形 b）** —— 依附录轮数即得 20/30/40。
- **④ 否（维持）**
- **⑤ 是：CIFAR-10 轮数消融（维持，已复核）**
- **⑥ 否（纯否）**

🔴 **备注：发现一个必须进正文红线段落的方向相反实例。**
该文的轮数消融结论与 Lüth 2025 **同向**，即训练越久、选择方法相对随机的优势越大：
> 「Compared with Random, Entropy achieves better performance when the model is
> trained using more epochs, e.g., Entropy boosts AUBC by around 1.5 when the
> model is trained 30 epochs.」（§3.3 Ablation Study，第 8 页）

论文 2.4 节现有的 `#honest` 块只举了 Lüth 2025 一例。
**须补入本例**，使「方向依任务与数据而定」一说有两个独立证据，
并再次重申不得推断本文方法在收敛协议下会显著。

#### [A-5] Lüth 等 2023，《Navigating the Pitfalls of Active Learning Evaluation》，NeurIPS 2023

🔴 **① 与 ③ 改判为「正文否 / 附录是」；上采样措施确认在附录，⑥ 归「有措施、无说明」。**

- 全文来源：arXiv:2301.10625。本地留档。首页核对符。
- **① 轮数：否（正文）／ 是（附录，200 轮）** —— 正文只声明轮数被固定，不给数值：
  > 「For Self-SL and ST models we use a fixed training recipe (Scheduler,
  > Optimizer, Batch Size, Epochs) and only optimize learning rate, weight decay
  > and data augmentations.」（§4 Hyperparameter selection，第 6 页）

  数值在附录：
  > 「ST models are trained for 200 epochs with Cosine Annealing and 10 epochs
  > warmup.」（Appendix，第 25 页）
- **② 部分（半监督以步数计）（维持）**
- **③ 每图遍历次数：否（正文）／ 是（附录，情形 c）** ——
  上采样至固定规模 5500，故遍历次数 $= 200 times 5500 slash N$，需知 $M = 5500$：
  > 「Models trained purely on the labeled dataset upsample it to a size of 5500
  > following [32] if the labeled train set is smaller.」（Appendix，第 25 页）

  🔴 **该上采样措施在附录，不在正文** —— 这一点比原记录更弱，必须如实修正。
  正文的「分类器配置」陷阱项明确只列学习率、权重衰减与数据增强。
- **④ 否（维持）**
- **⑤ 否（维持）**
- **⑥ 否，但属「采取了措施、未说明用意」，单列（维持，证据已落实）**

#### [A-6] Lüth、Traub 等 2025，《nnActive》，TMLR 08/2025（arXiv:2511.19183）

**维持原判，但 ③ 的理由须换成实际证据——不得按 nnU-Net 框架惯例推定。**

- 全文来源：arXiv:2511.19183，TMLR 2025。本地留档，62 页 / 158205 字符。
  ⚠️ 该 PDF 29 MB，`urllib` 一次性读会稳定截断在 14 MB；改用 `curl -C -` 续传才取全。
  这是**下载问题不是获取性问题**，不得记为「无法获取全文」（脚本已修）。
- **① 轮数：是（200 轮，正文）**
  > 「We used the 3D full resolution configuration of nnU-Net and trained for
  > 200 epochs.」（§4 Study Specific，第 6 页）
- **② 总步数：否（维持）**
- **③ 每图遍历次数：否** —— 🔴 **理由修正**：前序会话我按「nnU-Net 的一轮 = 250 次
  迭代」判否。**本次通读全文，该文从未给出这一定义**，故不得以此为据。
  改判否的实际理由是：该文是 3D **patch 级**主动学习（预算以 patch 数计，
  见附录表「Budget: Low [# Patches]」150/200/200/100），
  正文既未定义一轮对应多少次数据遍历，「每图遍历次数」在 patch 级设定下亦无定义。
- **④ 否（维持）**
- **⑤ 是：200 对 500 轮（维持，证据已落实）**
  > 「We evaluate the influence of the training length with three different
  > settings: training the model for 500 epochs (500 Epochs), training the model
  > for 200 epochs as in our main study (200 Epochs), and training the models for
  > 500 epochs on the entire query trajectories from the models trained with 200
  > epochs (Precomputed).」（§5 Training Length，第 12 页）
- **⑥ 否** —— 其训练时长是**全局**自变量，非跨预算的不等问题。
- 备注：该文的 "Precomputed" 对照设计值得一提 —— 它把「训练更久所以模型更好」
  与「训练更久所以查询更好」拆开。这与本文主张的精神一致，
  但针对的是训练时长的总体影响，不是跨预算的训练量不等。
  引用时**不得写成该文已处理本文所指的问题**。

### 本次新增

（下方逐篇填写，格式见模板）

<!--
#### [编号] 作者 年份，《标题》，venue
- 类别：A / B / C
- 全文来源：URL（arXiv / 期刊页 / 其它）
- 可获取性：✅ 全文可读 / ❌ 仅摘要 → 排除
- 纳入判据：是否有预算曲线
- ① 轮数：是/否 ——「正文原句」（第 X 节）
- ② 总步数：是/否 ——「正文原句」（第 X 节）
- ③ 遍历次数或收敛判据：是（遍历次数）/ 是（收敛判据）/ 否 ——「正文原句」（第 X 节）
- ④ N 与步数的算术关系：是/否
- ⑤ 量化训练量影响：是/否
- ⑥ 指出/控制跨预算训练量不等：是/否
- 备注：
-->

#### [A-1] Munjal 等 2022，《Towards Robust and Reproducible Active Learning Using Neural Networks》，CVPR 2022

- 类别：A（评估方法学 / reality-check）
- 全文来源：arXiv:2002.09564（v3, 2022-06-15），正式版 CVPR 2022 pp. 223–232，
  DOI 10.1109/CVPR52688.2022.00032。本地留档 `refs/munjal2022robust.pdf`，
  17 页 / 62730 字符，**含附录**（第 12–17 页为 Supplementary Section）。
  ⚠️ 时间窗：arXiv v1 在 2020 年，但**发表于 CVPR 2022**，按发表年计入 2021–2026 窗内。
- 可获取性：✅ 全文可读，抽取完整（无空白页告警）
- 纳入判据：✅ 有预算曲线 —— 横轴为 "% of Labeled Data"（10%→40%），
  纵轴测试精度（Fig. 1、Fig. 3、Fig. 5）
- **① 轮数：部分（仅 ImageNet，CIFAR 正文未给）**
  > 「ImageNet Training Details: ... We train the base classifier on $L_0$ for
  > 200 epochs (300 epochs when we include SWA and RA to the training pipeline)
  > ... For AL iterations we fine-tune the best model (picked by validation set
  > accuracy) from previous iteration for 100 epochs where lr = 0.01」（§5 Implementation Details）

  而 CIFAR-10/100 ——**该文的主实验**——正文明确不给：
  > 「For details on our training schedule we refer readers to the supplementary.」（§5 开头）
- **② 总步数：否** —— 正文未出现任何 iteration / step / update 的总数。
  正文出现的 "AL iterations" 指的是**查询轮次**（4 轮），不是梯度步数，不计入 ②。
- **③ 每图遍历次数可无歧义确定：部分**
  - ImageNet：**是（情形 b）**。「train ... on $L_0$ for 200 epochs」中的一轮
    即遍历一次标注集 $L_0$，故每图遍历次数 = 200（基模型）/ 100（各 AL 轮次）。
  - CIFAR（主实验）：**否**。正文不给训练时长，转附录。
    附录 Algorithm 1 第 12–14 行为
    > 「while convergence do / Train φ using only $L^i_0$ / end while」（Supp. §1.2）

    按本文件 §二 的规定，**附录不计入「正文是否报告」**，故 ③ 正文判否。
    且该判据不可操作：只写 "while convergence"，未说明如何判定收敛。
- **④ $N$ 与步数的算术关系：否** —— 正文无此关系式或等价陈述。
- **⑤ 量化训练量对结论的影响：否**（边界，见备注）
- **⑥ 指出/控制跨预算训练量不等：否**（边界，见备注）

**备注（三点，均为边界情形，计数时须能查到）**

1. 🔴 **⑥ 的边界**：该文确有一段与本文论点**相邻**的论述，但所指不同：
   > 「Fixing the hyper-parameters in AL iterations is questionable - with an
   > increase in AL iterations, the size of training data increases, and the
   > distribution changes ... Therefore, the hyper-parameters found to work well
   > in one AL iteration may not work well at further AL iterations.」（§4）

   它确实指出「随预算增大训练集规模变化，故不应沿用同一套配置」，
   但其所指是**超参数取值**（lr、wd），不是训练量；全文未提总步数或遍历次数
   随预算变化。故 ⑥ 判否，但**此篇是最接近的一篇，2.4 节引用计数时必须提到**，
   不得让读者以为该方向对此毫无察觉。
2. ⚠️ 其补救措施反而使步数**更**不可比：该文每个 AL 轮次用 AutoML 重搜超参数，
   搜索空间含
   > 「Batch Size: Categorical values from [8,16,32....1024]」（Supp. §1.3）

   批大小逐预算点变化 ⇒ 固定轮数下总步数 $E dot floor(N slash B)$ 随 $N$ 与 $B$
   双重变化。不过 CIFAR 侧「训至收敛」使该问题在该文中不影响结论
   —— 这正是本文建议的做法，须如实说明，**不得写成该文的缺陷**。
3. ⚠️ **⑤ 的边界**：该文在加入 SWA+RA 时把 ImageNet 轮数由 200 改为 300，
   并报告了精度差异，但该文将差异归因于**强正则化**，未将轮数作为自变量考察。
   故 ⑤ 判否。

#### [A-2] Beck 等 2021，《Effective Evaluation of Deep Active Learning on Image Classification Tasks》，arXiv:2106.15324

- 类别：A（评估方法学 / 统一重实现基准）
- 全文来源：arXiv:2106.15324（v3, 2021-11-02），DOI 10.48550/arXiv.2106.15324。
  本地留档 `refs/beck2021effective.pdf`，24 页 / 78352 字符（正文 10 页 + 附录）。
  ⚠️ **该文是预印本**，仅在 ICML 2021 "Subset Selection in Machine Learning" workshop
  报告过，无正式同行评议出处。计数时须单列为预印本，不得按会议论文计。
  纳入理由：该方向被引最多的统一重实现基准之一，工具包 DISTIL 被广泛沿用。
- 可获取性：✅ 全文可读，抽取完整
- 纳入判据：✅ 有预算曲线 —— 各 AL 轮次标注集规模 1000 → 25000 点对应测试精度
  （Fig. 1–6，Table 第 9 页）
- **① 轮数：是（上限 300）**
  > 「Each model is trained until either the training accuracy reaches 99%, the
  > epoch count reaches 300, or the training accuracy does not improve after 10
  > epochs.」（§4 Experimental Setup，第 6 页）

  ⚠️ 这里 300 是**上限**不是固定值 —— 实际轮数因预算点而异。这是收敛式协议的
  应有之义，不是缺陷。
- **② 总步数：否** —— 正文未给 iteration / step / update 总数。
  批大小（Opt. B. Size = 20 等）只出现在附录 A.3 的参数表，正文无。
- **③ 每图遍历次数可无歧义确定：是（情形 d，收敛判据）** —— 即上引同一句。
  该判据**完全可操作**：训练精度截断 0.99、耐心 10 轮、上限 300 轮，三者都给了数值，
  读者可照此复现。这是本次普查至今**正文给出可操作收敛判据的第一篇**。
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**（边界，见备注）
- **⑥ 指出/控制跨预算训练量不等：否 —— 但属「采取了措施、未说明用意」，单列**

**备注（三点）**

1. 🔴 **⑥ 归入「有措施、无说明」一类**（与 Lüth 2023 同类，但措施不同）。
   99% 训练精度截断 + 耐心 10 轮，实际效果是让**每个预算点都训到同一拟合水平**，
   从而消解了「固定轮数 ⇒ 总步数正比于 $N$」这一侧的偏差。
   但全文**未说明**这是为了跨预算可比 —— 该句只陈述协议，未给用意；
   文中讨论训练量的地方讲的都是**算力成本**，不是可比性：
   > 「The training time almost triples when we run for the entire data set. ...
   > we can increase the AL batch size, thereby reducing the number of training
   > rounds」（§5 Compute 讨论，第 10 页）

   故 ⑥ 判否，但计数时必须与「既无措施也无说明」区分开。
2. ⚠️ **⑤ 的边界**：该文系统考察了多项协议因素（数据增强、优化器 SGD/Adam、
   冗余度、每类样本数、AL 批大小、种子集初始化、**模型重置 vs 续训**），
   且确实观察到训练动力学的选择会改变结论：
   > 「Adam often results in reduced labeling efficiency and generalization
   > performance despite being faster to converge than SGD」（§1，第 3 页）

   但其自变量是优化器与正则化，**训练时长本身始终由同一套收敛判据决定**，
   未作为自变量考察。故 ⑤ 判否。
3. ⚠️ 「模型重置 vs 续训」一项与本文的训练量问题相邻但不同：
   > 「we interestingly observe that the resetting strategy as well as the
   > updating strategy converge to similar test accuracies at later rounds」（§4.6，第 9 页）

   该结论的前提正是**两者都训到收敛**；在固定轮数协议下未必成立。
   这一点本文 2.4 节可以引用，但须写成「该文的结论成立于其收敛式协议之下」，
   **不得写成该文验证了本文的论点**。

#### [B-1] Cai 等 2021，《Revisiting Superpixels for Active Learning in Semantic Segmentation with Realistic Annotation Costs》，CVPR 2021

- 类别：B（低标注预算语义分割）
- 全文来源：CVF Open Access（**该文无 arXiv 版**），CVPR 2021 pp. 10988–10997，
  DOI 10.1109/CVPR46437.2021.01084。本地留档 `refs/cai2021revisit.pdf`，
  10 页 / 44843 字符。首页核对：标题作者均符。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— 横轴标注成本（点击数，100k→400k clicks），
  纵轴 mIoU（Fig. 2a、2b）
- **① 轮数：否** —— 该文以迭代数而非轮数计训练时长，全文无 epoch 一词。
- **② 总步数：是**
  > 「For Cityscapes, we trained for a total of 60k iterations with a batch size
  > of 4 and input size of 769 × 769. ... For PASCAL VOC 2012, we trained for a
  > total of 30k iteration with a batch size of 12」（§4.1 Fully Supervised Baseline，第 5–6 页）

  且明确各预算点沿用同一配置：
  > 「The training hyper-parameters remain the same as the fully supervised
  > baseline.」（§4.1 Batch Training Details，第 6 页）
- **③ 每图遍历次数可无歧义确定：否** —— 理由见备注 2。
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否，既无措施说明也无提及）**

**备注（三点）**

1. 🔴 **这是「固定总步数」一侧的干净实例**：60k 迭代在全部四个预算点上不变。
   按本文的闭式，此协议下每图遍历次数 $approx S dot B slash N$ **反比于标注量**。
   该文未提及这一点。本文 2.4 节可据此举例（**不点名**，写成「该方向亦有工作
   在全部预算点上沿用同一总迭代数」）。
2. ⚠️ **③ 判否的理由必须写清，不得简化**：该文是**区域级**主动学习
   （超像素 / 矩形块），预算以点击数计，不以图像张数计。训练集中「图像张数」
   随预算如何变化，正文**未报告** —— 若训练时仍遍历全部 2975 张图而只在已标注
   区域上算损失，则每图遍历次数在各预算点恒为 $60000 times 4 slash 2975 approx 81$，
   不随预算变化；若只采样含已标注区域的图像，则该数随预算递减。
   正文不足以区分这两种实现，故判「无法无歧义确定」。
   🔴 **不得因为「固定总步数」就推定其遍历次数必然反比于预算** —— 区域级设定下
   这一推论需要额外前提，而该前提正文没给。
3. ⚠️ 该文的方法学主张与本文**同构但不同对象**：它指出成本度量选错
   （用标注像素百分比而非点击数）会低估超像素方法的效果 ——
   即「横轴画错」；本文指出的是「每个预算点训多久没定死」—— 即「纵轴怎么来的」。
   两者都属评估协议问题，2.4 节可并列提及，但**不得写成该文已涉及训练量问题**。

#### [B-2] Mittal 等 2023，《Best Practices in Active Learning for Semantic Segmentation》，DAGM GCPR 2023

- 类别：B（低标注预算语义分割；兼具 A 类的方法学性质）
- 全文来源：arXiv:2302.04075（v2, 2023-03-15），正式版 GCPR 2023，
  LNCS vol. 14264, pp. 427–442。本地留档 `refs/mittal2023best.pdf`，
  14 页 / 54194 字符（正文 + 附录）。首页核对：标题作者均符。
  ⚠️ 与表中 Mittal 等 2019 **不是同一篇**，仅第一作者相同，计数时须当作两篇。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— A2D2 上分 3 轮各选 1K 图，共 3K；按标注量报 mIoU
- **① 轮数：否（正文）／ 是（附录）** —— 这是本普查**最典型的一例**，
  正文明确把训练细节整体推给附录：
  > 「All other training details and hyperparameter information is included in
  > the Appendix.」（§5 Implementation details，第 5 页）

  附录才给出：
  > 「We utilize a polynomial learning rate scheduler with a batch size of 8 and
  > train a model in each AL cycle for 100 epochs.」（Appendix B. Training details，第 12 页）

  按本文件 §二 的规定，**附录不计入「正文是否报告」**，故 ① 正文判否。
  🔴 计数时此篇必须标注为「正文否 / 附录是」，**不得并入「完全未报告」**。
- **② 总步数：否**（正文与附录均无）
- **③ 每图遍历次数可无歧义确定：否（正文）／ 是（附录，情形 b）** ——
  附录给出「每个 AL 轮次训 100 epochs」，其一轮即遍历一次当前标注集，
  故每图遍历次数 = 100，**在各预算点上相同**。
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否）**

**备注（两点）**

1. 🔴 **这是「固定轮数」一侧的干净实例**，恰与 Cai 2021 相反：
   100 轮固定 ⇒ 每图遍历次数恒为 100（不随预算变），
   但总步数 $= 100 dot floor(N slash 8)$ **正比于标注量** ⇒ 低预算点欠训练。
   该文未提及这一点。
   ⚠️ 本条与 Cai 2021 合起来，是 2.4 节「两种通行做法各有一侧偏差、方向相反」
   这一陈述的**文献证据**。但两篇都**不点名**，正文写成「该方向两种做法并存」。
2. ⚠️ 该文确系方法学研究，系统考察了采集函数、半监督集成、**数据冗余度**等维度，
   与本文关切相邻。但其自变量不含训练时长，100 轮在全部实验中固定不变，
   故 ⑤ 判否。该文被引用时应写成「考察了采集与数据维度」，
   **不得写成「考察了训练协议」**。

#### [C-1] Shan 等 2024，《Edge-guided and Class-balanced Active Learning for Semantic Segmentation of Aerial Images》，arXiv:2405.18078

- 类别：C（遥感 / 航空影像的标注预算）
- 全文来源：arXiv:2405.18078（2024-05）。本地留档 `refs/shan2024edge.pdf`，
  15 页 / 77314 字符。首页核对：标题作者均符。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— 在 DeepGlobe / Potsdam / Vaihingen 上按已标注比例
  报告 mIoU，预算推进至全量的约 10%
- **① 轮数：是（每个 AL 轮次 50 轮）**
  > 「For each iteration, we randomly select 100 images (10,000 regions) from all
  > the unlabeled data and then select 500 regions according to our acquisition
  > strategy. After the acquisition, the model trained 50 epochs. Do this until
  > the annotated data reach a budget.」（§V 实验设置，第 8 页）
- **② 总步数：否** —— 正文无迭代总数，亦未给批大小。
- **③ 每图遍历次数可无歧义确定：否** —— 两处歧义叠加，见备注 1。
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否）**

**备注（两点）**

1. 🔴 **③ 判否的两处歧义，缺一不可**：
   (i) 该文每个 AL 轮次训 50 轮，而**到达各预算点所需的轮次数不同**
   （预算越大，累计经历的 AL 轮次越多），故累计训练量随预算增长；
   (ii) 正文**未说明每个 AL 轮次是重新初始化还是在上一轮权重上续训**
   —— 两种实现下累计遍历次数完全不同。
   🔴 该文的训练量**确实随预算单调增长**，但本文件**不得据此断言其增长的具体形式**
   —— 那需要知道 (ii) 的答案，而正文没给。写进 2.4 节时只能说「随预算增长」。
2. ⚠️ 与 Cai 2021 同为区域级 AL（每图 100 个区域），每图遍历次数这一概念本身
   在区域级设定下就需要重新定义。C 类论文多属此类，2.4 节陈述时
   **必须声明本文的口径是图像级预算**，不得把区域级工作的计数直接混入。

#### [C-2] Ma、Karakuş、Rosin 2025，《Integrating Semi-Supervised and Active Learning for Semantic Segmentation》，arXiv:2501.19227

- 类别：C（遥感：Vaihingen、RoadNet，兼 CityScapes）
- 全文来源：arXiv:2501.19227（v2, 2025）。本地留档 `refs/ma2025integrating.pdf`，
  32 页 / 66666 字符。首页核对：标题作者均符。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 纳入，但**是比较点不是曲线**，见备注 2
- **① 轮数：否** —— 全文无 epoch 计数。
- **② 总步数：否 —— 且这是一例值得专门记录的情形。**
  该文的学习率调度**显式依赖**总迭代数，但该数从未给出：
  > 「We train all our networks with a batch size of two and apply the "poly"
  > learning rate strategy and the initial learning rate is set to 0.001 and
  > multiplied by $(1 - "iter" slash "max-iter")^0.9$ at each iteration.」（§4.2 实现细节，第 13 页）

  `max-iter` 在全文中仅出现于该公式，**未给数值**。
  ⚠️ 这意味着训练时长不仅未报告，而且**学习率曲线的形状也无法复现** ——
  同一个 iter 在不同 max-iter 下对应的学习率不同（本研究自己也踩过这个坑：
  见 `docs/experiment-log.md` 关于 `target_steps` 探针的记录）。
  🔴 但写进 2.4 节时只写「总步数未报告」这一客观事实，
  **不得评论其可复现性** —— 那属于点名批评，违反纪律第 4 条。
- **③ 每图遍历次数可无歧义确定：否**
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否）**

**备注（两点）**

1. ⚠️ 该文的标注量是**自然形成**的，不是预设的预算网格：
   > 「We did not impose a limit on the use of labelled data, and the training
   > process gradually consumes some labelled data for the most uncertain area
   > in each iteration; by the end of the active learning training iterations,
   > 33% of the labelled data was automatically utilised」（§4.4，第 16 页）
2. 🔴 **纳入判据的边界，必须说明**：该文不画自己方法的预算曲线，
   而是作**跨标注量的点对点比较** —— 基线方法用 40% 标注，其方法用 33%（或 16%），
   据此主张以更少标注取得更高 mIoU。
   本文件判定：**纳入**，因为这正是「标注量—精度」关系的一种报告形式，
   且恰恰是训练量不等最可能影响结论的情形（不同标注量下训练时长未报告）。
   ⚠️ 但计数时须标为「点比较」而非「曲线」，与其余各篇分列。

#### [B-3] Kim 等 2023，《Adaptive Superpixel for Active Learning in Semantic Segmentation》，arXiv:2303.16817v2

- 类别：B（低标注预算语义分割）
- 全文来源：arXiv:2303.16817v2（2023-08-21）。本地留档 `refs/kim2023adaptive.pdf`，
  18 页 / 66013 字符（正文 + 附录 A–C）。首页核对：
  「Adaptive Superpixel for Active Learning in Semantic Segmentation /
  Hoyoung Kim, Minhyeon Oh, Sehyun Hwang, Suha Kwak, Jungseul Ok（POSTECH）」，符。
  ⚠️ **该 PDF 内无会议页眉，arXiv 元数据亦无 journal-ref**，
  故本文件**不声称其发表会议**，引用时只写 arXiv 版本号。
  （先前两次 406 未下到，本次用 `curl -sSL -C - --retry 3` 取得完整文件。）
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— Cityscapes 100k→250k clicks、PASCAL 10k→25k clicks，
  纵轴 mIoU（Fig. 3a、3b；附录 Fig. 9 延长至 400k）
- **① 轮数：否** —— 全文无 epoch 一词（`\bepoch` 零命中，全篇抽出 66013 字符，
  抽取质量正常，故此处的零命中可信）。
- **② 总步数：是**
  > 「For Cityscapes, we resize training images to 769 × 769 and train a model for
  > 60k iterations with a mini-batch size of 4. Similarly, for PASCAL, we resize
  > training images to 513 × 513 and train a model for 30k iterations with a
  > mini-batch size of 12.」（§4.1 Implementation details，第 5 页）

  且每一轮都**从 ImageNet 预训练权重重新初始化**，不是在上一轮权重上续训：
  > 「Analogously to the warm-up round, initializing model $theta$ at a model
  > pretrained on ImageNet, we obtain $theta_t$ trained to mainly minimize the
  > following CE loss」（§3.3，式 (10) 前，第 5 页）
- **③ 每图遍历次数可无歧义确定：否** —— 理由见备注 2。
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否）** ——
  按 §二 ⑥ 的细则补充，「固定 60k 迭代」属默认做法，不计为措施。
  ⚠️ 该文确有一处「使各条件可比」的安排，但对齐的是**方法之间**而非**预算之间**：
  > 「In the first round, we randomly select superpixels to train a model, ensuring
  > that all methods start at the same performance.」（§4.1 Evaluation protocol，第 6 页）

  这与本文关切相邻而不同，不得据此判 ⑥ 为「是」。

**备注（三点）**

1. 🔴 **与 Cai 2021 不是两个独立实例，计数时必须声明。** 该文的训练配置
   （Cityscapes 60k 迭代 / 批 4，PASCAL 30k 迭代 / 批 12）与 Cai 2021 **逐项相同**，
   且该文明言在预算口径上沿用 SP[4]（即 Cai 2021）：
   > 「Following SP [4], we use the number of clicks as the labeling budget.」（§4.1，第 6 页）

   ⚠️ 该文**没有**明说训练时长也沿用 SP，故「沿用」是本文件从数值一致作出的**推断**，
   按此标注。无论是否沿用，结论都一样：**「固定总步数侧有 2 篇」这句话不能写成
   「两项独立工作」**，只能写成「该协议在这一支工作中被延续使用」。
2. ⚠️ **③ 判否的理由与 Cai 2021 不同，须分别写清。** 该文的训练集 $D_t$ 由
   **含有已查询超像素的图像**构成（式 (9)），即图像张数**确随预算增长**，
   不像 Cai 2021 那样连增不增都无法判断。但正文**从未给出各预算点上 $D_t$ 的
   图像张数**，故每图遍历次数 $approx 60000 times 4 slash |D_t|$ 仍无法算出。
   🔴 这里有一个诱人但不可写的推断：Cityscapes 共 2975 张图，最低预算点
   已有 100k 次点击，按平均 256 像素的超像素算，**很可能**每张图都被覆盖到，
   于是 $|D_t|$ 在各预算点恒为 2975、遍历次数恒为约 81 次。
   **本文件不采信这一推断** —— 它要求假设超像素在图像间近似均匀分布，
   而正文没给覆盖率。写进 2.4 节时只能说「该量未报告」。
3. ⚠️ 与 Cai 2021、Shan 2024 同属**区域级**预算（点击数 / 超像素），
   「每图遍历次数」在该口径下本就需要重新定义。
   2.4 节陈述时须重申本文口径为**图像级预算**。

#### [C-3] Lenczner 等 2022，《DIAL: Deep Interactive and Active Learning for Semantic Segmentation in Remote Sensing》，arXiv:2201.01047

- 类别：C（遥感：INRIA Aerial Image Labelling、AIRS、ISPRS Potsdam）
- 全文来源：arXiv:2201.01047。本地留档 `refs/lenczner2022dial.pdf`，
  14 页 / 62537 字符（正文 + 附录 A、B）。首页核对：标题作者均符。
  ⚠️ 该 PDF 的页眉是 LaTeX 模板占位符
  （「JOURNAL OF LATEX CLASS FILES, VOL. X, NO. X, DECEMBER 2021」），
  arXiv 元数据亦无 journal-ref，故本文件**不声称其发表期刊**。
  （先前一次截断未下到，本次用 `curl -sSL -C -` 续传取得完整文件。）
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— Fig. 3「IoU evolution with respect to the number of
  annotated patches」（ISPRS / INRIA / AIRS / AIRS→ISPRS 四幅），
  Fig. 5、Fig. 6「Mean IoU w.r.t the number of annotations」，
  并以**随机顺序**为基线：
  > 「We compare the uncertainty-ordered sequences to a randomly-drawn one that
  > constitutes the baseline.」（§III.A.3，第 5 页）
- **① 轮数：否**（就预算曲线上的那个分割模型而言）——
  全篇唯一的 epoch 数是**辅助网络**的：
  > 「The ConfidNet auxiliary network is trained for 10 epochs per dataset with
  > Adam optimizer.」（§III.A.3，第 5 页）

  ConfidNet 只用于估计不确定度（即**选谁来标**），不是画预算曲线的那个模型，
  故不计入 ①。
- **② 总步数：部分** —— 增量阶段给了，初始模型没给。
  - 增量阶段（**随预算变化的那部分训练**）可算出：
    > 「For DISCA, during the interactive learning phase, we optimize the weights
    > using 10 stochastic gradient descent passes with a learning rate of 2e−6」（§III.A.2，第 5 页）

    配合「每个 patch 逐个重训」
    （「With DISCA, we retrain the network sequentially with each patch.」§III.A.3，第 5 页），
    预算为 $N$ 次标注时增量更新数 $= 10 N$。
  - **初始预训练网络的训练时长全文未报告**：正文只说
    > 「We propose to first train a neural network on the annotated database.」（§II 开头，第 2 页）

    以及用 LinkNet 架构（§III.A.2，第 5 页），**轮数、步数、批大小均无**。
    附录 A、B 亦无。
- **③ 每图遍历次数可无歧义确定：否** —— 见备注 2。
- **④ $N$ 与步数的算术关系：否** —— 正文从未把「总更新数 $= 10N$」写出来，
  也未提及它随预算增长。
- **⑤ 量化训练量对结论的影响：否** —— 10 次 SGD pass 在全部实验中固定不变。
- **⑥ 指出/控制跨预算训练量不等：否（纯否）**

**备注（三点）**

1. 🔴 **这是第三种协议形态，不属于「固定轮数 / 固定总步数」二分，必须单列。**
   该文的预算曲线不是「每个预算点各训一个模型」，而是**同一次序贯运行的轨迹快照**：
   标注逐个到来，每个标注给 10 次梯度更新，曲线上第 $N$ 点就是吃掉 $N$ 个标注之后的
   那个模型。
   其训练量与预算的关系是 $10 N$ —— **严格正比**，与「固定轮数」侧同向
   （低预算点训得少），但机制不同（前者是每轮重训，这里是增量累积）。
   2.4 节若要举「训练量随预算增长」的例子，可以把它与固定轮数侧并列，
   但**必须注明机制不同**，不得混为一谈。
2. ⚠️ **③ 判否的理由**：该文的预算单位是**点击 / patch**，不是图像张数；
   且增量重训只在该 patch 的已标注像素上算损失，**整幅图像根本不被完整遍历**。
   故「每图遍历次数」在该文的设定下无定义。
   ⚠️ 但有一点值得记下并与 Cai / Kim / Shan 区分开：在该文**自己的预算单位**下，
   「每单位被更新的次数」是**常数 10 且可从正文直接读出**。
   这说明区域级设定并不必然导致该量不可确定 —— 不可确定的是**图像级**的那个量。
   计数时仍判 ③ 否（本文口径为图像级），但 2.4 节不得写成「区域级工作一律报不出」。
3. ⚠️ 该文的核心主张是交互式精修与主动引导，**不是**标注预算下的训练协议；
   引用时写「该方向亦有工作以序贯增量方式画预算曲线」，不作任何严谨性评价。

#### [A-7] Hacohen 等 2022，《Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets》（TypiClust），arXiv:2202.02794v4

- 类别：A（主动学习的评估方法学；任务为图像分类，非分割 —— 见备注 3）
- 全文来源：arXiv:2202.02794v4。本地留档 `refs/hacohen2022typiclust.pdf`，
  21 页 / 82353 字符。首页核对：
  「Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets /
  Guy Hacohen, Avihu Dekel, Daphna Weinshall」，符。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— Fig. 1b、Fig. 4、Fig. 9 均以预算（标注样本数）为横轴，
  纵轴为相对随机基线的准确率差
- **① 轮数：否（正文）／ 部分（附录）** —— 正文把实现整体推给他人框架：
  > 「In (i) and (ii), we adopt the AL evaluation framework created by Munjal et al.
  > (2020), which implements several AL methods including all baselines used here
  > except BADGE.」（§4.1，第 7 页）

  附录只对两个数据集给出轮数，且是作为**修改说明**出现的：
  > 「TinyImageNet modifications. As training did not converge in the original
  > implementation over Tiny-ImageNet, we increased the number of epochs from 100
  > to 200 ... ImageNet hyper-parameters were identical to TinyImageNet except for
  > the number of epochs, which was set to 100 due to high computational cost」（Appendix F.2.1，第 19 页）

  **CIFAR 侧的轮数在正文与附录中均未给出**，只能回溯到 Munjal 2022 的框架
  —— 而本普查对该框架 CIFAR 侧的核对结果是「正文未给、附录为不可操作的
  `while convergence do`」（见 A-1）。
- **② 总步数：否（正文）／ 部分（附录，仅半监督分支）** ——
  附录 F.2.3 给出 FlexMatch 分支的 CIFAR-10/100 400k 迭代、TinyImageNet 1.1m 迭代。
  监督分支无步数。
- **③ 每图遍历次数可无歧义确定：否** —— 即便把附录计入也判否：
  附录给了 TinyImageNet 200 轮、ImageNet 100 轮，但该文**未定义「一轮」是遍历
  标注集还是遍历整个数据池**，而在低预算 AL 中二者相差好几个数量级。
  🔴 不得按「一般框架里一轮就是遍历标注集」推定 —— 那正是纪律第 1 条禁止的推测。
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否** —— 该文的自变量是**预算**，不是训练量。
- **⑥ 指出/控制跨预算训练量不等：否 —— 归「有措施、但陈述的用意是别的」一类，
  不计入纯否**（依据见备注 1）。此外附录另有一处**目前第二接近**的陈述
  （最接近的仍是 Munjal 2022，见 A-1）：
  > 「This ensured stable convergence and a more reliable evaluation in AL
  > experiments.」（Appendix F.2.1，第 19 页）

  ⚠️ 它把「训练是否收敛」与「AL 评估是否可靠」直接连在一起 —— 这与本文的论点同源。
  但它关切的是**绝对意义上的收敛**（原实现在 TinyImageNet 上根本没收敛），
  不是**跨预算点训练量不等**，且该陈述在附录。故判否。

**备注（三点）**

1. ⚠️ 附录另有一项措施：
   > 「Re-Initialize weights between iterations ... Therefore, we re-initialized the
   > weights between iterations.」（Appendix F.2.1，第 19 页）

   其陈述的用意是**低预算下损失尖峰导致的优化问题**，不是跨预算可比性。
   按 §二 ⑥ 细则补充，重新初始化属于使各预算点训练量不累积的做法，
   但**用意说的是别的**，故归「有措施、未说明（本问题的）用意」一类，计数单列。
2. 🔴 **一条诱人但绝不能写的推断。** 该文的核心结论是「策略优劣在低预算与高预算之间
   反转」（相变）；本文的论点是协议本身会制造一个**随预算变化**的偏差。
   两者在形式上相邻，于是很容易写成「该文观察到的相变可能部分由协议造成」。
   **不得这样写**，理由有二：(i) 本研究没有复现该文的任何实验，没有证据；
   (ii) 这属于对具体论文的严谨性评价，违反纪律第 4 条。
   2.4 节只能写一般性陈述：「结论依赖预算的研究，对每个预算点训多久尤其敏感。」
3. ⚠️ 任务是**图像分类**不是分割。纳入是因为它直接以预算为自变量、
   属 A 类评估方法学；但 2.4 节如要按任务分列计数，须把它与分割类分开报。

#### [A-8] Yehuda 等 2022，《Active Learning Through a Covering Lens》（ProbCover），arXiv:2205.11320v3

- 类别：A（同上，任务为图像分类）
- 全文来源：arXiv:2205.11320v3。本地留档 `refs/yehuda2022probcover.pdf`，
  19 页 / 57770 字符。首页核对：
  「Active Learning Through a Covering Lens / Ofer Yehuda, Avihu Dekel,
  Guy Hacohen, Daphna Weinshall」，符。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— Fig. 5 等以预算为横轴报告各策略准确率
- **① 轮数：否（正文与附录均无）** —— 🔴 **这是本普查里「训练时长被层层转手」
  最彻底的一例，值得单独记下。** 正文：
  > 「In frameworks (i) and (ii) we adopt the evaluation kit created by Munjal et al.
  > [33], in which we can compare multiple deep AL strategies in a principled way.」（§4.1，第 7 页）

  附录把它再转一手，转给另一篇论文和代码仓库：
  > 「We trained a ResNet18 on the labeled set, using the AL comparison framework
  > created by Munjal et al. [33], and following the protocol described in [19]
  > (see details in [19] and the shared code).」（Appendix C.2，第 17 页）

  （文中 [19] 即本表的 A-7 Hacohen 2022。）
  按 §二 的规定，**开源代码不计入「正文是否报告」**；
  而被指向的那两篇，本普查也已核过：A-7 的 CIFAR 侧没给，A-1 的 CIFAR 侧给的是
  不可操作的收敛判据。故此处判否不是「我没找到」，而是**顺着指引走到底也没有**。
- **② 总步数：否（正文）／ 部分（附录，仅半监督分支）** ——
  附录 C.4 给出 FlexMatch 分支 CIFAR 400k 迭代、TinyImageNet 1.1m 迭代。
- **③ 每图遍历次数可无歧义确定：否**
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否）**

**备注（一点）**

⚠️ 该文与 A-7 共享作者、框架与评估流程，**不是独立实例**。
与 Cai 2021 / Kim 2023 那一对同理：计数时各算一篇，
但「该做法有 N 篇为证」这类话**必须注明其中哪些出自同一支工作**。
目前 16 篇里共有两对这样的关系（Cai–Kim，Hacohen–Yehuda），
另有三篇（A-1、A-7、A-8）共用 Munjal 的那一套评估框架。

#### [B-4] Xie 等 2022，《Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation》（RIPU），arXiv:2111.12940v2

- 类别：B（低标注预算语义分割；领域自适应主动学习）
- 全文来源：arXiv:2111.12940v2。本地留档 `refs/xie2022ripu.pdf`，
  18 页 / 67682 字符（正文 + 附录）。首页核对：
  「Towards Fewer Annotations: ... / Binhui Xie, Longhui Yuan, Shuang Li,
  Chi Harold Liu, Xinjing Cheng」，符。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 纳入，但**是点比较不是曲线**（与 C-2 Ma 2025 同类，见备注 3）——
  两个标注预算档：区域式 2.2% / 5.0%，像素式每图 40 个像素；
  Table 4 在这两档上与 RAND 等基线逐一对比。
- **① 轮数：否** —— 全文无 epoch 一词。
- **② 总步数：是**
  > 「For all experiments, we train about 40K iterations with batch size of 2」（§4 Implementation details，第 5 页）

  ⚠️ 原文用的是「about 40K」，带约数。记录时照抄，不要写成精确值。
  附录确认各设置沿用同一配置：
  > 「With respect to training procedure, we keep in line with details of the main
  > paper, such as learning rate schedule, batch size, maximum iteration, and
  > input size etc.」（Appendix，第 14 页）
- **③ 每图遍历次数可无歧义确定：否 —— 但判否的理由与前面所有篇都不同，见备注 1。**
- **④ $N$ 与步数的算术关系：否**
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出/控制跨预算训练量不等：否（纯否）**

**备注（三点）**

1. 🔴 **这是本普查里第一篇「本文所述问题在其设定下根本不出现」的论文，
   必须如实记下，它是对本文论点的限制条件。**
   该文的目标图像**全部**参与每一次迭代（Algorithm 1 第 5 行：
   「Randomly select a batch source data and target and train the network」），
   标注预算只决定每张图里有多少像素带标签，**不决定有多少张图进训练集**。
   正文给出 Cityscapes「2,975 training images」（§4 Dataset，第 5 页）、
   40K 迭代、批大小 2，三者俱全。
   因此：**每张目标图被遍历的次数在各预算点上完全相同** ——
   既不随预算增长（不像固定轮数侧），也不随预算递减（不像固定总步数侧）。
   ⚠️ 2.4 节**不得**暗示所有低标注预算实验都受该偏差影响。
   正确的限定是：**该偏差出现在「训练集的图像张数随预算变化」的设定下**；
   在部分标注 / 区域级标注且全部图像始终参训的设定下，它不出现。
2. ⚠️ **③ 仍判否的原因是一个二义，与上一条不冲突**：正文说「batch size of 2」，
   但每次迭代同时取一个源域批和一个目标域批，**未说明这个 2 是两域各取 2 还是合计 2**。
   故每图遍历次数的**数值**在约 13 与约 27 次之间二义，判否。
   🔴 但「该数值不随预算变化」这一**性质**是无歧义的（见备注 1）。
   这两件事必须分开说：③ 问的是数值，⑥ 关心的是性质。
3. ⚠️ 纳入判据的边界同 C-2：该文不画预算曲线，只在 2–3 个预算档上作点比较。
   计数时标为「点比较」。

#### [C-4] Rückin 等 2023，《An Informative Path Planning Framework for Active Learning in UAV-based Semantic Mapping》，IEEE T-RO（arXiv:2302.03347v3）

🔴 **本篇是普查开始以来 ⑥ 的第一个「是」。它直接改变 2.4 节能写的措辞，见备注 1。**

- 类别：C（无人机 / 遥感的标注预算；且是**图像级**预算，与本文口径一致）
- 全文来源：arXiv:2302.03347v3。本地留档 `refs/rueckin2023ipp.pdf`，
  18 页 / 96441 字符。首页核对：页眉为
  「ACCEPTED FOR PUBLICATION IN IEEE TRANSACTIONS ON ROBOTICS, AUGUST 23, 2023」，
  标题与作者（Julius Rückin, Federico Magistri, Cyrill Stachniss, Marija Popović）均符。
  —— 这是本普查至今唯一一篇可从 PDF 内部确认发表期刊的 arXiv 稿。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线，且该文把这条曲线写成了评估方法本身：
  > 「In line with the standard in AL literature [...], our key evaluation metrics
  > assess semantic segmentation performance (dependent variable) over the number
  > of collected training images (independent variable).」（§V 评估指标，第 9 页）
- **① 轮数：否** —— 全文无 epoch 一词。
- **② 总步数：否** —— 全文无 iteration 计数。
- **③ 每图遍历次数可无歧义确定：否（落入判否情形三）** ——
  只说「训练至收敛」，而**全文没有任何可操作的判据**：
  无验证集、无 patience、无早停、无轮数上限（检索 validation / stopping /
  patience / plateau 全部零命中，全篇 96791 字符、抽取正常，故零命中可信）。
  > 「We re-train the model until convergence with batch size 8 and weight decay
  > $lambda = (1-p) slash 2N$ in Eq. (1), where $p = 0.5$ is the dropout
  > probability, and $N$ is the number of training images」（§V 训练流程，第 9 页）
- **④ $N$ 与步数的算术关系：否** ——
  ⚠️ 但它给出了**另一个**显式依赖 $N$ 的配置：权重衰减 $lambda = (1-p) slash 2N$。
  这说明该文清楚「$N$ 在各预算点上不同」，只是没有把这一依赖延伸到训练时长上。
  记录此点，但**不得写成该文给出了步数关系**。
- **⑤ 量化训练量对结论的影响：否**
- **⑥ 指出或控制跨预算训练量不等：是** —— 见备注 1。

**备注（三点）**

1. 🔴 **⑥ 判「是」的依据与它的两种读法，必须一并记下。** 原句是：
   > 「The model is pre-trained on the Cityscapes dataset [66] to start experiments
   > and training after each of the 10 subsequent data collection missions from the
   > same checkpoint. This **also avoids catastrophic forgetting and accumulating
   > train time**.」（§V 训练流程，第 9 页；着重为本文件所加）

   措施是**每个预算点都从同一个固定检查点重新开始训练**（而不是在上一次任务的
   权重上续训），陈述的用意之一明确就是避免训练量累积。这同时满足 §二 ⑥
   第二条的两个要件：有措施、且说明了用意。
   ⚠️ **另一种读法必须写明**：「accumulating train time」也可以理解成
   「省机时」。本文件不采这种读法，理由是从固定检查点重训**并不省机时**
   （反而更费），故该短语只能指模型侧累积的训练量。
   但这毕竟是一句顺带的话，不是专门的论证 —— 引用时应写成
   「有工作以『避免训练量累积』为由固定了各预算点的训练起点」，
   **不得写成该文系统处理了训练量不等问题**。
2. 🔴 **这一条直接改写 2.4 节的空白陈述。** 在此之前 16 篇中 ⑥ 全是否，
   很容易写成「该方向不处理这一问题」。现在**不能这么写**。
   可写的是：「该做法在所考察的文献中罕见（16 篇中 1 篇），
   且出现在机器人学方向而非主动学习评测方向。」
   —— 比例须等普查核满后按最终分母重算。
3. ⚠️ 值得注意的是：该文的预算单位是**图像张数**（「the number of collected
   training images」），与本文口径完全一致，而非区域级 / 点击级。
   即**在与本文口径相同的设定下，第一个处理了这个问题的实例出现了**。
   2.4 节提到这一点时须据实，不得把它归入「区域级因而不可比」而淡化。

#### [C-5] Rangnekar 等，《SpecAL: Towards Active Learning for Semantic Segmentation of Hyperspectral Imagery》

🔴🔴 **本篇是整份普查里最重要的一条。它明确指出了本文固定轮数侧的那个机制，
并为此作了专门的处理。2.4 节与第一章的新意边界都必须按它重写，见备注 1。**

- 类别：C（航空高光谱遥感语义分割；AeroRIT 数据集）
- 全文来源：作者自存版 `https://aneesh3108.github.io/papers/DDDAS_2022__SpecAL.pdf`，
  本地留档 `refs/rangnekar2024specal.pdf`，8 页 / 19223 字符。
  首页核对：「SpecAL: Towards Active Learning for Semantic Segmentation of
  Hyperspectral Imagery / Aneesh Rangnekar, Emmett Ientilucci, Christopher Kanan,
  and Matthew Hoffman / Rochester Institute of Technology」，符。
  ⚠️ **出处标注的限度**：该 PDF 内部**无会议或期刊页眉**（页眉是论文自己的短标题），
  文件名含「DDDAS_2022」。出版信息（Springer LNCS 13984，DDDAS 会议论文集，
  2024-02-27 在线）来自**出版方页面，不是 PDF 内部**。
  引用时按出版方页面写，并记住年份 2024 是在线发表年。
- 可获取性：✅ 全文可读
- 纳入判据：✅ 有预算曲线 —— Fig. 4「individual classes ... as a function of
  active learning (data querying) cycles」，Table 1 亦列出 21.5% / 29.5% / 100%
  三个利用率。
  ⚠️ **但该文没有随机选择基线**，也不比较不同采集策略：
  「We do not keep a preset data budget but instead strive to obtain performance
  comparable with the network trained on full data (502 images).」（§4.1，第 6 页）
  故它**不是策略比较类研究**，引用时不得与那一类混为一谈。
- **① 轮数：是（120）**
  > 「We use an initial learning rate of 1e−2, with drops of 0.1 at 50 and 90
  > epochs. We train all our networks for 120 epochs with standard cross-entropy
  > loss」（§4.1，第 6 页）
- **② 总步数：否** —— 全文**未给批大小**（`batch size` 零命中，全篇 19223 字符、
  8 页抽取正常，故零命中可信），因此总步数无法导出。
- **③ 每图遍历次数可无歧义确定：否** —— 🔴 **而且这是全普查后果最严重的一处二义，
  因为它恰好落在该文为本问题所做的那个处理上。** 详见备注 2。
- **④ $N$ 与步数的算术关系：部分** —— 它用**两个具体预算点的比值**把这个依赖说出来了，
  但没有给出一般关系式，也没有推广到曲线上的中间预算点：
  > 「The network sees 502 samples per epoch on the fully labeled dataset to learn
  > representations from scratch. Conversely, in the limited data regime, the
  > network only sees 108 samples per epoch; hence, we do not expect it to learn at
  > total capacity due to the **nearly 1/5th reduction of samples per epoch**.」
  > （§3.1 末至 §3.2 首，第 3–4 页；着重为本文件所加）

  🔴 因此 ④ 这一格**不能再报成「0 篇」**。正确口径见 §四。
- **⑤ 量化训练量对结论的影响：是 —— 但量化的是「对绝对性能」，不是「对策略比较结论」，
  须与另两篇分列。** Table 1 前两行是一次干净的受控对比：
  **同一预算点（21.50%）、只改训练量**，mIoU 由 51.70 升至 54.60（+2.90）：
  > 「An increase in the number of samples per epoch (5x data) results in an
  > improvement from a mIoU of 51.70 to 54.60」（§4.2，第 7 页）

  🔴 **方向与本文固定轮数侧的预测一致**：默认协议下低预算点确实处于欠训练状态，
  补足训练量即显著提升。这是普查中第一篇方向与本文同向的 ⑤。
  ⚠️ 但它只在**一个**预算点上做了这件事，没有沿预算扫一遍，
  故不能用来支持「失真幅度随预算递减」这一侧的任何说法。
- **⑥ 指出或控制跨预算训练量不等：是 —— 全普查中最明确的一例。**
  > 「Hence, our first significant adjustment is to increase the number of samples
  > seen per epoch during training in the limited data regime.」（§3.1，第 4 页）

  措施（在低预算下加长每轮的样本数）与用意（补偿每轮样本数随预算减少）
  都写在正文里，且用意正是本问题本身。无需做任何宽容解读即判「是」。

**备注（四点）**

1. 🔴🔴 **对本文新意边界的影响 —— 必须照此修改，不得回避。**
   在本篇之前，普查支持的说法是「该方向的报告惯例不提及跨预算训练量不等」。
   **本篇把这一说法推翻了**：它用三句话说清了固定轮数协议下低预算点欠训练的
   机制（每轮样本数随 $N$ 减少 → 低预算点学不到应有的容量），
   并为此专门改了训练安排。
   因此：
   - 2.4 节**不得**写「该方向未意识到这一问题」或任何同义表述；
   - 第一章**不得**出现「首次指出」类措辞（本就已被
     `uav-water-seg-novelty-preempted` 禁止，此处是更强的先例，纪律不变而证据加重）；
   - 可写的、且本普查支持的，是**本文相对该先例多出来的三件事**：
     (i) 该文只讲了固定轮数这**一侧**，未指出固定总步数侧方向相反，
         更未指出二者构成一个两侧闭式关系；
     (ii) 该文把它当作**提高自家数值的工程调整**（「our first significant
         adjustment」，与数据增强、批集成、置信度惩罚并列为三项调整之一），
         而不是**预算曲线可比性**的问题 —— 它从未讨论曲线上各预算点之间是否可比；
     (iii) 该文只在**一个**预算点上量化了幅度，且其处理规则对中间预算点未定义
         （见备注 2）。
   ⚠️ 这三条是**界定差异**，不是**批评该文**。按纪律第 4 条，2.4 节表述时
   不点名，写成「已有工作注意到固定轮数下低预算点的欠训练，并为此加长了训练，
   但未涉及另一侧，也未将其作为曲线可比性问题处理」。
2. 🔴 **③ 判否的理由：「5x data」这个规则本身没有定义。** Table 1 把
   21.50% 与 29.50% 两行**都**标为「5x data」。于是有两种读法，后果相反：
   - 读法甲「倍数固定为 5」⇒ 每轮样本数 $= 5N$，随预算增长；
     每图遍历次数 $= 5 times 120 = 600$，**各预算点相等**；总步数 $prop N$。
   - 读法乙「每轮样本数固定为约 540（= 5 × 108，对齐满标注的 502）」⇒
     每图遍历次数 $= 540 times 120 slash N$，**随预算递减**；总步数与预算无关。

   原文只说「increase the number of samples seen per epoch ... in the limited
   data regime」，**未说增到多少、也未说随预算如何调整**。
   两种读法在 21.5% 这一点上数值相同（都是 540），因此 Table 1 分辨不了它们；
   而它们对**曲线形状**的影响恰好相反。故 ③ 判否。
   ⚠️ **不得按「5x 多半是指 5 倍标注集」推定读法甲** —— 那正是纪律第 1 条禁止的事。
   ⚠️ 这一条很适合放进 2.4 节：**即便是明确意识到该问题并作了处理的工作，
   其处理规则仍未写到可复现的程度。** 这比「没人处理」是个更站得住、也更有分量的论点。
3. ⚠️ 该文另有一处与本文无关但不可忽略的设定：初始标注集是**空间上连续的一小块区域**
   （Fig. 3a，108 个相邻 patch），且**water 类在初始标注集中完全缺席**
   （「the water class is not present in the initial labeled set」，§3.1，第 3 页）。
   这与本研究记录的空间泄漏纪律是同一类问题的另一面（那里是切块随机划分导致泄漏，
   这里是按区域取样导致类别缺失）。**不要把它写成该文的缺陷** ——
   该文明确说明了这是有意的设计并给出了理由。
4. ⚠️ 该文第 7 页还报告了一次重复性检查：
   > 「further sampling 108 grid patches across random areas in the training set
   > ... the networks could reach similar performances in eight of the ten trials.」

   即 10 次试验中 8 次结果相近 —— 这是普查中少见的对**初始集随机性**的报告。
   与本文「5 个种子会产生假阳性」的记录是同一类关切，可在 2.4 节一并提及。

## 四、计数

**普查已完成：17 篇，达到 15–20 篇的目标区间；抽样框内无未结案条目。**

**已核 17 篇 / 因无法获取全文排除 0 篇。**
（排除 0 篇不是因为没遇到障碍，而是遇到的三次障碍都是**下载问题**
——两次大文件截断、一次 arXiv 406——均已用 `curl -sSL -C -` 续传解决。
按纪律第 2 条，命中即读，一篇未跳过。）

| # | 文献 | 类 | ① 轮数 | ② 总步数 | ③ 遍历次数 | ④ 算术关系 | ⑤ 量化训练量影响 | ⑥ 指出/控制跨预算不等 |
|---|---|---|---|---|---|---|---|---|
| A-1 | Munjal 2022 | A | 部分（仅 ImageNet） | 否 | 否（正文）/ 收敛判据在附录且不可操作 | 否 | 否 | 否（最接近，见条目） |
| A-2 | Beck 2021 | A | 是（上限 300） | 否 | **是（d 收敛判据）** | 否 | 否 | 否（有措施无说明） |
| A-3 | Mittal 2019 | A | 是（150） | 仅半监督分支 | **是（b）** | 否 | 否 | 否 |
| A-4 | Zhan 2022 | A | 否（正文）/ 是（附录） | 否 | 否（正文）/ 是（附录 b） | 否 | **是** | 否 |
| A-5 | Lüth 2023 | A | 否（正文）/ 是（附录） | 部分 | 否（正文）/ 是（附录 c） | 否 | 否 | 否（有措施无说明） |
| A-6 | Lüth 2025 | A | 是（200） | 否 | 否（patch 级无定义） | 否 | **是** | 否 |
| B-1 | Cai 2021 | B | 否 | **是（60k/30k 迭代）** | 否（区域级，图像数未报） | 否 | 否 | 否 |
| B-2 | Mittal 2023 | B | 否（正文）/ 是（附录 100） | 否 | 否（正文）/ 是（附录 b） | 否 | 否 | 否 |
| B-3 | Kim 2023 | B | 否 | **是（60k/30k 迭代）** | 否（区域级，$D_t$ 图像数未报） | 否 | 否 | 否 |
| C-1 | Shan 2024 | C | 是（每轮次 50） | 否 | 否（两处歧义） | 否 | 否 | 否 |
| C-2 | Ma 2025 | C | 否 | 否（max-iter 未给值） | 否 | 否 | 否 | 否 |
| C-3 | DIAL 2022 | C | 否（仅辅助网络 10） | 部分（增量 $10N$；初始网络未报） | 否（点击级，图像级无定义） | 否 | 否 | 否 |
| A-7 | TypiClust 2022 | A | 否（正文）/ 部分（附录，无 CIFAR） | 否（正文）/ 部分（附录，半监督分支） | 否 | 否 | 否 | 否（有措施、用意说的是别的） |
| A-8 | ProbCover 2022 | A | 否（正文与附录均无） | 否（正文）/ 部分（附录，半监督分支） | 否 | 否 | 否 | 否 |
| B-4 | RIPU 2022 | B | 否 | **是（约 40K 迭代）** | 否（批大小二义），但**性质无歧义：不随预算变** | 否 | 否 | 否 |
| C-4 | Rückin 2023 | C | 否 | 否 | 否（收敛判据不可操作） | 否（但 $lambda$ 显式依赖 $N$） | 否 | **是** |
| C-5 | SpecAL 2024 | C | 是（120） | 否（批大小未报） | 否（「5x data」规则未定义，见条目） | **部分（以两点比值「1/5」陈述）** | **是（对绝对性能）** | **是（最明确的一例）** |

**分布（17 篇）**

- **④ 给出 $N$ 与步数的算术关系：0 / 17 给出一般关系式，另 1 篇（SpecAL 2024）
  以两个具体预算点的比值把这一依赖说了出来**（「nearly 1/5th reduction of
  samples per epoch」）。
  🔴 **这一格不能再报成「无人提及」。** 可写的是
  「所考察的 17 篇中无一给出该依赖的一般关系式；其中 1 篇以两个预算点的比值指出了它，
  但未推广到曲线上的中间预算点」。
- **⑥ 指出或控制跨预算训练量不等：2 / 17 判「是」**
  （SpecAL 2024 最明确，见 C-5；Rückin 2023 见 C-4），
  另 **3 篇属「采取了措施但未说明（本问题的）用意」**
  （Beck 2021 的收敛判据、Lüth 2023 的上采样、TypiClust 2022 的轮间重新初始化），
  12 篇为纯否。
  🔴 **这三档必须分列**，不得合并成「17/17 未处理」或「15/17 未处理」。
  🔴🔴 **「该方向未意识到这一问题」「无人处理」「首次指出」这类措辞一律不可用。**
  SpecAL 2024 把固定轮数侧的机制讲清楚了并为此改了训练安排。
  本文相对它多出来的三点（单侧 vs 两侧、工程调整 vs 曲线可比性、
  一点量化 vs 沿预算量化）见 C-5 备注 1，那是**界定差异**而非批评。
  「措施」的边界见 §二 ⑥ 细则补充：固定轮数与固定总步数**不算**措施。
- **③ 每图遍历次数可从正文无歧义确定：2 / 17 判「是」**（Beck 2021 收敛判据、
  Mittal 2019 固定轮数），另 **3 篇正文否、附录是**（Zhan 2022、Lüth 2023、Mittal 2023），
  12 篇否。
  🔴 **因此「该方向普遍不报告每图被遍历次数」这句话按现有证据站不住**
  —— 若把附录计入，17 篇里有 5 篇可确定。修订 1 的判断得到数据支持。
  🔴 **取而代之的、本普查真正支持的论点**：
  **即便是明确意识到该问题并作了处理的那一篇（SpecAL 2024），
  其处理规则仍未写到可复现的程度** —— 「5x data」在曲线的中间预算点上
  有两种读法，而两种读法对每图遍历次数的依赖方向恰好相反（见 C-5 备注 2）。
  这比「没人处理」更站得住，也更有分量。
- **①「正文否、附录是（或部分是）」：4 / 17**（Zhan 2022、Lüth 2023、Mittal 2023、
  TypiClust 2022）。这一格本身是个可报告的现象（训练细节系统性地落在附录），
  但它**不是本文的论点**，宜在 2.4 节一笔带过。
  ⚠️ 其中 **ProbCover 2022 更进一步：附录把训练细节再转手给另一篇论文与代码仓库**，
  顺着指引走到底仍得不到轮数。按 §二，代码不计入「报告」。
- **⑤ 量化训练量对结论的影响：3 / 17，但须按量化对象分成两类，不可合并：**
  - **对策略比较结论的影响：2 篇**（Zhan 2022、Lüth 2025）。
    🔴 两篇方向一致，且都与「训练越久、选择优势越小」相反。见 A-4 备注。
    正文红线段落须由举一例改为举两例。
  - **对同一预算点绝对性能的影响：1 篇**（SpecAL 2024，mIoU 51.70 → 54.60）。
    🔴 **方向与本文固定轮数侧的预测一致**（低预算点确实欠训练），
    是普查中唯一与本文同向的证据；但它只做了一个预算点，
    **不能用来支持「失真幅度随预算递减」那一侧的任何说法**。
- **区域级 / 像素级 / 点击级等非图像级预算：5 / 17**
  （Cai 2021、Kim 2023、Shan 2024、DIAL 2022、RIPU 2022，另 Lüth 2025 为 3D patch 级）。
  🔴 2.4 节引用本表时**必须声明本文口径为图像级预算**，
  否则等于把口径不同的工作混进同一个分母。
- **🔴 本文所述偏差在其设定下结构性地不出现：1 / 17**（RIPU 2022）——
  全部图像在每个预算点都参训，预算只决定每图的标注密度，
  故每图遍历次数在各预算点相同。
  **2.4 节不得暗示所有低标注预算实验都受此影响。**
  正确的限定是：该偏差出现在**训练集图像张数随预算变化**的设定下。
- **无随机选择基线、因而不属策略比较类的：1 / 17**（SpecAL 2024）。
  引用其 ⑤⑥ 证据时不得把它归入策略比较类研究。
- **非独立实例（计数时各算一篇，但不得当作独立证据）：两对 + 一组**
  —— Cai 2021 / Kim 2023（配置逐项相同，后者明言沿用）；
  Hacohen 2022 / Yehuda 2022（同作者、同框架、同流程）；
  Munjal 2022 / TypiClust 2022 / ProbCover 2022 共用同一套评估框架。
  🔴 凡写「有 N 篇如此」，必须同时注明其中哪些出自同一支工作。

**三种协议形态的文献实例（供 2.4 节佐证「做法并存」，均不点名）**

- **训练量随预算增长**（低预算点欠训练）
  - 固定轮数：Mittal 2019（150）、Mittal 2023（100）、Shan 2024（每轮次 50）、
    Lüth 2025（200）、Zhan 2022（20/30/40）、SpecAL 2024（120）
  - 序贯增量（机制不同，须分别写）：DIAL 2022（每标注 10 次 SGD ⇒ 共 $10N$）
- **固定轮数 + 为本问题加长低预算侧的每轮样本数**：SpecAL 2024（「5x data」）。
  🔴 这是唯一一篇为此作了处理且说明了用意的，**单列**；
  但其规则在中间预算点上未定义（C-5 备注 2），故不能算作一个可照用的协议。
- **固定总步数**（每图遍历次数随预算递减）：Cai 2021（60k / 30k 迭代）、
  Kim 2023（同为 60k / 30k）。
  🔴 **这两篇不是独立实例**：配置逐项相同，后者明言在预算口径上沿用前者。
  写进正文时只能算作「该协议在这一支工作中被延续使用」，**不得报成 2 篇独立证据**。
- **固定总步数且图像集不变**（训练量与预算无关，偏差不出现）：RIPU 2022。
  这一形态与上一条形式相同而后果相反，**必须分列**。
- **收敛式**（本文建议的做法）：Beck 2021、Munjal 2022（CIFAR 侧，判据不可操作）、
  Rückin 2023（判据同样不可操作，但另以固定起点检查点控制了跨预算训练量）
- **训练时长整体未报告、亦未转手可追**：Ma 2025（max-iter 未给值）、
  ProbCover 2022（转手至他文与代码）

**抽样框已清空：无未结案条目。**
原清单中的 RIPU、TypiClust、ProbCover、Rückin、SpecAL 均已核完
（见 B-4、A-7、A-8、C-4、C-5）。SpecAL 在 arXiv 上没有预印本，
故 arXiv API 检索一直未命中；最终由作者主页自存版取得全文。
🔴 若日后扩充样本，**新增的论文必须按同一判据重核并记入 §五**，
不得只添对本文有利的条目。

## 五、规则修订记录

### 修订 1（2026-10-05，新增核对 0 篇时，即第一篇 Munjal 核到一半时）

**改了什么**：③ 由「报告每图遍历次数**或**收敛判据」改为
「每图遍历次数可从正文**无歧义确定**」，并写死 (a)–(d) 四种判「是」情形与
三种判「否」情形；新增 ⑥「是否指出或控制跨预算训练量不等」。

**为什么改**：原 ③ 是**自相矛盾**的。若一篇论文的「一轮 = 遍历一次当前标注集」
且轮数固定 $E$，那么每图遍历次数**恒等于** $E$ ——
报了轮数就等于报了遍历次数。按原判据，这类论文 ③ 必须判「是」，
于是 2.4 节计划写的「该方向普遍不报告每图被遍历次数」就是一句**失实的经验断言**。

这个矛盾不是措辞问题，是**论点落点选错了**。本文真正的发现不是
「这个数读不出来」，而是「不同预算点上的训练量**本来就不相等**，
而这一点在报告惯例中不被提及」。前者是可算的算术，后者才是空白。
故把空白陈述整体迁到新增的 ⑥ 上。

**改动前已核几篇**：新增核对 **0 篇**（本次修订发生在第一篇 Munjal 2022
核对过程中，其记录按修订后的判据填写）。

🔴 **对既有 4 篇的影响（尚未处理，不得忽略）**：论文 2.4 节
`tab:protocol-audit` 现有四篇的 ③ 列**全部填「否」**，这是按旧判据填的。
按新判据需逐篇复核：

- Lüth 2023：标注集**上采样至固定规模**后按轮训练 → ③ 仍判「否」（落入判否情形二），
  但 ⑥ 属「采取了措施但未说明用意」，须单列；
- Lüth 2025：基于 nnU-Net，其「一轮」是固定 250 次迭代而非一遍数据 →
  ③ 仍判「否」（落入判否情形一）。**但此点须回原文核实，不得按框架惯例推定**；
- Mittal 2019、Zhan 2022：③ 需**重新取证**。若其「一轮 = 一遍标注集」且轮数固定，
  按新判据应改判「是」。在取得全文复核之前，这两格**不得保留旧判定**。

在四篇复核完成前，2.4 节表格与正文不得定稿。

### 修订 2（2026-10-05，已核 10 篇、核第 11 篇 Kim 2023 时）

**改了什么**：只在 §二 ⑥ 下加了一段「什么才算措施」的细则补充，
把**固定轮数**与**固定总步数**这两种默认做法明确排除在「措施」之外，
并把「措施」限定为「非默认的、额外的、使训练量随预算的变化消失或变得无关」的安排。
判「是」「否」的主条文一字未动。

**为什么改**：核 Kim 2023 时遇到判定分叉 —— 它固定 60k 迭代且每轮从 ImageNet 重新
初始化，于是「总步数在各预算点完全相等」。若把这算成「采取了措施但未说明用意」，
那么 Cai 2021（同样 60k）就该一并改判，而对称地，固定轮数的 Mittal 2019 /
Mittal 2023 / Shan 2024 也「使每图遍历次数相等」，同样该改判。
推到底，**12 篇里没有一篇会留在「纯否」**，这一子类就失去了区分力，
而真实情况是这些论文只是沿用了默认配置，并没有针对本问题做任何处理。

把边界写死在「非默认的额外安排」上，才能让「纯否」这个数字保持它该有的含义。

**改动前已核几篇**：10 篇。**对既有判定的影响：0 处改判** ——
已逐条复查这 10 篇的 ⑥ 列：Beck 2021（收敛判据）与 Lüth 2023（上采样至 5500）
仍属「有措施无说明」，其余 8 篇仍为纯否，Cai 2021 的「纯否」在新边界下得到确认
（此前是按同样的理解填的，本次只是把理由写明）。

## 六、普查结果对正文的改动清单（2026-10-05，普查核满 17 篇后写定）

本节是从 §三、§四 导出的**待办**，逐条对应正文里一句具体的话。
🔴 改完一条划掉一条；未划掉的不得认为已处理。

**A. 必须删改的措辞（不改即构成失实陈述）**

1. 🔴 2.4 节现有空白陈述「该方向普遍不报告每图被遍历次数」——**整句删除**。
   依据：修订 1（该陈述与「报了轮数即报了遍历次数」自相矛盾）
   + §四（计入附录后 17 篇有 5 篇可确定）。
2. 🔴 凡「未有工作指出 / 未被意识到 / 首次指出」类措辞——**全部删除**，
   中英文稿都查。依据：C-5（SpecAL 2024 明确指出了固定轮数侧的机制并为此改了训练）。
3. 🔴 ④ 这一格不得报成「无人提及」。改为「无一给出一般关系式；其中 1 篇以
   两个预算点的比值指出了它」。
4. 🔴 不得暗示所有低标注预算实验都受此偏差影响。须加限定：
   该偏差出现在**训练集图像张数随预算变化**的设定下；
   在全部图像始终参训、预算只决定每图标注密度的设定下不出现（B-4）。

**B. 必须新增的内容**

5. 替代论点（这是普查真正支持的那一句，用它顶掉第 1 条删掉的位置）：
   **报告惯例使各预算点的训练量无法核对——即便是唯一为此作了处理的那篇，
   其规则在曲线的中间预算点上仍有两种方向相反的读法。**（C-5 备注 2）
6. 声明本文预算单位为**图像级**，并说明 17 篇中 5 篇为区域/像素/点击级，不同口径不混算。
7. `#honest` 反向证据块由**一例改两例**（Zhan 2022 §3.3 + Lüth 2025），
   并把 SpecAL 的同向证据单列、注明只做了一个预算点。
8. `tab:protocol-audit` 表身替换为 §四 的 17 行，新增 ⑥ 列；
   并修正旧表的两处已知错判（Mittal 2019 ③ 否→是；
   Zhan 2022 / Lüth 2023 / Mittal 2023 拆为「正文否／附录是」）。
9. 与 SpecAL 的差异写成三点、不点名（C-5 备注 1）：单侧 vs 两侧、
   工程调整 vs 曲线可比性、一个预算点 vs 沿预算量化。
10. 「非独立实例」的说明：凡写「有 N 篇如此」须注明其中哪些出自同一支工作
    （Cai–Kim、Hacohen–Yehuda、以及三篇共用 Munjal 框架）。

**C. 两稿同步**

11. 以上每一条都要在 `paper/paper-zh.typ` 与 `paper/paper-en.typ` 中各做一遍。
    英文稿不得只译不改——它现在带着同样的两处错判。
12. 改完删掉 2.4 节的 `#todo`，并把 §四 的计数写进正文（带分母）。

### 执行情况（2026-10-05 结）

**1–12 全部完成，中英文稿各做了一遍。** 两稿均编译通过（中 19 页 / 英 21 页），
表身逐行核对过渲染结果，17 行 + 本文行齐全。

⚠️ **第 9 条有一处有意偏离，须记录。** 该条原写「不点名」，但实际写成了
**点名的中性引用**：2.4 节直接写出 Rangnekar 等及其原句，并在三点差异后加
「以上三点是界定差异，不是对该文的评价」。理由：这一段的作用是**把先例的功劳
归给它**，匿名化反而会让读者以为本文在回避；红线禁止的是**不点名批评**，
而致信先例与中性引用不在其列（见 §五 措辞纪律）。
2.4 节中凡涉及**报告惯例的批评**处，仍一律写成「该方向的报告惯例」，未点名。

⚠️ 英文稿的表头用 `C1–C6` 而非 `①–⑥`：英文稿字体仅 Times New Roman，
带圈数字很可能缺字形。两稿列定义逐字对应，核对时按序号对齐即可。
