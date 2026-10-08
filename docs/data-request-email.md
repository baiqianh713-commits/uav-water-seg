# RivAIr 数据集申请邮件（草稿）

> ⚠️ **发送前必读**
> 1. **必须先与导师确认**，并按惯例把导师加为抄送或共同署名请求人 ——
>    以学生个人身份申请，回复率明显低于有导师背书的请求。
> 2. 方括号 `[...]` 内均为占位符，**必须替换成你的真实信息**，
>    不要漏掉任何一处就发出去。
> 3. 用学校邮箱发送（`.edu.cn` 结尾），个人邮箱容易被当作垃圾邮件。
> 4. 全文不要出现任何夸大或不实陈述（如虚构已发表成果）。
>    学术数据申请的信任一旦破坏无法修复。

---

## 收件人

| 项目 | 内容 |
|---|---|
| **To** | marco.lasalandra@uniba.it （Marco La Salandra，通信作者） |
| **Cc** | 你的导师邮箱 |
| **可选 Cc** | Domenico Capolongo（末位作者，University of Bari）—— 邮箱需在论文或学校主页查证，我未核实，不要凭猜测填写 |

## 主题

```
Data request for annotation-efficiency research — RivAIr dataset (doi:10.1016/j.jag.2025.104720)
```

---

## 正文

```
Dear Dr. La Salandra,

I am [你的姓名], a [硕士研究生 / 博士研究生 / 本科生] at [学校全称], China,
working under the supervision of [导师姓名和职称] on UAV-based remote sensing
for hydraulic engineering applications.

I read your recent paper with great interest:

  La Salandra, M., Colacicco, R., Panza, S., Fumai, G., Dellino, P., &
  Capolongo, D. (2025). RivAIr: A custom-designed UAV-based sensor for
  real-time water area segmentation and surface velocity estimation.
  International Journal of Applied Earth Observation and Geoinformation,
  142, 104720.

The system's on-board integration of YOLOv8m-seg with mask-constrained
Farneback optical flow is a substantial step toward operational river
monitoring, and the six-year multi-condition dataset along the Basento
River is, to my knowledge, unusually well documented for this task.

I am writing because the paper notes that the dataset is available upon
request for legitimate research purposes, and I would like to ask whether
it would be possible to obtain access.

**What I am working on**

My research addresses annotation efficiency for UAV water body
segmentation. Specifically: given a fixed annotation budget of N images
drawn from a large pool of unlabeled UAV frames, which frames should be
annotated in order to maximise segmentation accuracy? My approach combines
redundancy-aware selection (to avoid annotating near-duplicate frames)
with water-specific difficulty priors (specular glare, shadow confusion,
and texture flatness).

Your work is directly relevant to this question in two ways. First, your
paper reports that 7,630 images were annotated using a SAM-assisted,
expert-corrected workflow, and notes that this hybrid approach was adopted
precisely because fully manual annotation would be prohibitively
time-consuming. Second, the future direction outlined in Section 4.4 —
multiple condition-specific YOLOv8 models, each trained on a dataset
representative of a distinct flow regime and weather condition — implies
that the annotation requirement scales with the number of conditions
considered. My work aims to reduce exactly this cost.

I want to be explicit that my research is **not** an attempt to improve on
RivAIr or to compete with it. It concerns which images to annotate, and is
orthogonal to the segmentation architecture and the on-board processing
pipeline. If anything, a positive result would be directly useful to your
roadmap: it would indicate how many images per condition are actually
needed, which may also help with the 753-minute training time you identify
as a bottleneck in Section 4.1.

**Why your dataset in particular**

I have run preliminary experiments on public satellite land-cover data
(LoveDA), and the results were inconclusive — differences between selection
strategies were smaller than the variance across random seeds. The reason
appears to be structural: LoveDA consists of spatially independent scenes
from different cities, so there is little inter-frame redundancy for a
redundancy-aware method to exploit, and no specular glare (the glare
feature was essentially inactive on that data, which is expected for nadir
satellite imagery).

Your dataset has the opposite structure: repeated surveys of the same
3 km reach over six years, at ~1 cm GSD, under documented flow regimes and
weather conditions. That combination of high inter-frame redundancy and
explicit condition stratification (Table 1) is what makes it well suited —
and, as far as I have been able to find, close to unique — for this
question.

**What I am asking for**

If sharing the full dataset (~76 GB) is impractical, a subset would be very
helpful and probably sufficient, for example:
  - a few hundred images per flight campaign, preserving the six
    flow-regime / weather conditions in Table 1, with the corresponding
    labels; or
  - downsampled imagery, if full resolution is a constraint on transfer.

I am happy to work with whatever format and transfer method is most
convenient (institutional file transfer, cloud storage, or physical media
at my own cost).

**Commitments**

  - The data would be used solely for non-commercial academic research.
  - I would cite the RivAIr paper as the data source in any resulting
    publication, and follow any citation or acknowledgement wording you
    prefer.
  - I would not redistribute the data to third parties.
  - I would share my frame-selection code and results with you, including
    the per-condition budget findings, whether or not they turn out to be
    positive.
  - I would be glad to send you a draft before submission if you would
    like to review how your data and work are described.

I would also like to ask two questions, if you have the time:

  1. Would you be willing to be acknowledged, or to comment on a draft, if
     the work reaches publication? I want to be sure your dataset and
     contribution are represented accurately.
  2. Are there any conditions of use, embargo periods, or co-authorship
     expectations I should be aware of before proceeding?

Thank you for considering this request, and for making the RivAIr work
openly accessible. I understand that preparing and transferring data takes
time, and I am grateful for any assistance you are able to offer.

Best regards,

[你的姓名]
[学位项目 / 年级]
[院系全称]
[学校全称]
[学校邮箱]
[可选：ORCID 或个人主页]

Supervisor: [导师姓名和职称], [导师邮箱]
```

---

## 撰写说明（为什么这样写）

| 段落 | 作用 |
|---|---|
| 开头引用完整文献 | 表明确实读过，不是群发模板 |
| 提到具体技术细节（mask-constrained optical flow） | 同上，且显示专业性 |
| 明确「不竞争」 | 消除最主要的顾虑 —— 作者最怕数据被用来"打败"自己 |
| 引用其 §4.4 与 §4.1 | 证明你的工作服务于**他们自己写下的**未来方向 |
| 主动讲 LoveDA 失败 | **诚实反而增强可信度**，同时说明为何非他们数据不可 |
| 提出可接受子集 | 降低对方成本，大幅提高同意概率 |
| 承诺共享代码和结果 | 提供对等价值，不是单方索取 |
| 主动提出致谢/共同作者问题 | 表明尊重，也避免后续纠纷 |
| 询问使用条件 | 显示合规意识 |

## 后续跟进

- **两周**无回复：发一封简短礼貌的跟进邮件（不要催促语气）
- **一个月**无回复：转向退路数据集，不要反复追问
- 若对方要求共同作者：与导师商议，这在数据共享中是常见且合理的诉求
- 若对方同意：立刻确认数据的**许可条款**能否支持你在论文中发表基于它的结果

## 同时启动的退路

不要只等这一条线（沟通周期可能数周）：

1. **LKF-DCANet 东风渠数据集** —— 百度网盘已公开，提取码 `2025`，可立即下载评估
2. **FloodNet** —— 2,343 张无人机影像，公开
3. **自采数据** —— 无人机手续现在就该办（实名登记 + 操控员执照 + 飞行活动申请）
