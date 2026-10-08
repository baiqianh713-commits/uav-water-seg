# 低标注无人机水体分割：跨平台迁移与标注帧选择

面向 SCI 论文的实验代码库。研究问题：**给定只能标注 N 张图的预算，标哪 N 张？**

---

## 一、这个项目在回答什么问题

无人机航拍产出的是视频，一段 5 分钟航拍就是约 9000 帧，相邻帧高度冗余。
随机抽 200 帧标注，信息量可能还不如精选 20 帧。

所以本文的核心不是"少量标注能达到多少精度"（这是消融实验，撑不起一篇论文），
而是**在固定标注预算下如何选出信息量最大的帧**。

目标结论形态：*精选 N 张标注图达到随机 10N 张的精度，标注成本降低约 90%。*

### 选题查重结论（2026-09 核查）

完整论述见 **[`docs/related-work-differentiation.md`](docs/related-work-differentiation.md)**。

两篇需要重点区分的文献：

| 文献 | 内容 | 场景重合 | 贡献重合 |
|---|---|---|---|
| Chen & Tong, `arXiv:2507.10084` (IJRS) | 两阶段迁移 + SegFormer，GF-7→GF-2，水体 IoU 25.50%→64.84% | **低**（纯卫星，且刻意保持同分辨率） | 低 |
| Liao et al., `Appl. Sci. 2025, 15, 6292` (LKF-DCANet) | 轻量化架构(0.22M) + 知识蒸馏，迁移至四川东风渠灌区无人机数据，10% 标注 | **高** | 低 |

**差异化定位**：两篇都在回答"**怎么训**"（训练方法），且标注子集均为随机划分、
标注量固定在单一操作点；本文回答"**标哪些**"（数据选择）。二者正交可叠加，
本文的选帧策略可直接接在它们的 pipeline 前端。

四个可辩护的研究空白：① 无人研究"标哪些"；② 无标注预算曲线与饱和点；
③ 真正的跨平台尺度间隙（≈20× GSD 差）未被量化；④ 小样本实验缺乏方差报告。

---

## 二、贡献点与对应代码

| # | 贡献 | 代码位置 |
|---|---|---|
| 1 | 卫星→无人机跨平台标注预算基准 | `src/sweep.py` |
| 2 | **水体专属的标注帧选择策略**（核心创新） | `src/selection.py` |
| 3 | 自采无人机数据集 + 代码开源 | 待采集 |

⚠️ **创新性边界要诚实**：k-center greedy、分层采样都是已有的通用主动学习方法，
不能声称原创。可辩护的创新在于**水体专属难度先验**（镜面反光/阴影混淆/纹理平坦）
及其与冗余感知的几何加权组合方式，见 `selection.py` 的 `kcenter_greedy`。

---

## 三、快速开始

### 0. 当前进展

真实数据 baseline 已跑通，结果见 **[`docs/experiment-log.md`](docs/experiment-log.md)**：

- LoveDA 子集（268 图块，场景级无泄漏划分）：**test IoU 0.7687**
- 选帧策略对比（8 组）：**方差压倒效应，尚未能证明 hybrid 优于 random**
  —— 种子间标准差 0.17–0.26，策略间差异仅 0.025

诊断结论：LoveDA 是跨城市独立影像**而非视频**，冗余度低，冗余感知无用武之地。
必须换到高冗余的无人机数据（FloodNet / 东风渠数据集）才能验证方法。

```bash
# 复现上述 baseline（约 20 分钟 CPU）
python -m src.download_loveda --out data/raw/loveda_hf --n_train 150 --n_val 50
python -m src.prepare_data inspect --labels_dir data/raw/loveda_hf/train/masks --preset loveda
python -m src.prepare_data generic --src data/raw/loveda_hf --out data/processed/loveda_small \
    --splits train --tile 512 --water_ids 4
mv data/processed/loveda_small/train data/processed/loveda_small/_all
python -m src.make_splits --root data/processed/loveda_small --from_split _all --ratios 0.6 0.2 0.2
python -m src.train --config configs/loveda_small.yaml --strategy all --seed 0
```

### 1. 本地冒烟测试（不需要 GPU，几十秒）

**租云 GPU 之前一定先跑这一步**，确认流程无误，避免开着计费的卡调 bug。

```bash
# 生成合成数据（模拟了水体的三个真实难点：无纹理、镜面反光、蓝绿色调）
python -m src.prepare_data synthetic --out data/processed/synthetic --size 128

# 跑一次完整训练
python -m src.train --config configs/smoke.yaml --budget 12 --strategy hybrid --seed 0
```

已验证：流程跑通，且模型确实能学（合成数据上 40 epoch 后 test IoU 0.996）。
**合成数据的任何数字都不能用于论文**，它只验证代码正确性。

### 2. 准备真实数据

```bash
# 第一步永远是核对标签索引！索引错了实验全错，且很难从指标上发现
python -m src.prepare_data inspect --labels_dir data/raw/LoveDA/Train/Rural/masks_png --preset loveda

# 确认无误后再转换
python -m src.prepare_data loveda   --src data/raw/LoveDA   --out data/processed/loveda   --tile 512 --water_ids 4
python -m src.prepare_data floodnet --src data/raw/FloodNet --out data/processed/floodnet --tile 512 --water_ids 5 8
```

预处理是纯 CPU 任务，**在本地做完再上传处理好的数据**，不要浪费 GPU 计费时间。

### 3. 正式实验

```bash
# 先估时间和费用
python -m src.sweep --config configs/floodnet.yaml --dry_run

# 跑预算曲线
python -m src.sweep --config configs/floodnet.yaml \
    --budgets 5 10 20 50 100 200 \
    --strategies random uniform kcenter hybrid \
    --seeds 0 1 2 --include_full

# 中断后重跑：已完成的自动跳过，未完成的从 last.pt 续
```

---

## 四、数据集

**源域（卫星，用于预训练）**

| 数据集 | 规模 | 分辨率 | 获取 |
|---|---|---|---|
| LoveDA | 5987 张 1024² | 0.3 m | https://github.com/Junjue-Wang/LoveDA |
| WHDLD | 4940 张 256² | 2 m | https://github.com/AngieNikki/openWUSU |

**目标域（无人机低空）**

| 数据集 | 规模 | 说明 |
|---|---|---|
| FloodNet | 2343 张 1024² | DJI Mavic Pro @200ft，GSD≈1.5cm，`arXiv:2012.02951` |
| RescueNet | 4494 张 | FloodNet 同团队，超高分辨率 |
| 自采数据 | 待采集 | 六旋翼 850mm + Pixhawk/PX4 |

两个源域数据集都不是水体专用，需自行二值化（water→1，其余→0）。
LoveDA 类别分布不均衡，注意在论文中报告水体像素占比。

---

## 五、实验设计

主对照表，缺任何一组都会被要求补实验：

| 组 | 设置 | 作用 |
|---|---|---|
| A | ImageNet → UAV 微调 | 基线下界 |
| B | ImageNet → 卫星 → UAV 微调 | 验证中间域有效性 |
| C | B + **随机**选 N 张 | **主要对手** |
| D | B + **本文策略**选 N 张 | 本文方法 |
| E | 全量标注 | 性能上界 |
| F | D + 伪标签自训练 | 加分项 |

### 两条不可违反的规则

1. **每个设置跑 3–5 个随机种子，报均值±标准差。**
   小样本实验方差极大，单次结果没有说服力，审稿人必问。
2. **测试集固定，且绝不参与选帧。**
   选帧只能用无标注信息。一旦用到测试集标签，整篇论文作废。

### 消融实验

- `hybrid_alpha` 扫 0 → 1（0 = 纯覆盖度，1 = 近似纯难度），验证两个准则都有贡献
- 四个难度权重（glare/shadow/flatness/bluegreen）的敏感性分析
- `strategy=difficulty`（只选最难的）预期表现差 —— 这组能证明"光选难的不够，必须同时保证覆盖度"

---

## 六、代码结构

```
configs/
  base.yaml        # 所有默认参数，其他配置继承它
  smoke.yaml       # 本地 CPU 冒烟测试
src/
  utils.py         # 配置加载(支持 _base_ 继承)、随机种子、日志
  data.py          # Dataset、数据增强、水体占比统计
  models.py        # 模型构建(tiny_unet/smp/segformer) + BCE-Dice 损失
  metrics.py       # IoU/F1/边界F-score，聚合式与逐图式两种口径
  selection.py     # 选帧策略 ← 核心创新
  prepare_data.py  # 数据集转换、标签核对、合成数据生成
  train.py         # 训练主脚本，支持断点续训
  sweep.py         # 批量实验驱动 + 预算曲线绘图
```

### 为租云 GPU 做的设计

- **配置驱动**：批量实验不用改代码
- **断点续训**：`--resume` 从 `last.pt` 继续，断连不丢进度
- **跳过已完成**：sweep 重跑时自动跳过有 `result.json` 的组
- **预处理分离**：CPU 活在本地做完再上传
- **默认小模型**：`mit-b0` / `tiny_unet` 起步，显存友好

---

## 七、技术路线

| 阶段 | 模型 | 何时用 |
|---|---|---|
| 入门 | `tiny_unet`（本仓库内置，零依赖） | 验证流程 |
| 基线 | `smp_unet` + ImageNet ResNet34 | 3 行建模型，全流程可读 |
| 主力 | `segformer` (mit-b0/b2) | 便于和 IJRS 那篇对比 |
| 标注加速 | SAM / SAM2 辅助 + X-AnyLabeling | 本身就是"降低标注成本"的实招，可写进方法章节 |

**不要一上来自己写网络结构。** 创新点在实验设计和选帧算法，不在网络结构上。

---

## 八、目标期刊

| 期刊 | 分区 | 备注 |
|---|---|---|
| Remote Sensing (MDPI) | Q1/Q2 | 首选，审稿快 |
| Drones (MDPI) | Q2 | 无人机角度 |
| IEEE JSTARS | Q1 | 方法要更硬 |
| Water / WRM | Q1/Q2 | 需强化水利落地论证 |

MDPI 系有版面费（约 2000+ CHF），提前和导师确认经费。

---

## 九、待办

- [ ] 精读两篇高重合度文献，写出明确的差异化论述
- [ ] **办理无人机实名登记 + 操控员执照 + 飞行活动申请**（按《无人驾驶航空器飞行管理暂行条例》，以民航局及当地最新规定为准 —— 尽早开始，别等采数据时才发现飞不了）
- [ ] 下载 LoveDA / FloodNet，用 `inspect` 核对标签索引
- [ ] 跑通 B 组（卫星预训练 → UAV 微调）
- [ ] 预算曲线首版结果
- [ ] 确定自采水体场景（河流/水库/城市河道）并踩点
