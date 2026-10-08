# 数据许可 / Data Licence

本仓库的**代码**与**数据**分开授权。代码见 `LICENSE`（MIT）。
本文件只管数据。

Code and data in this repository are licensed separately. Code is MIT
(see `LICENSE`). This file covers the data only.

---

## 1. 自采数据 / Self-collected data

**范围**：`data/field`（341 帧，227 训练 / 29 验证 / 85 测试）
及其对应的像素级标注 `data/annotations`。
采集于 2026-09-29，青海省，单一水体，大疆无人机。

**许可**：[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

你可以自由使用、修改、再分发，包括商业用途，**条件是署名**。
建议的引用方式：

> Huang, B. and Li, W. (2026). UAV water-body segmentation dataset,
> Qinghai University. [DOI 待补 / DOI to be assigned]

**不在发布范围内**：原始航摄视频（`*.MP4`）与随附的大疆飞行日志
（`*.SRT`）。后者逐帧记录经纬度与绝对海拔，受测绘数据管理规定约束，
由作者留存，不对外分发。采集条件以脱敏后的
`docs/field-conditions-public.csv` 形式发布——其中绝对坐标已换算为
「距五架次质心的相对距离」，地名已替换为条件组代号。

## 2. 实验结果 / Experimental results

**范围**：`results_all_runs.json`（370 组单次运行的 `result.json` 汇总）。

**许可**：CC BY 4.0，同上。论文第五章的每一格都能在这里追溯到产生它的
那次运行（字段 `_run_dir`）。

## 3. 派生自 RivAIrSet 的内容 / Content derived from RivAIrSet

🔴 **这一条不可忽略。**

第 5.2.8 节的跨数据集复现使用了公开数据集 **RivAIrSet**
(Lasalandra et al., 2025)，其原始许可为 **CC BY 4.0**。

本仓库**不再分发** RivAIrSet 的任何影像或标注——请从原作者处获取。
但凡本仓库中由 RivAIrSet 派生的内容（如 `runs/` 下相关实验的结果文件、
`docs/` 中引述的统计量），其署名义务**随派生物传递**：使用这些内容时
必须同时引用 RivAIrSet 原文，不能只引用本文。

> Lasalandra, M. et al. (2025). RivAIrSet. CC BY 4.0.
> 完整条目见论文参考文献 `lasalandra2025rivairset`。

## 4. 为什么代码用 MIT、数据用 CC BY

MIT 是为软件写的，里面的「衍生作品」「源码/目标码」等概念套到图像上
会产生歧义；CC BY 是为数据和作品写的，但它不处理专利与担保免责，
套到代码上又缺一块。分开授权是两边都少踩坑的做法，也是 IEEE 与
Zenodo 对「代码+数据」型存档的常见建议。
