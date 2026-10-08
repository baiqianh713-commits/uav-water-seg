"""数据集与数据增强。

约定的预处理后目录结构（由 prepare_data.py 生成）：
    data/processed/<dataset>/<split>/images/xxx.png
    data/processed/<dataset>/<split>/masks/xxx.png     # 0 = 非水, 255 = 水

统一成这个结构后，卫星数据（LoveDA/WHDLD）和无人机数据（FloodNet/自采）
可以用同一套 Dataset 加载，跨平台迁移实验只需换配置里的路径。

增强只用 numpy + cv2 实现，不依赖 albumentations，
这样本地零安装就能跑冒烟测试。装了 albumentations 也可以在配置里切过去。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet 统计量。用 ImageNet 预训练权重时必须用这组值归一化，
# 否则输入分布和预训练时不一致，迁移效果会明显变差。
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_samples(root: str | Path, split: str) -> list[str]:
    """列出某个 split 下所有样本的文件名（不含扩展名），排序保证可复现。"""
    img_dir = Path(root) / split / "images"
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"找不到目录 {img_dir}。请先运行 prepare_data.py 生成预处理数据。"
        )
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    if not stems:
        raise RuntimeError(f"{img_dir} 里没有 png 文件。")
    return stems


class WaterSegDataset(Dataset):
    """二值水体分割数据集。

    参数
    ----
    root      : 数据集根目录，如 data/processed/floodnet
    split     : "train" / "val" / "test"
    stems     : 只加载指定的样本名列表。标注预算实验的关键——
                传入选中的 N 个样本即可，无需复制文件。None 表示全部。
    augment   : 是否做数据增强（只在训练集开启）
    crop_size : 随机裁剪尺寸。None 表示不裁剪，用整张图。
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        stems: list[str] | None = None,
        augment: bool = False,
        crop_size: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.stems = stems if stems is not None else list_samples(root, split)
        self.augment = augment
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.stems)

    def _load(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        img_path = self.img_dir / f"{stem}.png"
        mask_path = self.mask_dir / f"{stem}.png"

        # cv2 默认读成 BGR，转成 RGB 以匹配 ImageNet 预训练的通道顺序
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"读取失败：{img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"读取失败：{mask_path}")
        mask = (mask > 127).astype(np.uint8)  # 归一到 {0, 1}

        return image, mask

    def _random_crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = self.crop_size
        h, w = mask.shape
        # 图比裁剪尺寸小就先补边（反射填充比补零更自然）
        if h < size or w < size:
            pad_h, pad_w = max(0, size - h), max(0, size - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = mask.shape
        top = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return image[top : top + size, left : left + size], mask[top : top + size, left : left + size]

    def _center_crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """验证/测试用中心裁剪，保证每次结果一致（不能有随机性）。"""
        size = self.crop_size
        h, w = mask.shape
        if h < size or w < size:
            pad_h, pad_w = max(0, size - h), max(0, size - w)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = mask.shape
        top, left = (h - size) // 2, (w - size) // 2
        return image[top : top + size, left : left + size], mask[top : top + size, left : left + size]

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """几何 + 光度增强。

        航拍图像没有固定朝向，所以水平/垂直翻转和 90° 旋转都是合理的增强。
        光度扰动模拟不同光照和天气——这对水体尤其重要，因为水面反光
        随太阳角度剧烈变化，是本课题的核心难点之一。
        """
        if np.random.rand() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if np.random.rand() < 0.5:
            image, mask = image[::-1], mask[::-1]
        k = np.random.randint(0, 4)
        if k:
            image, mask = np.rot90(image, k), np.rot90(mask, k)

        # 亮度/对比度扰动：先转 float 防止 uint8 溢出截断
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(-25, 25)
            contrast = np.random.uniform(0.85, 1.15)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

        # ascontiguousarray：上面的负步长切片会产生非连续数组，torch 不接受
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        image, mask = self._load(stem)

        if self.crop_size is not None:
            if self.augment:
                image, mask = self._random_crop(image, mask)
            else:
                image, mask = self._center_crop(image, mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        # HWC uint8 -> CHW float，并做 ImageNet 归一化
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.from_numpy(image),
            # mask 加一个通道维，配合 1 通道输出的 BCE 损失
            "mask": torch.from_numpy(mask.astype(np.float32))[None],
            "stem": stem,
        }


def compute_water_ratio(root: str | Path, split: str, stems: list[str] | None = None) -> dict[str, float]:
    """统计每张图的水体像素占比。

    用途有两个：
    1. 报告数据集的类别不平衡程度（论文数据章节要写）。
    2. 选帧策略的特征之一——全是水或全无水的帧信息量低。
    """
    mask_dir = Path(root) / split / "masks"
    stems = stems if stems is not None else list_samples(root, split)
    ratios = {}
    for stem in stems:
        mask = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        ratios[stem] = float((mask > 127).mean())
    return ratios
