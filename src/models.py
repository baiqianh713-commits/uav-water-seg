"""模型构建。

三种来源，按需切换：
  - "smp_unet" / "smp_deeplabv3p"：用 segmentation_models_pytorch，正式实验主力。
  - "segformer"：用 HuggingFace transformers，便于和已有文献对比。
  - "tiny_unet"：本文件内置的纯 PyTorch 小 UNet，零额外依赖，
     用于本地 CPU 冒烟测试——租卡前先确认全流程没 bug。

所有模型统一输出 1 个通道的 logits，形状 (N, 1, H, W)，配合 BCE + Dice 做二分类。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 内置小 UNet（仅用于冒烟测试，不要用它跑论文结果）
# ----------------------------------------------------------------------------
def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """两层 3x3 卷积 + BN + ReLU，UNet 的基本单元。"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """4 层下采样的极简 UNet。参数量很小，CPU 上也能跑几个 step。"""

    def __init__(self, in_channels: int = 3, base: int = 16):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)

        # 上采样用转置卷积，再和 encoder 的同尺度特征拼接（skip connection）
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.head = nn.Conv2d(base, 1, 1)  # 输出 1 通道 logits
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ----------------------------------------------------------------------------
# SegFormer 包装：让它的输出和其他模型一样是全分辨率 (N, 1, H, W)
# ----------------------------------------------------------------------------
class SegFormerWrapper(nn.Module):
    """SegFormer 原生输出是 1/4 分辨率，这里统一上采样回原图尺寸。"""

    def __init__(self, backbone: str = "nvidia/mit-b0", pretrained: bool = True):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=1,
                ignore_mismatched_sizes=True,  # 替换分类头，尺寸不匹配是预期的
            )
        else:
            cfg = SegformerConfig.from_pretrained(backbone, num_labels=1)
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


# ----------------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------------
def build_model(cfg: dict) -> nn.Module:
    """按配置构建模型。cfg 为 config 里的 model 段。"""
    name = cfg.get("name", "tiny_unet")
    encoder = cfg.get("encoder", "resnet34")
    # 预训练权重来源："imagenet" / None。跨平台迁移实验里，
    # 阶段二会从卫星数据训好的 checkpoint 载入，那走 --resume/--init_from，不在这里。
    weights = cfg.get("encoder_weights", "imagenet")
    in_channels = cfg.get("in_channels", 3)

    if name == "tiny_unet":
        return TinyUNet(in_channels=in_channels, base=cfg.get("base_channels", 16))

    if name == "segformer":
        return SegFormerWrapper(
            backbone=cfg.get("backbone", "nvidia/mit-b0"),
            pretrained=bool(weights),
        )

    if name.startswith("smp_"):
        import segmentation_models_pytorch as smp

        arch = {
            "smp_unet": smp.Unet,
            "smp_unetpp": smp.UnetPlusPlus,
            "smp_deeplabv3p": smp.DeepLabV3Plus,
            "smp_fpn": smp.FPN,
        }.get(name)
        if arch is None:
            raise ValueError(f"未知的 smp 架构：{name}")
        return arch(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=1,
        )

    raise ValueError(f"未知模型名：{name}")


# ----------------------------------------------------------------------------
# 损失函数
# ----------------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    """BCE + Dice 组合损失。

    为什么组合：水体像素占比常常很低（类别不平衡），纯 BCE 会倾向全预测背景；
    Dice 直接优化重叠度，对不平衡更鲁棒。两者加权互补是分割任务的常用做法。
    pos_weight 可进一步给正样本加权，应对极端不平衡。
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        pw = (
            torch.tensor([self.pos_weight], device=logits.device)
            if self.pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

        probs = torch.sigmoid(logits)
        # 按每张图分别算 Dice 再平均，避免 batch 内大水体图主导梯度
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * target).sum(dims)
        union = probs.sum(dims) + target.sum(dims)
        dice = 1 - (2 * intersection + 1.0) / (union + 1.0)  # +1 平滑项，防除零

        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()
