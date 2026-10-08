"""二值水体分割的评价指标。

重要区分（论文里必须写清楚用的是哪种）：
  - dataset-level（聚合式）：把所有图的 TP/FP/FN 累加，最后算一次 IoU。
  - image-level（逐图平均）：每张图各算一个 IoU，再取平均。
两者数值可以差很多。小目标图（水体只占几个像素）在逐图平均里权重被放大。
本文件同时返回两种，避免和其他论文对比时口径不一致。
"""
from __future__ import annotations

import cv2
import numpy as np


class BinarySegMeter:
    """累积统计器。每个 batch 调用 update()，最后调用 compute()。"""

    def __init__(self, boundary_tolerance: int = 3, boundary_max_side: int = 512):
        # 边界 F-score 的容差（像素）。水体边界精度对水利应用有实际意义，
        # 例如岸线提取、水面面积估算。
        self.boundary_tolerance = boundary_tolerance
        # 边界指标涉及形态学腐蚀/膨胀，在全分辨率大图上内存与耗时都很高
        # （实测 1024×767 × 47 张会把内存耗尽导致进程被杀）。
        # 因此先把掩膜降采样到该长边再算，容差按同比例缩放，
        # 相对趋势不受影响。设为 0 表示不降采样。
        self.boundary_max_side = boundary_max_side
        self.reset()

    def reset(self) -> None:
        self.tp = 0  # 聚合式统计量
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.per_image_iou: list[float] = []
        self.per_image_f1: list[float] = []
        self.per_image_bf: list[float] = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """pred / target 形状为 (N, H, W) 或 (H, W)，取值 {0, 1}。"""
        pred = np.asarray(pred).astype(bool)
        target = np.asarray(target).astype(bool)
        if pred.ndim == 2:
            pred, target = pred[None], target[None]

        for p, t in zip(pred, target):
            tp = int(np.logical_and(p, t).sum())
            fp = int(np.logical_and(p, ~t).sum())
            fn = int(np.logical_and(~p, t).sum())
            tn = int(np.logical_and(~p, ~t).sum())

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn

            # 逐图指标。注意空标签的处理：如果这张图本来就没有水体，
            # 且模型也没预测出水体，IoU 定义为 1（完全正确）；
            # 若模型误报，则为 0。不能直接跳过，否则会系统性高估。
            denom = tp + fp + fn
            self.per_image_iou.append(1.0 if denom == 0 else tp / denom)
            f1_denom = 2 * tp + fp + fn
            self.per_image_f1.append(1.0 if f1_denom == 0 else 2 * tp / f1_denom)
            self.per_image_bf.append(
                boundary_f_score(p, t, tolerance=self.boundary_tolerance,
                                 max_side=self.boundary_max_side)
            )

    def compute(self) -> dict[str, float]:
        eps = 1e-9
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return {
            # 聚合式（论文主表推荐用这个，更稳定）
            "iou": tp / (tp + fp + fn + eps),
            "f1": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            # 逐图平均式
            "miou_per_image": float(np.mean(self.per_image_iou)) if self.per_image_iou else 0.0,
            "mf1_per_image": float(np.mean(self.per_image_f1)) if self.per_image_f1 else 0.0,
            # 边界质量
            "boundary_f": float(np.mean(self.per_image_bf)) if self.per_image_bf else 0.0,
            "n_images": len(self.per_image_iou),
        }


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """用形态学梯度提取二值掩膜的边界（1 像素宽）。"""
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    return (m - eroded).astype(bool)


def boundary_f_score(pred: np.ndarray, target: np.ndarray, tolerance: int = 3,
                     max_side: int = 0) -> float:
    """边界 F-score（BF score）。

    做法：分别提取预测和真值的边界，各自向外膨胀 tolerance 像素，
    再统计对方边界落在自己膨胀区内的比例，作为 precision / recall。
    tolerance 允许轻微的边界偏移，避免因 1-2 像素错位就判定为完全错误。

    max_side > 0 时先把掩膜降采样到该长边再计算，容差按同比例缩放。
    形态学运算在大图上开销很大，降采样能显著降低内存与耗时，
    且不改变不同方法之间的相对比较结果。
    """
    if max_side and max(pred.shape[:2]) > max_side:
        h, w = pred.shape[:2]
        scale = max_side / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        # 掩膜必须最近邻，插值会引入 0/1 之外的值
        pred = cv2.resize(pred.astype(np.uint8), (nw, nh),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        target = cv2.resize(target.astype(np.uint8), (nw, nh),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        tolerance = max(1, int(round(tolerance * scale)))

    pred_b = _extract_boundary(pred)
    target_b = _extract_boundary(target)

    # 两边都没有边界（整图全水或全非水且预测一致）视为完全正确
    if pred_b.sum() == 0 and target_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or target_b.sum() == 0:
        return 0.0

    kernel_size = 2 * tolerance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    pred_dil = cv2.dilate(pred_b.astype(np.uint8), kernel, iterations=1).astype(bool)
    target_dil = cv2.dilate(target_b.astype(np.uint8), kernel, iterations=1).astype(bool)

    # precision：预测边界中有多少落在真值边界的容差范围内
    precision = np.logical_and(pred_b, target_dil).sum() / pred_b.sum()
    # recall：真值边界中有多少被预测边界覆盖
    recall = np.logical_and(target_b, pred_dil).sum() / target_b.sum()

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
