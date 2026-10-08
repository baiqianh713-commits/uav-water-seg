"""训练主脚本。

核心特性（都是为"租云 GPU 按小时计费"这个场景设计的）：
  - 全配置驱动：批量实验不改代码，YAML + 命令行覆盖即可。
  - 断点续训：--resume 从上次中断处继续，云主机断连不丢进度。
  - 标注预算子集：--budget N 只用 N 张训练图，模拟低标注场景。
  - 两阶段迁移：--init_from 载入卫星数据预训练的权重。

用法
----
  # 冒烟测试（本地 CPU，用合成数据）
  python -m src.train --config configs/smoke.yaml

  # 正式实验：随机选 20 张 vs 本文方法选 20 张
  python -m src.train --config configs/floodnet.yaml --budget 20 --strategy random   --seed 0
  python -m src.train --config configs/floodnet.yaml --budget 20 --strategy hybrid   --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import re

from .data import WaterSegDataset, compute_water_ratio, list_samples
from .metrics import BinarySegMeter
from .models import BCEDiceLoss, build_model
from .selection import select_frames
from .utils import (
    append_jsonl,
    load_config,
    pick_device,
    save_config,
    set_seed,
    setup_logger,
    worker_init_fn,
)


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    boundary_tolerance: int = 3,
    amp: bool = False,
    boundary_max_side: int = 512,
) -> dict[str, float]:
    """在给定 loader 上评估。返回指标字典。"""
    model.eval()
    meter = BinarySegMeter(boundary_tolerance=boundary_tolerance,
                           boundary_max_side=boundary_max_side)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)

        # autocast 下 logits 可能是 fp16，先转回 fp32 再算 sigmoid 保证数值稳定
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        pred = (probs[:, 0] > threshold).astype(np.uint8)
        target = masks.numpy()[:, 0].astype(np.uint8)
        meter.update(pred, target)

    return meter.compute()


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_by_condition(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    pattern: str,
    logger,
    amp: bool = False,
) -> dict[str, dict[str, float]]:
    """把测试集按采集条件分组，逐组评估。

    为什么需要这个（贡献点 2 的核心判据）
    ------------------------------------
    测试集 85 帧里 A+B 占 68 帧、C 只占 17 帧。**总体 IoU 被 A+B 主导**，
    即使某个预算分配方案完全漏掉了 C（暗光），总体 IoU 也只掉两三个点，
    看起来"没什么影响"。但对水利应用而言，"在某一种采集条件下彻底失效"
    是不可接受的失败，不是一个可以被平均掉的小损失。

    故贡献点 2 的主判据取**最差条件 IoU**（三组中的最小值），
    总体 IoU 作为辅助。条件感知的分配策略即使不提高均值，
    只要能保证每个条件都分到样本、从而抬高最差条件，就是有价值的 ——
    而条件盲的随机抽样在低预算下有相当概率给 C 分到 0 帧。

    这同时解释了为什么必须单独写这个函数：现有的聚合指标**在设计上看不见**
    这个失败模式。

    返回 {条件ID: 指标字典}。某条件在测试集中不存在时不出现在结果里。
    """
    root = cfg["data"]["root"]
    split = cfg["data"].get("test_split", "test")
    stems = list_samples(root, split)

    rx = re.compile(pattern)
    groups: dict[str, list[str]] = {}
    for s in stems:
        mt = rx.match(s)
        groups.setdefault(mt.group(1) if mt else "(未知)", []).append(s)

    bs = cfg["train"].get("eval_batch_size", cfg["train"]["batch_size"])
    out: dict[str, dict[str, float]] = {}
    for cond in sorted(groups):
        ds = WaterSegDataset(root, split, stems=groups[cond], augment=False, crop_size=None)
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            num_workers=cfg["data"].get("num_workers", 0),
                            pin_memory=torch.cuda.is_available(),
                            worker_init_fn=worker_init_fn)
        met = evaluate(
            model, loader, device,
            threshold=cfg["train"].get("threshold", 0.5),
            boundary_tolerance=cfg["train"].get("boundary_tolerance", 3),
            amp=amp,
            boundary_max_side=cfg["train"].get("boundary_max_side", 512),
        )
        met["n_images"] = len(ds)
        out[cond] = met
        logger.info(f"    {cond:<16} n={len(ds):>3}  iou={met['iou']:.4f}  "
                    f"f1={met['f1']:.4f}  bf={met['boundary_f']:.4f}")
    return out


# ---------------------------------------------------------------------------
def build_loaders(cfg: dict, train_stems: list[str], logger) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    bs = cfg["train"]["batch_size"]
    workers = data_cfg.get("num_workers", 0)

    train_ds = WaterSegDataset(
        root, "train", stems=train_stems,
        augment=data_cfg.get("augment", True),
        crop_size=data_cfg.get("crop_size"),
    )
    # 验证/测试不做增强也不随机裁剪，保证每轮评估结果可比
    val_ds = WaterSegDataset(root, data_cfg.get("val_split", "val"), augment=False, crop_size=None)

    test_ds = None
    test_split = data_cfg.get("test_split")
    if test_split and (Path(root) / test_split / "images").is_dir():
        test_ds = WaterSegDataset(root, test_split, augment=False, crop_size=None)

    logger.info(
        f"数据量  train={len(train_ds)}  val={len(val_ds)}"
        + (f"  test={len(test_ds)}" if test_ds else "  test=(无)")
    )

    # pin_memory 只对 GPU 传输有意义；CPU 上开启不但无用还额外占内存
    common = dict(num_workers=workers,
                  pin_memory=torch.cuda.is_available(),
                  worker_init_fn=worker_init_fn)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        # 小预算时（比如只有 5 张图）drop_last 会把整个 batch 丢掉导致无法训练
        drop_last=len(train_ds) > bs,
        **common,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["train"].get("eval_batch_size", bs), shuffle=False, **common)
    test_loader = (
        DataLoader(test_ds, batch_size=cfg["train"].get("eval_batch_size", bs), shuffle=False, **common)
        if test_ds else None
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="实验输出目录，默认按配置自动生成")
    parser.add_argument("--budget", type=int, default=None, help="标注预算 N，不指定则用全部训练数据")
    parser.add_argument("--strategy", default=None, help="条件内选帧策略：random/uniform/kcenter/stratified/difficulty/hybrid/all")
    parser.add_argument("--alloc_strategy", default=None,
                        help="条件间预算分配策略：proportional/uniform/sqrt/diversity/difficulty/div_diff。"
                             "不指定则不分层，全部数据视为单一池子")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="从输出目录的 last.pt 继续训练")
    parser.add_argument("--init_from", default=None, help="载入预训练权重（两阶段迁移的第二阶段用）")
    parser.add_argument("--opts", nargs="*", default=[], help="配置覆盖，如 train.epochs=5")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)

    # 命令行参数优先级高于配置文件
    if args.budget is not None:
        cfg.setdefault("budget", {})["n"] = args.budget
    if args.strategy is not None:
        cfg.setdefault("budget", {})["strategy"] = args.strategy
    if args.alloc_strategy is not None:
        cfg.setdefault("budget", {})["alloc_strategy"] = args.alloc_strategy
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = cfg.get("seed", 0)
    budget_cfg = cfg.get("budget", {}) or {}
    budget_n = budget_cfg.get("n")
    strategy = budget_cfg.get("strategy", "random")

    # 输出目录名里带上关键变量，避免不同实验互相覆盖
    if args.out:
        out_dir = Path(args.out)
    else:
        alloc_tag = budget_cfg.get("alloc_strategy")
        tag = f"{cfg.get('name', 'exp')}"
        if alloc_tag:
            tag += f"_A-{alloc_tag}"
        tag += f"_{strategy}_n{budget_n or 'all'}_s{seed}"
        out_dir = Path(cfg.get("out_root", "runs")) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir)
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    device = pick_device(cfg.get("device", "auto"))
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"

    logger.info(f"实验目录 {out_dir}")
    logger.info(f"设备 {device}  AMP={amp}  种子={seed}")

    # ---- 选帧：决定用哪 N 张图去训练 ----
    all_train_stems = list_samples(cfg["data"]["root"], "train")
    if budget_n is None or strategy == "all" or budget_n >= len(all_train_stems):
        train_stems = all_train_stems
        logger.info(f"使用全部 {len(train_stems)} 张训练图（性能上界组）")
    else:
        t0 = time.time()
        alloc_strategy = budget_cfg.get("alloc_strategy")
        record_extra: dict = {}

        if alloc_strategy:
            # 两层选择：先在采集条件间分配预算，再在条件内选帧（贡献点 2）
            from .allocation import load_condition_map, select_with_allocation

            cond_map = None
            cond_map_path = budget_cfg.get("condition_map")
            if cond_map_path:
                cond_map = load_condition_map(cond_map_path)

            train_stems, detail = select_with_allocation(
                cfg["data"]["root"], "train", budget_n,
                alloc_strategy=alloc_strategy,
                frame_strategy=strategy,
                seed=seed,
                all_stems=all_train_stems,
                condition_pattern=budget_cfg.get("condition_pattern"),
                condition_map=cond_map,
                difficulty_weights=budget_cfg.get("difficulty_weights"),
                hybrid_alpha=budget_cfg.get("hybrid_alpha", 0.5),
                div_diff_alpha=budget_cfg.get("div_diff_alpha", 0.5),
                return_detail=True,
            )
            logger.info(
                f"分配策略={alloc_strategy}  条件内选帧={strategy}  "
                f"预算={budget_n}  实选={len(train_stems)}  耗时={time.time() - t0:.1f}s"
            )
            for cond in sorted(detail):
                d = detail[cond]
                logger.info(
                    f"    {cond:<16} 配额={d['quota']:>4}  池={d['pool_size']:>5}  "
                    f"多样性={d['diversity']:.4f}  难度={d['difficulty_mean']:.4f}"
                )
            record_extra = {
                "alloc_strategy": alloc_strategy,
                "allocation": {c: detail[c]["quota"] for c in detail},
            }
        else:
            # 单层选择：把全部训练数据视为同一个池子
            train_stems = select_frames(
                cfg["data"]["root"], "train", budget_n,
                strategy=strategy, seed=seed, all_stems=all_train_stems,
                difficulty_weights=budget_cfg.get("difficulty_weights"),
                hybrid_alpha=budget_cfg.get("hybrid_alpha", 0.5),
            )
            logger.info(
                f"选帧策略={strategy}  预算={budget_n}  实选={len(train_stems)}  "
                f"耗时={time.time() - t0:.1f}s"
            )

        # 存下选中的样本名。这是实验可复现的关键，也是论文附录材料。
        with open(out_dir / "selected_stems.json", "w", encoding="utf-8") as f:
            json.dump({"strategy": strategy, "n": budget_n, "seed": seed,
                       "stems": train_stems, **record_extra},
                      f, indent=2, ensure_ascii=False)

    # 报告所选子集的水体占比分布——用于论证策略确实选到了有信息量的帧
    ratios = compute_water_ratio(cfg["data"]["root"], "train", train_stems)
    if ratios:
        vals = np.array(list(ratios.values()))
        logger.info(
            f"所选子集水体占比  均值={vals.mean():.3f}  中位={np.median(vals):.3f}  "
            f"全无水的图={int((vals < 1e-6).sum())}/{len(vals)}"
        )

    train_loader, val_loader, test_loader = build_loaders(cfg, train_stems, logger)

    # ---- 模型、损失、优化器 ----
    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型 {cfg['model']['name']}  可训练参数 {n_params / 1e6:.2f}M")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"已从 {args.init_from} 载入权重（缺失 {len(missing)} 项，多余 {len(unexpected)} 项）")

    criterion = BCEDiceLoss(
        bce_weight=cfg["train"].get("bce_weight", 0.5),
        pos_weight=cfg["train"].get("pos_weight"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    epochs = cfg["train"]["epochs"]

    # ---- 按总梯度步数反推轮数（预算曲线实验必须用这个）----
    # 固定 epochs 时，各预算得到的梯度步数与样本量成正比：
    # N=5 只有 12 步，而全量 227 张有 348 步，相差 29 倍。后果有二：
    #   1) 预算曲线里混进了"训练量曲线"，分不清精度提升来自数据还是来自多训；
    #   2) 低预算组严重欠训练，模型连自己那几张图都没学透，
    #      "挑的是哪几张"就显不出差别 —— 选帧策略的信号被实验设计压住了。
    # 固定总步数让各预算都训练到可比的程度，是数据效率类工作的标准做法。
    target_steps = cfg["train"].get("target_steps")
    if target_steps:
        steps_per_epoch = max(1, len(train_loader))
        epochs = max(1, math.ceil(target_steps / steps_per_epoch))
        logger.info(f"固定步数模式：目标 {target_steps} 步，每轮 {steps_per_epoch} 步"
                    f" -> 训练 {epochs} 轮（实际 {steps_per_epoch * epochs} 步）")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    # ---- 断点续训 ----
    start_epoch = 0
    best_metric = -1.0
    monitor = cfg["train"].get("monitor", "iou")
    last_path = out_dir / "last.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1.0)
        logger.info(f"已从 epoch {start_epoch} 恢复训练，历史最佳 {monitor}={best_metric:.4f}")

    save_config(cfg, out_dir / "config.yaml")

    # ---- 训练循环 ----
    eval_every = cfg["train"].get("eval_every", 1)
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, n_batches = 0.0, 0

        # disable=None：非交互终端（重定向到文件、nohup 后台跑）自动关闭进度条，
        # 否则日志文件会被几万行进度条刷爆
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False, disable=None)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(images), masks)

            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip"):
                # unscale 之后才能正确裁剪梯度（否则裁的是放大后的值）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, n_batches)

        record = {"epoch": epoch + 1, "train_loss": train_loss, "lr": scheduler.get_last_lr()[0]}

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            val_metrics = evaluate(
                model, val_loader, device,
                threshold=cfg["train"].get("threshold", 0.5),
                boundary_tolerance=cfg["train"].get("boundary_tolerance", 3),
                amp=amp,
                boundary_max_side=cfg["train"].get("boundary_max_side", 512),
            )
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.info(
                f"epoch {epoch + 1}/{epochs}  loss={train_loss:.4f}  "
                f"val_iou={val_metrics['iou']:.4f}  val_f1={val_metrics['f1']:.4f}  "
                f"val_bf={val_metrics['boundary_f']:.4f}"
            )

            if val_metrics[monitor] > best_metric:
                best_metric = val_metrics[monitor]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch,
                     "best_metric": best_metric, "config": cfg},
                    out_dir / "best.pt",
                )
                logger.info(f"  -> 新的最佳模型，{monitor}={best_metric:.4f}")
        else:
            logger.info(f"epoch {epoch + 1}/{epochs}  loss={train_loss:.4f}")

        append_jsonl(out_dir / "history.jsonl", record)

        # 每个 epoch 都存 last.pt，云主机随时断连都能续上
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict() if amp else None,
             "epoch": epoch, "best_metric": best_metric, "config": cfg},
            last_path,
        )

    # ---- 最终测试：只在此处碰测试集 ----
    # select_by_val=False 时用**最后一个 epoch** 的模型，不按验证集挑。
    # 自采数据的 val 只有 29 帧且时间连续（彼此近重复，等效独立样本仅三五张），
    # 用它在若干检查点之间挑选等于掷硬币，会给每组实验注入一份与策略无关的方差。
    # 低预算下（N=5/10）这份方差足以淹没策略间差异 —— LoveDA 那次就是这么失败的。
    # 固定轮数对所有策略一视同仁，是更干净的比较基准。
    select_by_val = cfg["train"].get("select_by_val", True)
    result = {
        "name": cfg.get("name", "exp"), "strategy": strategy,
        "alloc_strategy": budget_cfg.get("alloc_strategy"), "budget": budget_n,
        "seed": seed, "n_train_used": len(train_stems), f"best_val_{monitor}": best_metric,
        "select_by_val": cfg["train"].get("select_by_val", True),
    }

    if test_loader is not None:
        best_path = out_dir / "best.pt"
        if select_by_val and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
            logger.info("已载入 best.pt 用于测试集评估")
        else:
            logger.info(f"用最后一个 epoch 的模型评估测试集"
                        f"（select_by_val={select_by_val}）")
        test_metrics = evaluate(
            model, test_loader, device,
            threshold=cfg["train"].get("threshold", 0.5),
            boundary_tolerance=cfg["train"].get("boundary_tolerance", 3),
            amp=amp,
            boundary_max_side=cfg["train"].get("boundary_max_side", 512),
        )
        result.update({f"test_{k}": v for k, v in test_metrics.items()})
        logger.info(
            f"测试集  iou={test_metrics['iou']:.4f}  f1={test_metrics['f1']:.4f}  "
            f"boundary_f={test_metrics['boundary_f']:.4f}"
        )

        # ---- 分条件评估（默认关闭）----
        # 默认 false，是为了让已在跑的网格走与之前**完全相同**的代码路径：
        # 同一个网格里若前后若干组用了不同代码，格子之间就不可比了。
        # 只有贡献点 2（预算分配）需要它，由 configs/field_alloc.yaml 打开。
        # try 包住：这是附加指标，任何异常都不该让一组训练白跑。
        cond_pattern = cfg["train"].get("eval_by_condition")
        if cond_pattern:
            logger.info("分条件评估测试集：")
            try:
                by_cond = evaluate_by_condition(
                    model, cfg, device, cond_pattern, logger, amp=amp)
                result["test_by_condition"] = by_cond
                if by_cond:
                    ious = {c: v["iou"] for c, v in by_cond.items()}
                    worst = min(ious, key=ious.get)
                    # 最差条件 IoU 是贡献点 2 的主判据，见 evaluate_by_condition 的说明
                    result["test_worst_condition_iou"] = ious[worst]
                    result["test_worst_condition"] = worst
                    result["test_mean_condition_iou"] = float(np.mean(list(ious.values())))
                    logger.info(f"  最差条件 = {worst}  iou={ious[worst]:.4f}"
                                f"  条件均值 iou={result['test_mean_condition_iou']:.4f}")
            except Exception as e:                      # noqa: BLE001
                logger.warning(f"分条件评估失败（不影响主结果）：{e}")

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 清理权重文件：last.pt 只为断点续训服务，实验结束后无价值；
    # best.pt 可按需保留。批量实验时每组 200+MB，30 组就是 7GB，
    # 磁盘会被占满导致后续实验失败。
    if cfg["train"].get("cleanup_checkpoints", False):
        for name in ("last.pt", "best.pt"):
            fp = out_dir / name
            if fp.exists():
                fp.unlink()
        logger.info("已清理权重文件（cleanup_checkpoints=true）")
    elif cfg["train"].get("cleanup_last", True):
        fp = out_dir / "last.pt"
        if fp.exists():
            fp.unlink()
            logger.info("已清理 last.pt（保留 best.pt）")
    # 汇总到一个总表，sweep.py 直接读这个文件画预算曲线
    append_jsonl(Path(cfg.get("out_root", "runs")) / "all_results.jsonl", result)

    logger.info(f"完成。结果已写入 {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
