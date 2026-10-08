"""租 GPU 前后的**机时标定**脚本：量出真实的「秒/步」，再据此定网格规模。

为什么要有这个脚本
------------------
按小时租 GPU，需要先回答「租几个小时」。而这个问题只能靠实测回答 ——
同一个 U-Net+ResNet18 在不同卡、不同 vCPU 数、不同 num_workers 下，
每步耗时能差好几倍，且本模型只有 14.33M 参数，瓶颈常常**不在 GPU 而在数据加载**。
凭显卡型号估机时会错得很远。

所以流程是：开机 → 跑本脚本 10 分钟 → 拿到秒/步 → 由脚本算出各网格要几小时
→ 再决定跑哪些、租多久。不要跳过这一步直接开长网格。

本脚本做三件事
--------------
1. **环境体检**：torch 版本、CUDA 是否可用、卡型号与显存、CPU 核数。
2. **可复现性体检**（🔴 最容易踩的坑）：本项目 `deterministic: true`。
   该开关在 CPU 上无代价，但在 CUDA 上要求所有算子都有确定性实现 ——
   个别算子没有，会直接抛 RuntimeError。若开长网格前没验，
   可能跑了两小时才崩。脚本会**真的跑一步**来验，而不是只读配置。
3. **测速**：在真实配置（真实模型、真实数据、真实损失、真实 batch 与裁剪尺寸）
   下跑若干步，报 中位数 秒/步，并按文末的网格定义外推总机时。

🔴 一条必须守住的纪律：CPU 与 GPU 的结果不可混入同一个配对比较
----------------------------------------------------------------
两者的数值路径（fp32 / AMP fp16、不同卷积算子实现）与随机数流都不同，
**同一个 seed 编号在 CPU 和 GPU 上不是同一个实验**。
本文的策略比较是「按种子配对」做检验的，因此：

    一个比较里的所有格（所有策略 × 所有种子）必须在同一设备、
    同一 AMP 开关下重跑；不能只把新增的种子放到 GPU 上。

否则就是在一篇专门讲口径陷阱的论文里，新踩一个口径陷阱。
AMP 同理：决定开或关之后全程固定，不要中途改。

用法
----
    python -m src.bench_device                      # 用 configs/field_steps.yaml
    python -m src.bench_device --steps 30           # 多测几步更稳
    python -m src.bench_device --config configs/rivair.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import time

import torch
from torch.utils.data import DataLoader

from .data import WaterSegDataset, list_samples
from .models import BCEDiceLoss, build_model
from .utils import load_config, set_seed   # 与 train.py 同一套配置加载

# ---------------------------------------------------------------------------
# 待外推的网格定义。改这里即可问「这个规模要几小时」。
#   runs        : 总组数
#   steps       : 每组的梯度步数（固定步数协议下 train.py 实际跑的步数）
#   note        : 打印时的说明
# ---------------------------------------------------------------------------
GRIDS = [
    dict(name="收敛协议 · 同分布策略网格",
         runs=4 * 3 * 10, steps=616,
         note="4 预算 x 3 策略 x 10 种子，S=600。消掉 5.5.3 的具名局限"),
    dict(name="收敛协议 · RivAIrSet 扩池（非饱和设置）",
         runs=4 * 3 * 10, steps=616,
         note="同上规模，换到扩池后的 RivAIrSet 上；该设置未饱和"),
    dict(name="图 5-3 两组重训",
         runs=2, steps=1232,
         note="跨数据集与跨条件各一组，关掉 cleanup 以保留权重"),
]

# 评估与建模型等固定开销占比的经验值。纯测步耗时会低估真实机时，
# 这里按经验上浮；外推值只用于「租几小时」的决策，不要当精确预算。
OVERHEAD = 1.25


def human_hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h < 1:
        return f"{seconds / 60:.0f} 分钟"
    return f"{h:.1f} 小时"


def describe_env() -> torch.device:
    """打印环境体检结果并返回将要使用的设备。"""
    print("环境")
    print("-" * 68)
    print(f"  platform      {platform.platform()}")
    print(f"  python        {platform.python_version()}")
    print(f"  torch         {torch.__version__}")
    print(f"  CPU 核数      {os.cpu_count()}")
    cuda = torch.cuda.is_available()
    print(f"  CUDA 可用     {cuda}")
    if cuda:
        print(f"  CUDA 版本     {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}         {p.name}  显存 {p.total_memory / 1024**3:.1f} GB")
        return torch.device("cuda")
    print("  [i] 没有 CUDA，本次在 CPU 上测 —— 这个数就是 GPU 加速比的分母，"
          "留着和 GPU 上的结果对比")
    return torch.device("cpu")


def build_probe(cfg: dict, device: torch.device, amp: bool):
    """造出与 train.py 完全同构的一套 模型/数据/损失/优化器。

    之所以不复用 train.py 的 main()：那个函数会写实验目录、存权重、跑完整流程，
    而这里只想测速，不能污染 runs/ 下的实验结果。
    """
    model = build_model(cfg["model"]).to(device)
    criterion = BCEDiceLoss(
        bce_weight=cfg["train"].get("bce_weight", 0.5),
        pos_weight=cfg["train"].get("pos_weight"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    stems = list_samples(cfg["data"]["root"], "train")
    ds = WaterSegDataset(
        cfg["data"]["root"], "train", stems=stems,
        augment=cfg["data"].get("augment", True),
        crop_size=cfg["data"].get("crop_size"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        drop_last=len(ds) > cfg["train"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return model, criterion, optimizer, scaler, loader


def time_steps(cfg: dict, device: torch.device, amp: bool,
               deterministic: bool, n_steps: int, warmup: int) -> list[float] | str:
    """跑 warmup + n_steps 步，返回每步耗时列表；若该组合不可用则返回错误字符串。

    为什么要 warmup：首批 step 含 cudnn 算子选择、显存分配器预热、
    数据加载进程启动，耗时能是稳定态的几倍。计入会高估机时。
    """
    set_seed(0, deterministic=deterministic)
    try:
        model, criterion, optimizer, scaler, loader = build_probe(cfg, device, amp)
    except Exception as e:                      # 建模型就失败（如依赖缺失）
        return f"构建失败：{type(e).__name__}: {e}"

    times: list[float] = []
    done = 0
    model.train()
    try:
        # 数据可能不够跑满，所以外面再套一层循环反复取
        while done < warmup + n_steps:
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)

                if device.type == "cuda":
                    torch.cuda.synchronize()    # GPU 是异步的，不同步会测出假的快
                t0 = time.perf_counter()

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = criterion(model(images), masks)
                scaler.scale(loss).backward()
                if cfg["train"].get("grad_clip"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                done += 1
                if done > warmup:
                    times.append(dt)
                if done >= warmup + n_steps:
                    break
    except RuntimeError as e:
        # 🔴 这正是要抓的那类错误：确定性算子缺失在 CUDA 上表现为 RuntimeError
        return f"RuntimeError: {str(e).splitlines()[0]}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0]}"

    return times


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/field_steps.yaml")
    ap.add_argument("--steps", type=int, default=20, help="计时步数（不含 warmup）")
    ap.add_argument("--warmup", type=int, default=5, help="预热步数，不计入")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = describe_env()

    print(f"\n配置 {args.config}")
    print("-" * 68)
    print(f"  模型          {cfg['model']['name']} / {cfg['model']['encoder']}")
    print(f"  batch_size    {cfg['train']['batch_size']}")
    print(f"  crop_size     {cfg['data'].get('crop_size')}")
    print(f"  num_workers   {cfg['data'].get('num_workers', 0)}"
          + ("   [!] Linux 云机上建议调到 8，否则数据加载可能成为瓶颈"
             if cfg['data'].get('num_workers', 0) == 0 else ""))
    print(f"  deterministic {cfg.get('deterministic', True)}")

    # 三种组合：确定性+无AMP（本文现行口径）、确定性+AMP、非确定性+AMP
    # 只有 CUDA 上 AMP 才生效，CPU 上后两项与第一项等价，故跳过。
    combos = [("deterministic=True, AMP=off", True, False)]
    if device.type == "cuda":
        combos += [("deterministic=True, AMP=on ", True, True),
                   ("deterministic=False, AMP=on", False, True)]

    print(f"\n测速（{args.warmup} 步预热 + {args.steps} 步计时）")
    print("-" * 68)
    results: dict[str, float] = {}
    for label, det, amp in combos:
        out = time_steps(cfg, device, amp, det, args.steps, args.warmup)
        if isinstance(out, str):
            print(f"  {label}   [-] 不可用 -> {out}")
            continue
        med = statistics.median(out)
        lo, hi = min(out), max(out)
        results[label] = med
        print(f"  {label}   {med:.3f} 秒/步   （区间 {lo:.3f}~{hi:.3f}）")

    if not results:
        print("\n[-] 没有任何可用组合，先解决上面的报错再谈机时。")
        return

    # 取本文现行口径（确定性 + 无 AMP）作为主参考；没有则取最快的可用项
    main_key = "deterministic=True, AMP=off"
    sec = results.get(main_key, min(results.values()))
    print(f"\n按 {sec:.3f} 秒/步 外推（已上浮 {OVERHEAD:.2f}x 计评估与建模开销）")
    print("=" * 68)
    total = 0.0
    for g in GRIDS:
        steps = g["runs"] * g["steps"]
        est = steps * sec * OVERHEAD
        total += est
        print(f"  {g['name']}")
        print(f"      {g['runs']} 组 x {g['steps']} 步 = {steps:,} 步"
              f"  ->  {human_hours(est)}")
        print(f"      {g['note']}")
    print(f"\n  三项合计  {human_hours(total)}（不含装环境、传数据、下载 RivAIrSet）")

    if device.type == "cuda":
        print("\n  [i] 本模型只有 14.33M 参数，单组跑不满一张 4090。"
              "若 vCPU 足够（>=16），可并行跑 4~6 组，")
        print("      总机时大致按并行度缩短 —— 但并行前先确认数据加载不是瓶颈"
              "（把 num_workers 调起来再测一次）。")

    print("\n" + "=" * 68)
    print("[!] 纪律提醒：CPU 与 GPU 的结果不可混入同一个按种子配对的比较。")
    print("    一个比较里的所有格必须在同一设备、同一 AMP 开关下重跑，")
    print("    不能只把新增的种子放到 GPU 上。AMP 开关定了之后全程不要改。")
    print("[!] 若上面 deterministic=True 那一行报了 RuntimeError，")
    print("    说明该卡上缺确定性算子实现 —— 要么换算子，要么改为 "
          "deterministic=false 并在论文中注明，不可默默改掉。")


if __name__ == "__main__":
    main()
