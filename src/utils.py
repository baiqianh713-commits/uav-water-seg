"""通用工具：配置加载、随机种子、日志。

设计目标：实验全部由 YAML 配置驱动，命令行只做少量覆盖。
这样批量实验不需要改代码，租 GPU 时可以直接排队跑。
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
def _deep_update(base: dict, new: dict) -> dict:
    """递归合并字典，new 覆盖 base。用于实现配置继承。"""
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """加载 YAML 配置。

    支持两个特性：
    1. `_base_: xxx.yaml` —— 继承另一个配置文件，自己只写差异部分。
    2. overrides —— 形如 ["train.epochs=5", "budget.n=20"] 的命令行覆盖。
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 处理继承：先加载父配置，再用当前配置覆盖
    base_name = cfg.pop("_base_", None)
    if base_name is not None:
        base_cfg = load_config(path.parent / base_name)
        cfg = _deep_update(base_cfg, cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆盖参数格式应为 key=value，收到：{item!r}")
        key, raw_value = item.split("=", 1)
        # 用 YAML 解析值，这样 "5" -> int，"true" -> bool，"[1,2]" -> list
        value = yaml.safe_load(raw_value)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """把最终生效的配置存到实验目录，保证实验可复现。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ----------------------------------------------------------------------------
# 随机种子
# ----------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定所有随机源。

    小样本实验方差极大，必须多种子重复取均值±标准差，
    否则单次结果没有说服力（审稿人一定会问这一点）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # 牺牲一点速度换取可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 多进程时给每个 worker 不同但可复现的种子。"""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ----------------------------------------------------------------------------
# 日志与结果记录
# ----------------------------------------------------------------------------
def setup_logger(out_dir: str | Path, name: str = "train") -> logging.Logger:
    """同时输出到控制台和文件。租卡断连后可以看文件回溯。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复添加 handler 导致日志打印多遍

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(out_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """逐行追加 JSON 记录。断点续训时不会丢历史，也方便 pandas 读。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pick_device(prefer: str = "auto") -> torch.device:
    """选择设备。本地没 GPU 时自动退回 CPU，方便先跑冒烟测试。"""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
