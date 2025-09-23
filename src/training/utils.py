# FILE: src/training/utils.py
from __future__ import annotations

import os
import sys
import time
import json
import math
import random
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# -----------------------------
# Logging helpers
# -----------------------------

def _ts() -> str:
    """Timestamp like 2025-09-19 23:59:59"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, *, file: Optional[Path] = None):
    """Simple console logger with timestamp; optional append to file."""
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if file is not None:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, label: str = "", echo: bool = True):
        self.label = label
        self.echo = echo
        self._t0 = 0.0
        self._elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self.echo and self.label:
            log(f"▶ {self.label} ...")
        return self

    def __exit__(self, exc_type, exc, tb):
        self._elapsed = time.perf_counter() - self._t0
        if self.echo and self.label:
            log(f"✔ {self.label} done in {self.pretty()}")

    @property
    def seconds(self) -> float:
        return self._elapsed if self._elapsed else (time.perf_counter() - self._t0)

    def pretty(self) -> str:
        s = self.seconds
        if s < 1:
            return f"{s*1e3:.1f} ms"
        if s < 60:
            return f"{s:.2f} s"
        m = math.floor(s / 60)
        return f"{m}m {s - 60*m:.1f}s"


# -----------------------------
# Reproducibility / device
# -----------------------------

def set_seed(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
    """
    Set seeds for Python, NumPy, and (if available) PyTorch.
    If deterministic=True, we switch cuDNN to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms") and deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not deterministic)


def get_device(requested: str = "auto") -> str:
    """
    Pick device string: 'cuda' if available and requested in ('auto','cuda'), else 'cpu'.
    """
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req in ("auto", "cuda"):
        if _HAS_TORCH and torch.cuda.is_available():
            return "cuda"
    return "cpu"


# Handy for DataLoader workers (if когда-нибудь пригодится)
def make_worker_init_fn(seed: int):
    """
    Returns a worker_init_fn that sets per-worker seeds (NumPy + random).
    Use with DataLoader(worker_init_fn=make_worker_init_fn(seed)).
    """
    def _init(worker_id: int):
        s = seed + worker_id
        random.seed(s)
        np.random.seed(s % (2**32 - 1))
    return _init


# -----------------------------
# Small utilities
# -----------------------------

def count_parameters(model, trainable_only: bool = True) -> int:
    """Number of parameters in a torch model."""
    if not _HAS_TORCH:
        return 0
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_json(obj: Dict[str, Any], path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def pushdir(path: Path):
    """
    Temporarily change working directory.
    """
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


__all__ = [
    "log",
    "Timer",
    "set_seed",
    "get_device",
    "make_worker_init_fn",
    "count_parameters",
    "save_json",
    "load_json",
    "pushdir",
]
