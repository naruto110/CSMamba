"""Utility helpers for reproducibility and logging."""

from __future__ import annotations

import datetime
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and prefer != "cpu":
        return torch.device("cuda")
    return torch.device("cpu")


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
