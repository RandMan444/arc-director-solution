"""Seeding. Every run records its seeds; every seed sets every generator."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_seed", "new_rng"]


def set_seed(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python, numpy and torch (if installed).

    ``deterministic_torch`` also pins cuDNN, which costs throughput. Use it for
    the identical-start checks and the ablation runs, not for exploratory work.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def new_rng(seed: Optional[int] = None) -> np.random.Generator:
    """A fresh generator. Prefer this over the global numpy state."""
    return np.random.default_rng(seed)
