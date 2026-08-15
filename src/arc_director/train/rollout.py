"""Rollout storage and the tensor plumbing between numpy envs and torch.

The buffer keeps ``T`` steps for ``B`` environments plus one extra observation
so a value can be bootstrapped. Observations are stored as stacked tensors on
the training device, because the update pass re-runs the whole segment: the
trunk once, batched over ``T*B``, and then the recurrent core step by step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

__all__ = ["to_tensors", "RolloutBuffer"]

_LONG_KEYS = ("last_action",)
_BOOL_KEYS = ("op_mask", "arg_mask")


def to_tensors(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move one batched observation dict onto the device with the right dtypes."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if key in _LONG_KEYS:
            tensor = tensor.long()
        elif key in _BOOL_KEYS:
            tensor = tensor.bool()
        elif key == "grids":
            tensor = tensor.long()
        elif key == "grid_shapes":
            tensor = tensor.float()
        else:
            tensor = tensor.float()
        out[key] = tensor.to(device, non_blocking=True)
    return out


@dataclass
class RolloutBuffer:
    """One on-policy segment.

    Everything is appended per step and stacked once at the end, which keeps
    the collection loop free of pre-allocated shape bookkeeping.
    """

    device: torch.device
    worker_temperature: float = 1.0
    manager_temperature: float = 1.0
    obs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    ops: List[torch.Tensor] = field(default_factory=list)
    args: List[torch.Tensor] = field(default_factory=list)
    manager_codes: List[torch.Tensor] = field(default_factory=list)
    manager_active: List[torch.Tensor] = field(default_factory=list)
    feedback: List[torch.Tensor] = field(default_factory=list)
    # Log-probabilities as sampled. Not used by A2C -- the update recomputes
    # them -- but keeping them makes the replay path testable, and a PPO ratio
    # would need exactly this.
    behaviour_logp: List[torch.Tensor] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)
    infos: List[List[dict]] = field(default_factory=list)
    final_obs: Optional[Dict[str, torch.Tensor]] = None
    final_feedback: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.ops)

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        op: torch.Tensor,
        args: torch.Tensor,
        codes: torch.Tensor,
        active: torch.Tensor,
        feedback: torch.Tensor,
        reward: np.ndarray,
        done: np.ndarray,
        info: List[dict],
        behaviour_logp: Optional[torch.Tensor] = None,
    ) -> None:
        self.obs.append(obs)
        if behaviour_logp is not None:
            self.behaviour_logp.append(behaviour_logp)
        self.ops.append(op)
        self.args.append(args)
        self.manager_codes.append(codes)
        self.manager_active.append(active)
        self.feedback.append(feedback)
        self.rewards.append(reward.copy())
        self.dones.append(done.copy())
        self.infos.append(info)

    # -- stacked views ---------------------------------------------------
    def stacked_obs(self) -> Dict[str, torch.Tensor]:
        """All observations as ``(T*B, ...)`` for one batched trunk pass."""
        return {
            k: torch.stack([o[k] for o in self.obs], dim=0).flatten(0, 1)
            for k in self.obs[0]
        }

    def tensor(self, name: str) -> torch.Tensor:
        return torch.stack(getattr(self, name), dim=0)

    def reward_tensor(self) -> torch.Tensor:
        return torch.as_tensor(np.stack(self.rewards), dtype=torch.float32, device=self.device)

    def done_tensor(self) -> torch.Tensor:
        return torch.as_tensor(np.stack(self.dones), dtype=torch.float32, device=self.device)

    def finished_infos(self) -> List[dict]:
        """Terminal info dicts from every episode that ended in this segment."""
        out = []
        for step_infos, dones in zip(self.infos, self.dones):
            for info, done in zip(step_infos, dones):
                if done:
                    out.append(info)
        return out
