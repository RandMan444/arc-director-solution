"""The goal autoencoder: Director's abstract action space.

Director's manager does not emit a raw state vector -- that space is far too
large to explore. It emits a short code, which a learned decoder turns back
into a point in state space. The code space is trained as a discrete
autoencoder over states the agent has actually visited, so every goal the
manager can name is a goal something like which has been seen, and the manager's
action space is ``n_groups`` categorical choices instead of a continuous vector.

Two quantities come out of this module:

``goal``
    ``decode(code)``, the vector the worker is rewarded for moving toward.
``novelty``
    the reconstruction error of a state. A state the autoencoder cannot
    reproduce is a state unlike anything in recent experience, which is exactly
    Director's exploration reward for the manager.

The autoencoder is trained on *detached* trunk features. It is a description of
the representation, not a second objective pulling on it; letting its gradient
into the encoder would make the goal space and the policy chase each other.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GoalAutoencoder", "max_cosine"]


def max_cosine(goal: torch.Tensor, state: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Director's goal-reaching similarity.

    Plain cosine ignores magnitude, so a worker could score well by pointing in
    the right direction from anywhere. This is cosine scaled by the ratio of
    the smaller norm to the larger, i.e. ``g . s / max(|g|, |s|)^2``, which
    rewards matching direction *and* magnitude and is bounded by 1.
    """
    dot = (goal * state).sum(-1)
    scale = torch.maximum(goal.norm(dim=-1), state.norm(dim=-1)).clamp(min=eps) ** 2
    return dot / scale


class GoalAutoencoder(nn.Module):
    """Discrete (categorical) autoencoder over trunk states.

    Parameters
    ----------
    n_groups, n_codes:
        The code is ``n_groups`` independent categoricals of ``n_codes`` values,
        so the manager's action space is ``n_groups`` heads of ``n_codes`` --
        Director's 8x8 by default, giving 8^8 nameable goals.
    kl_weight:
        Pull of the code posterior toward uniform. Without it the encoder
        collapses onto a handful of codes and the manager's action space
        quietly shrinks to nothing.
    """

    def __init__(
        self,
        state_dim: int,
        n_groups: int = 8,
        n_codes: int = 8,
        hidden: int = 256,
        kl_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_groups = n_groups
        self.n_codes = n_codes
        self.kl_weight = kl_weight

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, n_groups * n_codes),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_groups * n_codes, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, state_dim),
        )

    # -- pieces ----------------------------------------------------------
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Code logits ``(..., n_groups, n_codes)``."""
        logits = self.encoder(state)
        return logits.view(*state.shape[:-1], self.n_groups, self.n_codes)

    def quantize(self, logits: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """Straight-through one-hot: hard forward, soft gradient."""
        probs = F.softmax(logits, dim=-1)
        if not hard:
            return probs
        index = probs.argmax(dim=-1, keepdim=True)
        onehot = torch.zeros_like(probs).scatter_(-1, index, 1.0)
        return onehot + probs - probs.detach()

    def decode(self, onehot: torch.Tensor) -> torch.Tensor:
        """Turn a code (one-hot or straight-through) into a goal vector."""
        flat = onehot.reshape(*onehot.shape[:-2], self.n_groups * self.n_codes)
        return self.decoder(flat)

    def decode_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Turn integer codes ``(..., n_groups)`` into a goal vector."""
        onehot = F.one_hot(codes.long(), self.n_codes).float()
        return self.decode(onehot)

    # -- training / use --------------------------------------------------
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.encode(state)
        onehot = self.quantize(logits)
        return self.decode(onehot), logits

    def loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Reconstruction + uniform KL, on states that must already be detached."""
        recon, logits = self(state)
        recon_loss = F.mse_loss(recon, state)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # KL(q || uniform) = log K - H(q)
        entropy = -(probs * log_probs).sum(-1).mean()
        kl = torch.log(torch.tensor(float(self.n_codes), device=state.device)) - entropy

        loss = recon_loss + self.kl_weight * kl
        stats = {
            "goal_ae/recon": float(recon_loss.detach()),
            "goal_ae/code_entropy": float(entropy.detach()),
            "goal_ae/loss": float(loss.detach()),
        }
        return loss, stats

    @torch.no_grad()
    def novelty(self, state: torch.Tensor) -> torch.Tensor:
        """Per-state reconstruction error -- the manager's exploration reward."""
        recon, _ = self(state)
        return ((recon - state) ** 2).mean(dim=-1)
