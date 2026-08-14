"""Attention blocks: the demonstration set, and the EPN episodic memory.

Two different attentions, for two different reasons.

``SetAttention`` runs over the demonstration slots. Demonstrations are a *set*
-- their order carries no information -- so it uses no positional encoding and
max-pools the result. That is what makes 2 demonstrations and 6 demonstrations
the same computation with a different mask, and it is the direct answer to
"how do we handle 1-10 training examples".

``EpisodicMemory`` is the block from the A2C-EPN Alchemy work, transplanted:
each past step is a memory slot, the *current* state is concatenated onto every
slot before projection, the slots are attended with a fill mask, and the result
is max-pooled into a single vector that feeds the LSTM. Only written slots are
visible, so attention is causal by construction rather than by triangular mask.
Positional encoding *is* used here, because for a program being written one
statement at a time the order is the point.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SetAttention", "EpisodicMemory", "PositionalEncoding"]


class PositionalEncoding(nn.Module):
    """Standard sinusoidal encoding, added in place."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class _Block(nn.Module):
    """Pre-norm attention + MLP with the EPN's ELU residual style."""

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None) -> None:
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ELU(), nn.Linear(d_ff, d_model), nn.ELU()
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        attended, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        # nn.MultiheadAttention emits NaN for a fully masked row. Rows are never
        # fully masked in practice (slot 0 is always valid), but a NaN here
        # would silently poison every subsequent update, so it is clamped.
        attended = torch.nan_to_num(attended)
        x = F.elu(x + attended)
        return self.mlp(x)


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max over dim 1, ignoring masked positions. ``mask`` is 1 for valid."""
    neg = torch.finfo(x.dtype).min
    filled = x.masked_fill(mask.unsqueeze(-1) < 0.5, neg)
    pooled = filled.amax(dim=1)
    empty = mask.sum(dim=1, keepdim=True) < 0.5
    return torch.where(empty, torch.zeros_like(pooled), pooled)


class SetAttention(nn.Module):
    """Permutation-invariant encoder over the demonstration slots."""

    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(d_model, n_heads) for _ in range(n_layers))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``tokens`` (B, S, D), ``mask`` (B, S) with 1 for real slots.

        Returns ``(pooled (B, D), tokens (B, S, D))``. Both max and mean pooling
        were plausible; max is what the EPN used and it is the better fit here,
        because "some demonstration is still wrong" is a max-flavoured fact.
        """
        pad = mask < 0.5
        x = tokens
        for block in self.blocks:
            x = block(x, pad)
        x = self.out_norm(x)
        return _masked_max(x, mask), x


class EpisodicMemory(nn.Module):
    """The A2C-EPN memory block over the statements written so far.

    Parameters
    ----------
    mem_dim:
        Width of a stored memory token.
    d_model:
        Width of the attention.
    """

    def __init__(
        self,
        mem_dim: int,
        state_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        max_len: int = 32,
    ) -> None:
        super().__init__()
        self.mem_dim = mem_dim
        self.proj = nn.Linear(mem_dim + state_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ELU(), nn.Linear(d_model, d_model), nn.ELU()
        )
        self.d_model = d_model

    def forward(
        self, memory: torch.Tensor, mask: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """``memory`` (B, M, mem_dim), ``mask`` (B, M), ``state`` (B, state_dim)."""
        b, m, _ = memory.shape
        expanded = state.unsqueeze(1).expand(b, m, state.shape[-1])
        x = self.proj(torch.cat([memory, expanded], dim=-1))
        x = self.pos(x)
        h = self.norm(x)
        attended, _ = self.attn(h, h, h, key_padding_mask=mask < 0.5, need_weights=False)
        attended = torch.nan_to_num(attended)
        x = F.elu(x + attended)
        x = self.mlp(x)
        return _masked_max(x, mask)


def write_memory(
    memory: torch.Tensor, mask: torch.Tensor, token: torch.Tensor, position: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functionally write one token per batch row at ``position``.

    Out-of-range positions (an episode longer than the buffer) are dropped
    rather than wrapping: for program synthesis the *first* statements are the
    ones worth remembering.
    """
    b, m, _ = memory.shape
    pos = position.clamp(0, m - 1).long()
    in_range = (position >= 0) & (position < m)
    onehot = F.one_hot(pos, m).to(memory.dtype) * in_range.unsqueeze(-1).to(memory.dtype)
    new_memory = memory * (1.0 - onehot).unsqueeze(-1) + token.unsqueeze(1) * onehot.unsqueeze(-1)
    new_mask = torch.clamp(mask + onehot, max=1.0)
    return new_memory, new_mask
