"""Grid and demonstration-pair encoders.

Cost drives every choice here. One environment step presents
``n_envs x slots x 3`` grids (input, canvas, target), and a rollout is dozens of
steps, so the grid encoder is the single hottest module in the system. It is
therefore a small strided convolution stack rather than anything patch- or
attention-based, it works at a configurable side length (warm-up stages use
12x12 grids and have no reason to pay for 30x30), and it emits a modest
embedding that the pair encoder combines with cheap hand-computed features the
environment already had to calculate anyway.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..arc.grid import MAX_SIDE, NUM_COLORS

__all__ = ["GridEncoder", "PairEncoder"]


class GridEncoder(nn.Module):
    """Encode a padded colour grid into a fixed-length vector.

    Parameters
    ----------
    side:
        Grids arrive padded to 30x30; only the top-left ``side x side`` window
        is looked at. Compute scales with the square of this, so a curriculum
        that never shows a grid wider than 12 should set 12.
    channels:
        Channel widths of the three stride-2 convolutions.

    A validity channel marks which cells are inside the real grid, so padding
    is distinguishable from genuine background -- without it a 4x4 grid and a
    4x4 pattern in the corner of a 30x30 grid would encode identically.
    """

    def __init__(
        self,
        side: int = MAX_SIDE,
        channels: Sequence[int] = (24, 48, 48),
        out_dim: int = 64,
        n_colors: int = NUM_COLORS,
    ) -> None:
        super().__init__()
        if not 1 <= side <= MAX_SIDE:
            raise ValueError(f"side must be in 1..{MAX_SIDE}, got {side}")
        if len(channels) != 3:
            raise ValueError("channels must have three entries")
        self.side = int(side)
        self.n_colors = int(n_colors)
        self.out_dim = int(out_dim)

        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(n_colors + 1, c1, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        # Global pooling keeps the head independent of `side`, so a checkpoint
        # trained on small warm-up grids can be evaluated at 30x30.
        self.head = nn.Sequential(
            nn.Linear(2 * c3 + 2 + n_colors, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, grids: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        """``grids`` ``(..., 30, 30)`` int, ``shapes`` ``(..., 2)`` int."""
        lead = grids.shape[:-2]
        side = self.side
        g = grids[..., :side, :side].reshape(-1, side, side).long()
        s = shapes.reshape(-1, 2).float()

        rows = torch.arange(side, device=g.device).view(1, side, 1)
        cols = torch.arange(side, device=g.device).view(1, 1, side)
        valid = (rows < s[:, 0].view(-1, 1, 1)) & (cols < s[:, 1].view(-1, 1, 1))

        onehot = F.one_hot(g, self.n_colors).permute(0, 3, 1, 2).float()
        onehot = onehot * valid.unsqueeze(1)
        x = torch.cat([onehot, valid.unsqueeze(1).float()], dim=1)

        feat = self.conv(x)
        pooled = torch.cat([feat.amax(dim=(2, 3)), feat.mean(dim=(2, 3))], dim=-1)

        # Two facts a strided convolution reports badly and ARC cares about a
        # lot: the exact grid size, and the colour histogram.
        area = valid.sum(dim=(1, 2)).clamp(min=1).float()
        hist = (onehot.sum(dim=(2, 3)) / area.unsqueeze(-1))
        extras = torch.cat([s / MAX_SIDE, hist], dim=-1)

        out = self.head(torch.cat([pooled, extras], dim=-1))
        return out.view(*lead, self.out_dim)


class PairEncoder(nn.Module):
    """Turn one demonstration slot into one token.

    A slot is ``(input, canvas, target)`` plus the environment's per-slot
    features (role, shapes, canvas-vs-target accuracy). The three grid
    embeddings are concatenated *with their differences* -- what changed from
    input to canvas, and how far the canvas is from the target are the two
    quantities the policy actually reasons about, and making them explicit
    saves the MLP from having to learn subtraction.
    """

    def __init__(
        self,
        grid_encoder: GridEncoder,
        feat_dim: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.grid_encoder = grid_encoder
        g = grid_encoder.out_dim
        self.proj = nn.Sequential(
            nn.Linear(5 * g + feat_dim, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model),
        )
        self.d_model = d_model

    def forward(
        self,
        grids: torch.Tensor,   # (N, S, 3, 30, 30)
        shapes: torch.Tensor,  # (N, S, 3, 2)
        feats: torch.Tensor,   # (N, S, F)
    ) -> torch.Tensor:
        """Returns slot tokens ``(N, S, d_model)``.

        ``N`` is whatever leading dimension the caller has -- during a policy
        update it is ``timesteps * envs``, because every grid in a rollout is
        encoded in one batched call and only the recurrent core is stepped in a
        Python loop.
        """
        embedded = self.grid_encoder(grids, shapes)      # (N, S, 3, D)
        e_in, e_canvas, e_target = embedded.unbind(dim=2)

        x = torch.cat(
            [e_in, e_canvas, e_target, e_canvas - e_in, e_canvas - e_target, feats],
            dim=-1,
        )
        return self.proj(x)
