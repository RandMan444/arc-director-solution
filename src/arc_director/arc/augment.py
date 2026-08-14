"""Endless, rule-preserving ARC augmentations.

Adapted from the augmentation scheme in the ARC-AGI-2 "Golden Notebook"
(colour permutation -> horizontal flip -> k*90 degree rotation), generalised
here into an explicit, invertible, seedable transform.

Why these are safe
------------------
If a task's hidden rule is ``f`` with ``f(input) = output``, then for any
bijection ``g`` on grids the augmented task is solved by the conjugated rule
``g . f . g^-1``: applying ``g`` to every input *and* every output yields a
self-consistent task with the same structure and difficulty. Both the dihedral
group D4 (8 elements) and colour permutations are such bijections.

The augmentation must therefore be sampled **once per task** and applied to
every demonstration pair and the test pair alike. Sampling per-pair would
destroy the rule.

Cardinality
-----------
``8 * 9! = 2,903,040`` distinct transforms with the background colour pinned,
``8 * 10! = 29,030,400`` without. Multiplied by demonstration reordering and
leave-one-out re-rooting, the supply is effectively unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .grid import BACKGROUND, NUM_COLORS, Grid

__all__ = [
    "Augmentation",
    "identity_augmentation",
    "sample_augmentation",
    "augment_pairs",
    "enumerate_dihedral",
]


@dataclass(frozen=True)
class Augmentation:
    """An invertible, rule-preserving grid transform.

    Applied in the order: colour permutation, then horizontal flip, then
    ``rot90`` repeated ``rot`` times (counter-clockwise). This matches the
    Golden Notebook ordering so results stay comparable.

    Parameters
    ----------
    flip:
        Mirror left-right before rotating.
    rot:
        Number of counter-clockwise 90 degree turns, 0-3.
    color_perm:
        Length-10 tuple where ``color_perm[old] = new``.
    """

    flip: bool = False
    rot: int = 0
    color_perm: Tuple[int, ...] = field(default=tuple(range(NUM_COLORS)))

    def __post_init__(self) -> None:
        if self.rot not in (0, 1, 2, 3):
            raise ValueError(f"rot must be 0-3, got {self.rot}")
        if len(self.color_perm) != NUM_COLORS:
            raise ValueError(f"color_perm must have {NUM_COLORS} entries")
        if sorted(self.color_perm) != list(range(NUM_COLORS)):
            raise ValueError("color_perm must be a permutation of 0..9")

    # -- application ----------------------------------------------------
    def apply(self, grid: Grid) -> Grid:
        lut = np.asarray(self.color_perm, dtype=np.int8)
        out = lut[grid]
        if self.flip:
            out = np.fliplr(out)
        if self.rot:
            out = np.rot90(out, k=self.rot)
        return np.ascontiguousarray(out, dtype=np.int8)

    def invert(self, grid: Grid) -> Grid:
        """Map a grid from augmented space back to original space."""
        out = grid
        if self.rot:
            out = np.rot90(out, k=-self.rot)
        if self.flip:
            out = np.fliplr(out)
        inverse_lut = np.empty(NUM_COLORS, dtype=np.int8)
        for old, new in enumerate(self.color_perm):
            inverse_lut[new] = old
        out = inverse_lut[out]
        return np.ascontiguousarray(out, dtype=np.int8)

    @property
    def is_identity(self) -> bool:
        return (
            not self.flip
            and self.rot == 0
            and tuple(self.color_perm) == tuple(range(NUM_COLORS))
        )

    def key(self) -> str:
        """Stable short identifier, useful for logging and dedup."""
        perm = "".join(str(c) for c in self.color_perm)
        return f"f{int(self.flip)}r{self.rot}c{perm}"


def identity_augmentation() -> Augmentation:
    return Augmentation()


def sample_augmentation(
    rng: np.random.Generator,
    *,
    permute_colors: bool = True,
    permute_background: bool = False,
    dihedral: bool = True,
) -> Augmentation:
    """Draw a random augmentation.

    Parameters
    ----------
    permute_background:
        When False (default) colour 0 is pinned. Most ARC tasks treat 0 as
        "empty canvas" rather than as a paintable colour, so remapping it
        changes task semantics more than the other colours. The Golden
        Notebook shuffled all ten; set True to reproduce that.
    """
    flip = bool(rng.integers(2)) if dihedral else False
    rot = int(rng.integers(4)) if dihedral else 0

    perm = list(range(NUM_COLORS))
    if permute_colors:
        if permute_background:
            rng.shuffle(perm)
        else:
            movable = perm[BACKGROUND + 1 :]
            rng.shuffle(movable)
            perm = perm[: BACKGROUND + 1] + movable
    return Augmentation(flip=flip, rot=rot, color_perm=tuple(perm))


def enumerate_dihedral() -> List[Augmentation]:
    """All 8 elements of D4 with the identity colour map."""
    return [
        Augmentation(flip=flip, rot=rot)
        for flip in (False, True)
        for rot in range(4)
    ]


def augment_pairs(
    pairs: Sequence[Tuple[Grid, Grid]], aug: Augmentation
) -> List[Tuple[Grid, Grid]]:
    """Apply one augmentation consistently across a list of (input, output) pairs."""
    return [(aug.apply(inp), aug.apply(out)) for inp, out in pairs]
