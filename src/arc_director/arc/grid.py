"""Grid primitives.

An ARC grid is an ``np.ndarray`` of shape ``(H, W)`` and dtype ``int8`` holding
colour indices 0-9. Colour 0 is the background by convention.

Everything downstream (DSL, scoring, augmentation) agrees on this one
representation so no conversion layers are needed.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

# ARC-AGI-2 grids never exceed 30x30 and use 10 colours.
MAX_SIDE = 30
NUM_COLORS = 10
BACKGROUND = 0

Grid = np.ndarray


class GridError(ValueError):
    """A grid violated the ARC representation invariants."""


def to_grid(rows: Sequence[Sequence[int]]) -> Grid:
    """Convert nested lists (the on-disk ARC JSON form) into a validated grid."""
    # Validate before narrowing to int8.  Casting first is lossy: e.g. 256
    # becomes 0 and a malformed dataset cell would silently turn into a valid
    # background pixel.  It also truncates floats such as 1.5 to 1.
    try:
        raw = np.asarray(rows)
    except (TypeError, ValueError) as exc:
        raise GridError(f"grid must be a rectangular 2-D array: {exc}") from exc
    if raw.ndim != 2:
        raise GridError(f"grid must be 2-D, got shape {raw.shape}")
    if raw.dtype.kind not in "iu":
        raise GridError(f"grid cells must be integers, got dtype {raw.dtype}")
    validate(raw)
    return np.asarray(raw, dtype=np.int8)


def to_lists(grid: Grid) -> List[List[int]]:
    """Convert back to the nested-list form used by the ARC JSON files."""
    return [[int(v) for v in row] for row in grid]


def validate(grid: Grid) -> None:
    h, w = grid.shape
    if h == 0 or w == 0:
        raise GridError("grid must be non-empty")
    if h > MAX_SIDE or w > MAX_SIDE:
        raise GridError(f"grid {h}x{w} exceeds the {MAX_SIDE}x{MAX_SIDE} ARC limit")
    lo, hi = int(grid.min()), int(grid.max())
    if lo < 0 or hi >= NUM_COLORS:
        raise GridError(f"colour out of range [0,{NUM_COLORS - 1}]: min={lo} max={hi}")


def is_valid(grid: object) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    try:
        validate(grid)
    except GridError:
        return False
    return True


def empty(h: int, w: int, color: int = BACKGROUND) -> Grid:
    return np.full((h, w), color, dtype=np.int8)


def colors_used(grid: Grid) -> List[int]:
    return sorted(int(c) for c in np.unique(grid))


def foreground_mask(grid: Grid, background: int = BACKGROUND) -> np.ndarray:
    return grid != background


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return ``(top, left, bottom, right)`` inclusive for a boolean mask.

    ``None`` when the mask is empty.
    """
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def crop_to_content(grid: Grid, background: int = BACKGROUND) -> Grid:
    """Crop away uniform background border. Returns the grid unchanged if blank."""
    box = bounding_box(foreground_mask(grid, background))
    if box is None:
        return grid.copy()
    t, l, b, r = box
    return grid[t : b + 1, l : r + 1].copy()


def grids_equal(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def format_grid(grid: Grid) -> str:
    """Compact text rendering for LLM prompts: one row per line, digits packed.

    A 30x30 grid costs ~30 lines of 30 chars, which tokenises far more cheaply
    than JSON-with-commas.
    """
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def parse_grid(text: str) -> Grid:
    """Inverse of :func:`format_grid`."""
    rows = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not rows:
        raise GridError("no rows to parse")
    width = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise GridError(f"row {i} has width {len(row)}, expected {width}")
        if not row.isdigit():
            raise GridError(f"row {i} contains non-digit characters: {row!r}")
    return to_grid([[int(ch) for ch in row] for row in rows])


def shapes(grids: Iterable[Grid]) -> List[tuple[int, int]]:
    return [tuple(g.shape) for g in grids]  # type: ignore[misc]
