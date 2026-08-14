"""DSL value types.

Six runtime types (plan section 6): Grid, Object, ObjectSet, Color, Integer,
Bool. Static type checking happens in the parser using the operator registry's
declared signatures, so a program that parses is guaranteed to be
type-consistent before a single cell is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np

from ..arc.grid import BACKGROUND, Grid

__all__ = ["DslType", "Obj", "ObjectSet", "type_of", "DSL_VERSION"]

DSL_VERSION = "v0"


class DslType(str, Enum):
    GRID = "Grid"
    OBJECT = "Object"
    OBJECT_SET = "ObjectSet"
    COLOR = "Color"
    INTEGER = "Integer"
    BOOL = "Bool"

    def __str__(self) -> str:  # keeps error messages readable
        return self.value


Cell = Tuple[int, int]


@dataclass(frozen=True)
class Obj:
    """A set of cells carrying colours, together with the shape of its host grid.

    Keeping ``grid_shape`` lets an object be painted back onto a canvas of the
    original size without threading the source grid through every operator.
    """

    cells: FrozenSet[Cell]
    colors: Tuple[Tuple[Cell, int], ...]
    grid_shape: Tuple[int, int]

    @classmethod
    def from_mapping(cls, mapping: Dict[Cell, int], grid_shape: Tuple[int, int]) -> "Obj":
        items = tuple(sorted(mapping.items()))
        return cls(
            cells=frozenset(mapping.keys()),
            colors=items,
            grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        )

    @property
    def color_map(self) -> Dict[Cell, int]:
        return dict(self.colors)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """``(top, left, bottom, right)`` inclusive. Empty objects give (0,0,-1,-1)."""
        if not self.cells:
            return (0, 0, -1, -1)
        rows = [r for r, _ in self.cells]
        cols = [c for _, c in self.cells]
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def height(self) -> int:
        t, _, b, _ = self.bbox
        return max(0, b - t + 1)

    @property
    def width(self) -> int:
        _, l, _, r = self.bbox
        return max(0, r - l + 1)

    def dominant_color(self) -> int:
        """Most common colour, ties broken by lowest colour index."""
        if not self.colors:
            return BACKGROUND
        counts: Dict[int, int] = {}
        for _, color in self.colors:
            counts[color] = counts.get(color, 0) + 1
        return min(counts, key=lambda c: (-counts[c], c))

    def normalized_shape(self) -> FrozenSet[Cell]:
        """Cells translated so the bounding box starts at (0, 0), colours dropped.

        Two objects are "the same shape" when these agree.
        """
        if not self.cells:
            return frozenset()
        t, l, _, _ = self.bbox
        return frozenset((r - t, c - l) for r, c in self.cells)

    def to_grid(self, background: int = BACKGROUND) -> Grid:
        """Render just this object, cropped to its bounding box."""
        if not self.cells:
            return np.full((1, 1), background, dtype=np.int8)
        t, l, b, r = self.bbox
        out = np.full((b - t + 1, r - l + 1), background, dtype=np.int8)
        for (row, col), color in self.colors:
            out[row - t, col - l] = color
        return out

    def translated(self, dr: int, dc: int) -> "Obj":
        return Obj.from_mapping(
            {(r + dr, c + dc): color for (r, c), color in self.colors}, self.grid_shape
        )

    def recolored(self, color: int) -> "Obj":
        return Obj.from_mapping({cell: color for cell in self.cells}, self.grid_shape)


@dataclass(frozen=True)
class ObjectSet:
    """An ordered collection of objects.

    Order is deterministic (reading order of each object's top-left cell) so
    that ``LARGEST`` and friends never depend on set iteration order.
    """

    objects: Tuple[Obj, ...]

    @classmethod
    def of(cls, objects: Iterable[Obj]) -> "ObjectSet":
        ordered = sorted(objects, key=lambda o: (o.bbox[0], o.bbox[1], -o.size))
        return cls(objects=tuple(ordered))

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)

    @property
    def is_empty(self) -> bool:
        return len(self.objects) == 0


def type_of(value: object) -> Optional[DslType]:
    """Runtime type of a DSL value, or None if it is not a DSL value.

    ``bool`` is checked before ``int`` because Python's bool subclasses int.
    """
    if isinstance(value, bool):
        return DslType.BOOL
    if isinstance(value, np.ndarray):
        return DslType.GRID
    if isinstance(value, Obj):
        return DslType.OBJECT
    if isinstance(value, ObjectSet):
        return DslType.OBJECT_SET
    if isinstance(value, (int, np.integer)):
        return DslType.INTEGER
    return None
