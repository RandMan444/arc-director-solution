"""Operator set v2: the extra mechanisms a register-machine worker needs.

Why this file exists
--------------------
The predecessor project audited DSL v0 against ARC and against Hodel's 400
hand-written ARC-1 solvers. v0 certified exact programs for only 1.8% of
ARC-AGI-2 training tasks, and the audit named the missing mechanism classes:

    per-object mapping                66% of known solutions
    set / cell-mask algebra           55%
    coordinate geometry, lines, rays  28%
    arithmetic, comparison, branching 23%
    palette / colour statistics       20%
    splitting, scaling, cell-wise     18%

v1 closed those gaps with 178 *functional* primitives (nested calls, lambdas),
which only works as generated text. A Director worker emits one flat statement
per environment step, so the action space has to stay first-order: no lambdas,
no nesting. This module therefore re-expresses the missing mechanisms as
**flat, vectorised operators** -- ``MAP_RECOLOR(s, c)`` instead of
``apply(rbind(recolor, c), s)`` -- and registers them into the same registry
v0 uses, so the parser, executor, machine and synthetic generator pick them up
with no changes.

What is deliberately still missing is a general ``FOREACH`` block (bind each
object of a set in turn, run a learned body). That is the single largest
remaining expressivity item and it is written up in DESIGN.md; the vectorised
map operators here cover the common "do the same thing to every object" case
without adding control flow to the action space.

Orientation conventions follow ``operators.py``: ``ROTATE90`` is clockwise,
``FLIP_H`` mirrors left-right. Directions are encoded as integers throughout:

    0 = up, 1 = down, 2 = left, 3 = right
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..arc.grid import BACKGROUND, MAX_SIDE, Grid
from .errors import DslLimitError, DslRuntimeError
from .operators import (
    Param,
    _check_color,
    _check_grid,
    _components,
    register,
)
from .types import Cell, DslType, Obj, ObjectSet

__all__ = ["DIRECTIONS", "OPS_V2_VERSION"]

OPS_V2_VERSION = "v2"

G, O, OS, C, I, B = (
    DslType.GRID,
    DslType.OBJECT,
    DslType.OBJECT_SET,
    DslType.COLOR,
    DslType.INTEGER,
    DslType.BOOL,
)
GRID_OR_OBJ = (G, O)

# (dr, dc) for the four direction codes.
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIR_CHOICES = (0, 1, 2, 3)


def _dir(direction: int) -> Tuple[int, int]:
    direction = int(direction)
    if direction not in (0, 1, 2, 3):
        raise DslRuntimeError(
            f"direction must be 0=up 1=down 2=left 3=right, got {direction}",
            code="bad_direction",
        )
    return DIRECTIONS[direction]


def _nonempty(o: Obj, who: str) -> Obj:
    if not o.cells:
        raise DslRuntimeError(f"{who} on an empty object", code="empty_object")
    return o


def _nonempty_set(s: ObjectSet, who: str) -> ObjectSet:
    if s.is_empty:
        raise DslRuntimeError(f"{who} on an empty ObjectSet", code="empty_set")
    return s


# ---------------------------------------------------------------------------
# Palette and colour statistics
# ---------------------------------------------------------------------------


@register(
    "MOST_COMMON_COLOR",
    [Param("g", G)],
    C,
    "The colour covering the most cells, background included. Ties: lowest colour.",
)
def _most_common_color(g: Grid) -> int:
    values, counts = np.unique(g, return_counts=True)
    order = np.lexsort((values, -counts))
    return int(values[order[0]])


@register(
    "MAIN_COLOR",
    [Param("g", G)],
    C,
    "The most common non-background colour. Errors on a blank grid.",
)
def _main_color(g: Grid) -> int:
    fg = g[g != BACKGROUND]
    if fg.size == 0:
        raise DslRuntimeError("MAIN_COLOR on a blank grid", code="blank_grid")
    values, counts = np.unique(fg, return_counts=True)
    order = np.lexsort((values, -counts))
    return int(values[order[0]])


@register(
    "RAREST_COLOR",
    [Param("g", G)],
    C,
    "The least common non-background colour. Errors on a blank grid.",
)
def _rarest_color(g: Grid) -> int:
    fg = g[g != BACKGROUND]
    if fg.size == 0:
        raise DslRuntimeError("RAREST_COLOR on a blank grid", code="blank_grid")
    values, counts = np.unique(fg, return_counts=True)
    order = np.lexsort((values, counts))
    return int(values[order[0]])


@register("COLOR_COUNT", [Param("g", G), Param("color", C)], I, "Cells of the given colour.")
def _color_count(g: Grid, color: int) -> int:
    return int((g == _check_color(color)).sum())


@register("PALETTE_SIZE", [Param("g", G)], I, "Number of distinct colours present, background included.")
def _palette_size(g: Grid) -> int:
    return int(np.unique(g).size)


# ---------------------------------------------------------------------------
# Arithmetic, comparison, branching
# ---------------------------------------------------------------------------


def _clip_int(v: int) -> int:
    # Keeps derived integers inside a range where they can still index a grid.
    return int(np.clip(int(v), -4 * MAX_SIDE, 4 * MAX_SIDE))


@register("ADD", [Param("a", I), Param("b", I)], I, "a + b.")
def _add(a: int, b: int) -> int:
    return _clip_int(int(a) + int(b))


@register("SUB", [Param("a", I), Param("b", I)], I, "a - b.")
def _sub(a: int, b: int) -> int:
    return _clip_int(int(a) - int(b))


@register("MUL", [Param("a", I), Param("b", I)], I, "a * b.")
def _mul(a: int, b: int) -> int:
    return _clip_int(int(a) * int(b))


@register("DIV", [Param("a", I), Param("b", I)], I, "Integer division, truncated toward zero. b=0 errors.")
def _div(a: int, b: int) -> int:
    b = int(b)
    if b == 0:
        raise DslRuntimeError("division by zero", code="div_zero")
    return _clip_int(int(int(a) / b))


@register("MAX_INT", [Param("a", I), Param("b", I)], I, "The larger of two integers.")
def _max_int(a: int, b: int) -> int:
    return int(max(int(a), int(b)))


@register("MIN_INT", [Param("a", I), Param("b", I)], I, "The smaller of two integers.")
def _min_int(a: int, b: int) -> int:
    return int(min(int(a), int(b)))


@register("EQ", [Param("a", I), Param("b", I)], B, "a == b.")
def _eq(a: int, b: int) -> bool:
    return bool(int(a) == int(b))


@register("LT", [Param("a", I), Param("b", I)], B, "a < b.")
def _lt(a: int, b: int) -> bool:
    return bool(int(a) < int(b))


@register("GT", [Param("a", I), Param("b", I)], B, "a > b.")
def _gt(a: int, b: int) -> bool:
    return bool(int(a) > int(b))


@register("NOT", [Param("x", B)], B, "Logical negation.")
def _not(x: bool) -> bool:
    return bool(not x)


@register("AND", [Param("a", B), Param("b", B)], B, "Logical and.")
def _and(a: bool, b: bool) -> bool:
    return bool(a and b)


@register("OR", [Param("a", B), Param("b", B)], B, "Logical or.")
def _or(a: bool, b: bool) -> bool:
    return bool(a or b)


@register(
    "IF_GRID",
    [Param("cond", B), Param("a", G), Param("b", G)],
    G,
    "Return `a` when `cond` is true, otherwise `b`. The DSL's only branch.",
)
def _if_grid(cond: bool, a: Grid, b: Grid) -> Grid:
    return _check_grid((a if cond else b).copy())


@register(
    "IF_INT",
    [Param("cond", B), Param("a", I), Param("b", I)],
    I,
    "Return `a` when `cond` is true, otherwise `b`.",
)
def _if_int(cond: bool, a: int, b: int) -> int:
    return int(a) if cond else int(b)


# ---------------------------------------------------------------------------
# Scaling, splitting, cell-wise combination
# ---------------------------------------------------------------------------


@register(
    "UPSCALE",
    [Param("g", G), Param("k", I)],
    G,
    "Blow every cell up into a k x k block.",
)
def _upscale(g: Grid, k: int) -> Grid:
    k = int(k)
    if k <= 0:
        raise DslRuntimeError(f"UPSCALE factor must be positive, got {k}", code="bad_dims")
    if g.shape[0] * k > MAX_SIDE or g.shape[1] * k > MAX_SIDE:
        raise DslLimitError(f"UPSCALE would produce {g.shape[0] * k}x{g.shape[1] * k}")
    return _check_grid(np.kron(g, np.ones((k, k), dtype=np.int8)))


@register(
    "DOWNSCALE",
    [Param("g", G), Param("k", I)],
    G,
    "Collapse every k x k block to its most common colour. Dimensions must divide by k.",
)
def _downscale(g: Grid, k: int) -> Grid:
    k = int(k)
    if k <= 0:
        raise DslRuntimeError(f"DOWNSCALE factor must be positive, got {k}", code="bad_dims")
    h, w = g.shape
    if h % k or w % k:
        raise DslRuntimeError(f"DOWNSCALE needs {h}x{w} divisible by {k}", code="bad_dims")
    blocks = g.reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(h // k, w // k, k * k)
    out = np.zeros((h // k, w // k), dtype=np.int8)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            values, counts = np.unique(blocks[r, c], return_counts=True)
            out[r, c] = values[np.lexsort((values, -counts))[0]]
    return _check_grid(out)


@register(
    "PART",
    [Param("g", G), Param("n", I), Param("i", I), Param("vertical", B, default=False)],
    G,
    "Split into n equal parts and take part i (0-based). Horizontal by default; "
    "`vertical=true` splits into stacked rows instead.",
)
def _part(g: Grid, n: int, i: int, vertical: bool = False) -> Grid:
    n, i = int(n), int(i)
    if n <= 0:
        raise DslRuntimeError(f"PART count must be positive, got {n}", code="bad_dims")
    if not 0 <= i < n:
        raise DslRuntimeError(f"PART index {i} outside 0..{n - 1}", code="bad_index")
    axis_len = g.shape[0] if vertical else g.shape[1]
    if axis_len % n:
        raise DslRuntimeError(f"PART cannot split {axis_len} into {n}", code="bad_dims")
    step = axis_len // n
    if vertical:
        return _check_grid(g[i * step : (i + 1) * step, :].copy())
    return _check_grid(g[:, i * step : (i + 1) * step].copy())


@register(
    "SUBGRID",
    [Param("g", G), Param("top", I), Param("left", I), Param("h", I), Param("w", I)],
    G,
    "The h x w window whose top-left corner is (top, left). Must lie inside the grid.",
)
def _subgrid(g: Grid, top: int, left: int, h: int, w: int) -> Grid:
    top, left, h, w = int(top), int(left), int(h), int(w)
    gh, gw = g.shape
    if h <= 0 or w <= 0:
        raise DslRuntimeError(f"SUBGRID size must be positive, got {h}x{w}", code="bad_dims")
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        raise DslRuntimeError(
            f"SUBGRID {h}x{w} at ({top},{left}) leaves a {gh}x{gw} grid", code="out_of_bounds"
        )
    return _check_grid(g[top : top + h, left : left + w].copy())


@register(
    "TRIM",
    [Param("g", G), Param("n", I)],
    G,
    "Remove an n-cell border from all four sides. The inverse of PAD.",
)
def _trim(g: Grid, n: int) -> Grid:
    n = int(n)
    if n < 0:
        raise DslRuntimeError(f"TRIM width must be >= 0, got {n}", code="bad_dims")
    h, w = g.shape
    if 2 * n >= h or 2 * n >= w:
        raise DslRuntimeError(f"TRIM {n} would empty a {h}x{w} grid", code="bad_dims")
    return _check_grid(g[n : h - n, n : w - n].copy())


@register(
    "COMPRESS",
    [Param("g", G)],
    G,
    "Delete every row and column that is entirely background.",
)
def _compress(g: Grid) -> Grid:
    keep_rows = ~(g == BACKGROUND).all(axis=1)
    keep_cols = ~(g == BACKGROUND).all(axis=0)
    if not keep_rows.any() or not keep_cols.any():
        raise DslRuntimeError("COMPRESS on a blank grid", code="blank_grid")
    return _check_grid(g[np.ix_(keep_rows, keep_cols)].copy())


@register(
    "CELLWISE",
    [Param("a", G), Param("b", G), Param("fallback", C, default=BACKGROUND)],
    G,
    "Keep cells where two equally sized grids agree; elsewhere write `fallback`.",
)
def _cellwise(a: Grid, b: Grid, fallback: int = BACKGROUND) -> Grid:
    if a.shape != b.shape:
        raise DslRuntimeError(
            f"CELLWISE needs equal shapes, got {a.shape} and {b.shape}", code="shape_mismatch"
        )
    fallback = _check_color(fallback)
    return _check_grid(np.where(a == b, a, np.int8(fallback)))


# ---------------------------------------------------------------------------
# Geometry: gravity, rays, symmetry, frames
# ---------------------------------------------------------------------------


@register(
    "GRAVITY",
    [Param("g", G), Param("direction", I, choices=_DIR_CHOICES)],
    G,
    "Slide every non-background cell as far as it goes in `direction` "
    "(0=up 1=down 2=left 3=right), stacking against the wall and each other.",
)
def _gravity(g: Grid, direction: int) -> Grid:
    dr, dc = _dir(direction)
    out = np.full(g.shape, BACKGROUND, dtype=np.int8)
    if dc == 0:
        for col in range(g.shape[1]):
            values = [v for v in g[:, col] if v != BACKGROUND]
            if dr > 0:
                out[g.shape[0] - len(values) :, col] = values
            else:
                out[: len(values), col] = values
    else:
        for row in range(g.shape[0]):
            values = [v for v in g[row, :] if v != BACKGROUND]
            if dc > 0:
                out[row, g.shape[1] - len(values) :] = values
            else:
                out[row, : len(values)] = values
    return _check_grid(out)


@register(
    "RAY",
    [Param("g", G), Param("o", O), Param("direction", I, choices=_DIR_CHOICES), Param("color", C)],
    G,
    "From every cell of the object, draw a line of `color` in `direction` until "
    "it leaves the grid. The object's own cells are not overwritten.",
)
def _ray(g: Grid, o: Obj, direction: int, color: int) -> Grid:
    _nonempty(o, "RAY")
    dr, dc = _dir(direction)
    color = _check_color(color)
    out = g.copy()
    h, w = out.shape
    for r, c in o.cells:
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            if (nr, nc) not in o.cells:
                out[nr, nc] = color
            nr, nc = nr + dr, nc + dc
    return _check_grid(out)


@register(
    "MIRROR_CONCAT",
    [Param("g", G), Param("direction", I, choices=_DIR_CHOICES)],
    G,
    "Join the grid with its own mirror image on the given side.",
)
def _mirror_concat(g: Grid, direction: int) -> Grid:
    direction = int(direction)
    _dir(direction)
    if direction in (0, 1):
        mirrored = np.flipud(g)
        joined = np.concatenate([mirrored, g] if direction == 0 else [g, mirrored], axis=0)
    else:
        mirrored = np.fliplr(g)
        joined = np.concatenate([mirrored, g] if direction == 2 else [g, mirrored], axis=1)
    return _check_grid(joined)


@register(
    "SYMMETRIZE",
    [Param("g", G), Param("vertical", B, default=False)],
    G,
    "Overlay the grid with its own mirror, filling background from the reflection. "
    "Mirrors left-right by default, up-down when `vertical` is true.",
)
def _symmetrize(g: Grid, vertical: bool = False) -> Grid:
    mirrored = np.flipud(g) if vertical else np.fliplr(g)
    out = g.copy()
    holes = out == BACKGROUND
    out[holes] = mirrored[holes]
    return _check_grid(out)


@register(
    "FRAME",
    [Param("g", G), Param("color", C)],
    G,
    "Paint the outermost ring of cells in `color`.",
)
def _frame(g: Grid, color: int) -> Grid:
    color = _check_color(color)
    out = g.copy()
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    return _check_grid(out)


# ---------------------------------------------------------------------------
# Cell-set algebra
# ---------------------------------------------------------------------------


@register(
    "FOREGROUND",
    [Param("g", G)],
    O,
    "Every non-background cell of the grid as a single object.",
)
def _foreground(g: Grid) -> Obj:
    cells = {(int(r), int(c)): int(g[r, c]) for r, c in zip(*np.nonzero(g != BACKGROUND))}
    if not cells:
        raise DslRuntimeError("FOREGROUND on a blank grid", code="blank_grid")
    return Obj.from_mapping(cells, tuple(g.shape))


@register(
    "MASK_OF_COLOR",
    [Param("g", G), Param("color", C)],
    O,
    "Every cell of the given colour as a single object, connected or not.",
)
def _mask_of_color(g: Grid, color: int) -> Obj:
    color = _check_color(color)
    cells = {(int(r), int(c)): color for r, c in zip(*np.nonzero(g == color))}
    if not cells:
        raise DslRuntimeError(f"no cells of colour {color}", code="empty_object")
    return Obj.from_mapping(cells, tuple(g.shape))


@register("UNION", [Param("a", O), Param("b", O)], O, "All cells of either object; `a` wins overlaps.")
def _union(a: Obj, b: Obj) -> Obj:
    merged = dict(b.color_map)
    merged.update(a.color_map)
    if not merged:
        raise DslRuntimeError("UNION produced an empty object", code="empty_object")
    return Obj.from_mapping(merged, a.grid_shape)


@register("INTERSECT", [Param("a", O), Param("b", O)], O, "Cells present in both objects, coloured from `a`.")
def _intersect(a: Obj, b: Obj) -> Obj:
    shared = {cell: color for cell, color in a.colors if cell in b.cells}
    if not shared:
        raise DslRuntimeError("INTERSECT produced an empty object", code="empty_object")
    return Obj.from_mapping(shared, a.grid_shape)


@register("DIFFERENCE", [Param("a", O), Param("b", O)], O, "Cells of `a` that are not in `b`.")
def _difference(a: Obj, b: Obj) -> Obj:
    rest = {cell: color for cell, color in a.colors if cell not in b.cells}
    if not rest:
        raise DslRuntimeError("DIFFERENCE produced an empty object", code="empty_object")
    return Obj.from_mapping(rest, a.grid_shape)


@register(
    "TRANSLATE_OBJ",
    [Param("o", O), Param("dr", I), Param("dc", I)],
    O,
    "Move an object by (dr, dc) without touching any grid.",
)
def _translate_obj(o: Obj, dr: int, dc: int) -> Obj:
    return _nonempty(o, "TRANSLATE_OBJ").translated(int(dr), int(dc))


@register(
    "MOVE_TO",
    [Param("o", O), Param("top", I), Param("left", I)],
    O,
    "Move an object so its bounding box starts at (top, left).",
)
def _move_to(o: Obj, top: int, left: int) -> Obj:
    _nonempty(o, "MOVE_TO")
    t, l, _, _ = o.bbox
    return o.translated(int(top) - t, int(left) - l)


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------


@register(
    "STAMP",
    [Param("g", G), Param("o", O)],
    G,
    "Paint the object onto the grid keeping its own colours. Cells outside are dropped.",
)
def _stamp(g: Grid, o: Obj) -> Grid:
    out = g.copy()
    h, w = out.shape
    for (r, c), color in o.colors:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color
    return _check_grid(out)


@register(
    "ERASE",
    [Param("g", G), Param("o", O)],
    G,
    "Set the object's cells back to background.",
)
def _erase(g: Grid, o: Obj) -> Grid:
    out = g.copy()
    h, w = out.shape
    for r, c in o.cells:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = BACKGROUND
    return _check_grid(out)


@register(
    "PAINT_SET",
    [Param("g", G), Param("s", OS), Param("color", C)],
    G,
    "Paint every object of the set onto the grid in one colour.",
)
def _paint_set(g: Grid, s: ObjectSet, color: int) -> Grid:
    color = _check_color(color)
    out = g.copy()
    h, w = out.shape
    for obj in s.objects:
        for r, c in obj.cells:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color
    return _check_grid(out)


@register(
    "STAMP_SET",
    [Param("g", G), Param("s", OS)],
    G,
    "Paint every object of the set onto the grid keeping their own colours.",
)
def _stamp_set(g: Grid, s: ObjectSet) -> Grid:
    out = g.copy()
    h, w = out.shape
    for obj in s.objects:
        for (r, c), color in obj.colors:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color
    return _check_grid(out)


@register(
    "ERASE_SET",
    [Param("g", G), Param("s", OS)],
    G,
    "Set every cell of every object in the set back to background.",
)
def _erase_set(g: Grid, s: ObjectSet) -> Grid:
    out = g.copy()
    h, w = out.shape
    for obj in s.objects:
        for r, c in obj.cells:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = BACKGROUND
    return _check_grid(out)


# ---------------------------------------------------------------------------
# Object-set selection and vectorised mapping
# ---------------------------------------------------------------------------


@register(
    "ALL_OBJECTS",
    [Param("g", G), Param("conn", I, default=8, choices=(4, 8))],
    OS,
    "Connected components allowed to span colours -- multi-coloured sprites.",
)
def _all_objects(g: Grid, conn: int = 8) -> ObjectSet:
    return ObjectSet.of(_components(g, int(conn), same_color=False))


@register(
    "SORT_BY_SIZE",
    [Param("s", OS), Param("descending", B, default=True)],
    OS,
    "Order the set by cell count. Ties keep reading order.",
)
def _sort_by_size(s: ObjectSet, descending: bool = True) -> ObjectSet:
    ordered = sorted(
        s.objects,
        key=lambda o: (-o.size if descending else o.size, o.bbox[0], o.bbox[1]),
    )
    return ObjectSet(objects=tuple(ordered))


@register(
    "NTH",
    [Param("s", OS), Param("i", I)],
    O,
    "The i-th object of the set, 0-based, in the set's current order.",
)
def _nth(s: ObjectSet, i: int) -> Obj:
    _nonempty_set(s, "NTH")
    i = int(i)
    if not 0 <= i < len(s):
        raise DslRuntimeError(f"NTH index {i} outside 0..{len(s) - 1}", code="bad_index")
    return s.objects[i]


@register("MERGE", [Param("s", OS)], O, "Fuse every object of the set into one object.")
def _merge(s: ObjectSet) -> Obj:
    _nonempty_set(s, "MERGE")
    cells: Dict[Cell, int] = {}
    for obj in s.objects:
        cells.update(obj.color_map)
    return Obj.from_mapping(cells, s.objects[0].grid_shape)


@register(
    "ODD_ONE_OUT",
    [Param("s", OS)],
    O,
    "The object whose normalised shape is unique in the set. Errors when there "
    "is no unique odd one out.",
)
def _odd_one_out(s: ObjectSet) -> Obj:
    _nonempty_set(s, "ODD_ONE_OUT")
    counts: Dict[frozenset, int] = {}
    for obj in s.objects:
        key = obj.normalized_shape()
        counts[key] = counts.get(key, 0) + 1
    singles = [o for o in s.objects if counts[o.normalized_shape()] == 1]
    if len(singles) != 1:
        raise DslRuntimeError(
            f"ODD_ONE_OUT found {len(singles)} unique shapes, need exactly 1", code="ambiguous"
        )
    return singles[0]


@register(
    "MOST_COMMON_SHAPE",
    [Param("s", OS)],
    O,
    "One representative of the most frequent normalised shape in the set.",
)
def _most_common_shape(s: ObjectSet) -> Obj:
    _nonempty_set(s, "MOST_COMMON_SHAPE")
    counts: Dict[frozenset, int] = {}
    for obj in s.objects:
        key = obj.normalized_shape()
        counts[key] = counts.get(key, 0) + 1
    best = max(counts, key=lambda k: (counts[k], -len(k)))
    for obj in s.objects:
        if obj.normalized_shape() == best:
            return obj
    raise DslRuntimeError("MOST_COMMON_SHAPE found nothing", code="empty_set")  # pragma: no cover


@register(
    "FILTER_LARGER",
    [Param("s", OS), Param("size", I)],
    OS,
    "Objects with strictly more than `size` cells.",
)
def _filter_larger(s: ObjectSet, size: int) -> ObjectSet:
    return ObjectSet.of(o for o in s.objects if o.size > int(size))


@register(
    "FILTER_SMALLER",
    [Param("s", OS), Param("size", I)],
    OS,
    "Objects with strictly fewer than `size` cells.",
)
def _filter_smaller(s: ObjectSet, size: int) -> ObjectSet:
    return ObjectSet.of(o for o in s.objects if o.size < int(size))


@register(
    "EXCLUDE_COLOR",
    [Param("s", OS), Param("color", C)],
    OS,
    "Objects whose dominant colour is *not* the given one.",
)
def _exclude_color(s: ObjectSet, color: int) -> ObjectSet:
    color = _check_color(color)
    return ObjectSet.of(o for o in s.objects if o.dominant_color() != color)


@register(
    "MAP_RECOLOR",
    [Param("s", OS), Param("color", C)],
    OS,
    "Recolour every object of the set. The flat stand-in for `apply(recolor, s)`.",
)
def _map_recolor(s: ObjectSet, color: int) -> ObjectSet:
    color = _check_color(color)
    return ObjectSet(objects=tuple(o.recolored(color) for o in s.objects))


@register(
    "MAP_MOVE",
    [Param("s", OS), Param("dr", I), Param("dc", I)],
    OS,
    "Translate every object of the set by the same offset.",
)
def _map_move(s: ObjectSet, dr: int, dc: int) -> ObjectSet:
    dr, dc = int(dr), int(dc)
    return ObjectSet(objects=tuple(o.translated(dr, dc) for o in s.objects))


@register(
    "MAP_BOX",
    [Param("s", OS)],
    OS,
    "Replace every object by its filled bounding box.",
)
def _map_box(s: ObjectSet) -> ObjectSet:
    out: List[Obj] = []
    for o in s.objects:
        if not o.cells:
            continue
        t, l, b, r = o.bbox
        color = o.dominant_color()
        out.append(
            Obj.from_mapping(
                {(row, col): color for row in range(t, b + 1) for col in range(l, r + 1)},
                o.grid_shape,
            )
        )
    if not out:
        raise DslRuntimeError("MAP_BOX on an empty ObjectSet", code="empty_set")
    return ObjectSet(objects=tuple(out))


@register(
    "MAP_OUTLINE",
    [Param("s", OS)],
    OS,
    "Replace every object by its border cells.",
)
def _map_outline(s: ObjectSet) -> ObjectSet:
    out: List[Obj] = []
    for o in s.objects:
        cmap = o.color_map
        border = {
            cell: cmap[cell]
            for cell in o.cells
            if any(
                (cell[0] + dr, cell[1] + dc) not in o.cells for dr, dc in DIRECTIONS
            )
        }
        if border:
            out.append(Obj.from_mapping(border, o.grid_shape))
    if not out:
        raise DslRuntimeError("MAP_OUTLINE on an empty ObjectSet", code="empty_set")
    return ObjectSet(objects=tuple(out))


# ---------------------------------------------------------------------------
# Canvas helpers keyed off another grid
# ---------------------------------------------------------------------------


@register(
    "CANVAS_LIKE",
    [Param("g", G), Param("color", C, default=BACKGROUND)],
    G,
    "A blank grid the same size as `g`.",
)
def _canvas_like(g: Grid, color: int = BACKGROUND) -> Grid:
    return _check_grid(np.full(g.shape, _check_color(color), dtype=np.int8))


@register(
    "OBJ_GRID",
    [Param("o", O), Param("background", C, default=BACKGROUND)],
    G,
    "Render the object on a canvas the size of its host grid, not cropped.",
)
def _obj_grid(o: Obj, background: int = BACKGROUND) -> Grid:
    _nonempty(o, "OBJ_GRID")
    out = np.full(o.grid_shape, _check_color(background), dtype=np.int8)
    h, w = out.shape
    for (r, c), color in o.colors:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color
    return _check_grid(out)
