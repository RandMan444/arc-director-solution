"""The v0 operator registry.

Each operator declares its parameter types and return type. The parser type
checks against these declarations; the executor calls ``fn``. Adding an
operator means adding one :func:`register` call and nothing else -- the parser,
the prompt reference and the synthetic generator all read from this registry.

Orientation conventions, pinned here because they are a classic source of
silent bugs:

* ``FLIP_H`` mirrors left-right (``np.fliplr``), ``FLIP_V`` mirrors up-down.
* ``ROTATE90`` turns **clockwise**. ``ROTATE270`` is therefore counter-clockwise.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..arc.grid import BACKGROUND, MAX_SIDE, NUM_COLORS, Grid
from .errors import DslLimitError, DslRuntimeError
from .types import Cell, DslType, Obj, ObjectSet

__all__ = ["OpSpec", "Param", "OPERATORS", "get_op", "op_names", "is_assignable"]

# Guard rails for execution. A program that blows past these is rejected with a
# limit error rather than being allowed to exhaust memory.
MAX_CELLS = MAX_SIDE * MAX_SIDE
MAX_OBJECTS = 200

TypeSpec = Union[DslType, Tuple[DslType, ...]]

_NO_DEFAULT = object()


def is_assignable(src: DslType, dst: TypeSpec) -> bool:
    """Static assignability.

    ``Color`` and ``Integer`` are mutually assignable: a colour is an integer,
    and integer literals are how colours are written. Range is checked at
    runtime. Every other pair must match exactly, which is what catches the
    errors that actually matter (passing a Grid where an Object is required).
    """
    targets = dst if isinstance(dst, tuple) else (dst,)
    for t in targets:
        if src == t:
            return True
        if {src, t} == {DslType.COLOR, DslType.INTEGER}:
            return True
    return False


@dataclass(frozen=True)
class Param:
    name: str
    type: TypeSpec
    default: object = _NO_DEFAULT
    choices: Tuple[object, ...] = ()

    @property
    def required(self) -> bool:
        return self.default is _NO_DEFAULT

    def type_names(self) -> str:
        if isinstance(self.type, tuple):
            return "|".join(str(t) for t in self.type)
        return str(self.type)


@dataclass(frozen=True)
class OpSpec:
    name: str
    params: Tuple[Param, ...]
    ret: Union[DslType, Callable[[Tuple[DslType, ...]], DslType]]
    fn: Callable
    doc: str

    @property
    def arity(self) -> int:
        return len(self.params)

    @property
    def n_required(self) -> int:
        return sum(1 for p in self.params if p.required)

    def return_type(self, arg_types: Sequence[DslType]) -> DslType:
        if callable(self.ret):
            return self.ret(tuple(arg_types))
        return self.ret

    def signature(self) -> str:
        parts = []
        for p in self.params:
            s = f"{p.name}: {p.type_names()}"
            if not p.required:
                s += f" = {p.default}"
            parts.append(s)
        ret = self.ret if not callable(self.ret) else "<depends on args>"
        return f"{self.name}({', '.join(parts)}) -> {ret}"


OPERATORS: Dict[str, OpSpec] = {}


def register(
    name: str,
    params: Sequence[Param],
    ret: Union[DslType, Callable],
    doc: str,
) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        if name in OPERATORS:
            raise RuntimeError(f"operator {name} registered twice")
        OPERATORS[name] = OpSpec(name=name, params=tuple(params), ret=ret, fn=fn, doc=doc)
        return fn

    return deco


def get_op(name: str) -> Optional[OpSpec]:
    return OPERATORS.get(name)


def op_names() -> List[str]:
    return sorted(OPERATORS)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

G, O, OS, C, I, B = (
    DslType.GRID,
    DslType.OBJECT,
    DslType.OBJECT_SET,
    DslType.COLOR,
    DslType.INTEGER,
    DslType.BOOL,
)
GRID_OR_OBJ = (G, O)


def _check_grid(grid: Grid, what: str = "result") -> Grid:
    h, w = grid.shape
    if h == 0 or w == 0:
        raise DslRuntimeError(f"{what} grid is empty ({h}x{w})", code="empty_grid")
    if h > MAX_SIDE or w > MAX_SIDE:
        raise DslLimitError(f"{what} grid {h}x{w} exceeds {MAX_SIDE}x{MAX_SIDE}")
    return np.ascontiguousarray(grid, dtype=np.int8)


def _check_color(color: int) -> int:
    color = int(color)
    if not 0 <= color < NUM_COLORS:
        raise DslRuntimeError(f"colour {color} outside 0..{NUM_COLORS - 1}", code="bad_color")
    return color


def _same_shape_pad(a: Grid, b: Grid) -> Tuple[Grid, Grid]:
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.full((h, w), BACKGROUND, dtype=np.int8)
    pb = np.full((h, w), BACKGROUND, dtype=np.int8)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b
    return pa, pb


def _components(grid: Grid, conn: int, same_color: bool) -> List[Obj]:
    if conn not in (4, 8):
        raise DslRuntimeError(f"conn must be 4 or 8, got {conn}", code="bad_conn")
    h, w = grid.shape
    if conn == 4:
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    seen = np.zeros((h, w), dtype=bool)
    out: List[Obj] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BACKGROUND:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: Dict[Cell, int] = {}
            while queue:
                cr, cc = queue.popleft()
                cells[(cr, cc)] = int(grid[cr, cc])
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    val = int(grid[nr, nc])
                    if val == BACKGROUND:
                        continue
                    if same_color and val != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            out.append(Obj.from_mapping(cells, (h, w)))
            if len(out) > MAX_OBJECTS:
                raise DslLimitError(f"more than {MAX_OBJECTS} objects found")
    return out


def _obj_to_grid_arg(value) -> Grid:
    """Coerce a Grid-or-Object argument to a grid for measurement operators."""
    if isinstance(value, Obj):
        return value.to_grid()
    return value


# ---------------------------------------------------------------------------
# Grid transforms
# ---------------------------------------------------------------------------


@register("INPUT", [], G, "The task's input grid. Bound by the executor.")
def _input() -> Grid:  # pragma: no cover - executor special-cases this
    raise DslRuntimeError("INPUT must be bound by the executor", code="unbound_input")


@register("COPY", [Param("g", G)], G, "Identity on grids.")
def _copy(g: Grid) -> Grid:
    return _check_grid(g.copy())


@register("ROTATE90", [Param("g", G)], G, "Rotate 90 degrees clockwise.")
def _rot90(g: Grid) -> Grid:
    return _check_grid(np.rot90(g, k=-1))


@register("ROTATE180", [Param("g", G)], G, "Rotate 180 degrees.")
def _rot180(g: Grid) -> Grid:
    return _check_grid(np.rot90(g, k=2))


@register("ROTATE270", [Param("g", G)], G, "Rotate 270 degrees clockwise (= 90 counter-clockwise).")
def _rot270(g: Grid) -> Grid:
    return _check_grid(np.rot90(g, k=1))


@register("FLIP_H", [Param("g", G)], G, "Mirror left-right.")
def _flip_h(g: Grid) -> Grid:
    return _check_grid(np.fliplr(g))


@register("FLIP_V", [Param("g", G)], G, "Mirror up-down.")
def _flip_v(g: Grid) -> Grid:
    return _check_grid(np.flipud(g))


@register("TRANSPOSE", [Param("g", G)], G, "Swap rows and columns.")
def _transpose(g: Grid) -> Grid:
    return _check_grid(g.T)


@register(
    "CROP",
    [Param("x", GRID_OR_OBJ)],
    G,
    "Crop to content: for a Grid, the bounding box of non-background cells; "
    "for an Object, its bounding box.",
)
def _crop(x) -> Grid:
    if isinstance(x, Obj):
        if not x.cells:
            raise DslRuntimeError("cannot crop an empty object", code="empty_object")
        return _check_grid(x.to_grid())
    mask = x != BACKGROUND
    if not mask.any():
        raise DslRuntimeError("cannot crop a blank grid", code="blank_grid")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return _check_grid(x[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy())


@register(
    "PAD",
    [Param("g", G), Param("n", I), Param("color", C, default=BACKGROUND)],
    G,
    "Add an n-cell border of `color` on all four sides.",
)
def _pad(g: Grid, n: int, color: int = BACKGROUND) -> Grid:
    n = int(n)
    if n < 0:
        raise DslRuntimeError(f"pad width must be >= 0, got {n}", code="bad_pad")
    color = _check_color(color)
    return _check_grid(np.pad(g, n, mode="constant", constant_values=color))


@register(
    "TRANSLATE",
    [Param("g", G), Param("dr", I), Param("dc", I), Param("fill", C, default=BACKGROUND)],
    G,
    "Shift contents by (dr, dc), keeping the grid size. Vacated cells get `fill`.",
)
def _translate(g: Grid, dr: int, dc: int, fill: int = BACKGROUND) -> Grid:
    dr, dc, fill = int(dr), int(dc), _check_color(fill)
    h, w = g.shape
    out = np.full((h, w), fill, dtype=np.int8)
    src_r0, dst_r0 = (0, dr) if dr >= 0 else (-dr, 0)
    src_c0, dst_c0 = (0, dc) if dc >= 0 else (-dc, 0)
    nh = max(0, h - abs(dr))
    nw = max(0, w - abs(dc))
    if nh and nw:
        out[dst_r0 : dst_r0 + nh, dst_c0 : dst_c0 + nw] = g[src_r0 : src_r0 + nh, src_c0 : src_c0 + nw]
    return _check_grid(out)


@register(
    "RECOLOR",
    [Param("x", GRID_OR_OBJ), Param("color", C)],
    lambda args: args[0],
    "Set every non-background cell to `color`.",
)
def _recolor(x, color: int):
    color = _check_color(color)
    if isinstance(x, Obj):
        return x.recolored(color)
    out = x.copy()
    out[out != BACKGROUND] = color
    return _check_grid(out)


@register(
    "REPLACE_COLOR",
    [Param("g", G), Param("from_color", C), Param("to_color", C)],
    G,
    "Replace every occurrence of one colour with another.",
)
def _replace_color(g: Grid, from_color: int, to_color: int) -> Grid:
    from_color, to_color = _check_color(from_color), _check_color(to_color)
    out = g.copy()
    out[g == from_color] = to_color
    return _check_grid(out)


# ---------------------------------------------------------------------------
# Object operations
# ---------------------------------------------------------------------------


@register(
    "COMPONENTS",
    [Param("g", G), Param("conn", I, default=4, choices=(4, 8)), Param("same_color", B, default=True)],
    OS,
    "Connected components of non-background cells. `conn` is 4 or 8; when "
    "`same_color` is true a component may not span two colours.",
)
def _components_op(g: Grid, conn: int = 4, same_color: bool = True) -> ObjectSet:
    return ObjectSet.of(_components(g, int(conn), bool(same_color)))


@register(
    "OBJECTS_OF_COLOR",
    [Param("g", G), Param("color", C), Param("conn", I, default=4, choices=(4, 8))],
    OS,
    "Connected components made only of cells of the given colour.",
)
def _objects_of_color(g: Grid, color: int, conn: int = 4) -> ObjectSet:
    color = _check_color(color)
    masked = np.where(g == color, g, np.int8(BACKGROUND))
    return ObjectSet.of(_components(masked, int(conn), True))


@register("LARGEST", [Param("s", OS)], O, "The object with the most cells (ties: reading order).")
def _largest(s: ObjectSet) -> Obj:
    if s.is_empty:
        raise DslRuntimeError("LARGEST on an empty ObjectSet", code="empty_set")
    return max(s.objects, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))


@register("SMALLEST", [Param("s", OS)], O, "The object with the fewest cells (ties: reading order).")
def _smallest(s: ObjectSet) -> Obj:
    if s.is_empty:
        raise DslRuntimeError("SMALLEST on an empty ObjectSet", code="empty_set")
    return min(s.objects, key=lambda o: (o.size, o.bbox[0], o.bbox[1]))


@register(
    "FILTER_COLOR", [Param("s", OS), Param("color", C)], OS, "Objects whose dominant colour matches."
)
def _filter_color(s: ObjectSet, color: int) -> ObjectSet:
    color = _check_color(color)
    return ObjectSet.of(o for o in s.objects if o.dominant_color() == color)


@register("FILTER_SIZE", [Param("s", OS), Param("size", I)], OS, "Objects with exactly `size` cells.")
def _filter_size(s: ObjectSet, size: int) -> ObjectSet:
    size = int(size)
    return ObjectSet.of(o for o in s.objects if o.size == size)


@register(
    "FILTER_SHAPE",
    [Param("s", OS), Param("template", O)],
    OS,
    "Objects whose translated cell pattern equals the template's (colour ignored).",
)
def _filter_shape(s: ObjectSet, template: Obj) -> ObjectSet:
    want = template.normalized_shape()
    return ObjectSet.of(o for o in s.objects if o.normalized_shape() == want)


@register(
    "BOUNDING_BOX",
    [Param("o", O)],
    O,
    "The filled rectangle spanning the object, in its dominant colour.",
)
def _bounding_box(o: Obj) -> Obj:
    if not o.cells:
        raise DslRuntimeError("BOUNDING_BOX on an empty object", code="empty_object")
    t, l, b, r = o.bbox
    color = o.dominant_color()
    cells = {(row, col): color for row in range(t, b + 1) for col in range(l, r + 1)}
    return Obj.from_mapping(cells, o.grid_shape)


# ---------------------------------------------------------------------------
# Construction / composition
# ---------------------------------------------------------------------------


@register(
    "EMPTY_GRID",
    [Param("h", I), Param("w", I), Param("color", C, default=BACKGROUND)],
    G,
    "A new h x w grid filled with `color`.",
)
def _empty_grid(h: int, w: int, color: int = BACKGROUND) -> Grid:
    h, w, color = int(h), int(w), _check_color(color)
    if h <= 0 or w <= 0:
        raise DslRuntimeError(f"EMPTY_GRID needs positive dims, got {h}x{w}", code="bad_dims")
    return _check_grid(np.full((h, w), color, dtype=np.int8))


@register(
    "FILL",
    [Param("g", G), Param("o", O), Param("color", C)],
    G,
    "Paint the object's cells onto the grid in `color`. Cells outside the grid are ignored.",
)
def _fill(g: Grid, o: Obj, color: int) -> Grid:
    color = _check_color(color)
    out = g.copy()
    h, w = out.shape
    for r, c in o.cells:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color
    return _check_grid(out)


@register(
    "OUTLINE",
    [Param("o", O)],
    O,
    "The border cells of an object: those with a 4-neighbour outside the object.",
)
def _outline(o: Obj) -> Obj:
    if not o.cells:
        raise DslRuntimeError("OUTLINE on an empty object", code="empty_object")
    cmap = o.color_map
    border = {
        cell: cmap[cell]
        for cell in o.cells
        if any((cell[0] + dr, cell[1] + dc) not in o.cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    }
    return Obj.from_mapping(border, o.grid_shape)


@register(
    "OVERLAY",
    [Param("base", G), Param("top", G)],
    G,
    "Draw `top` over `base`; non-background cells of `top` win. Grids are "
    "padded to a common size at the top-left.",
)
def _overlay(base: Grid, top: Grid) -> Grid:
    a, b = _same_shape_pad(base, top)
    out = a.copy()
    mask = b != BACKGROUND
    out[mask] = b[mask]
    return _check_grid(out)


@register(
    "TILE", [Param("g", G), Param("nh", I), Param("nw", I)], G, "Repeat the grid nh times down and nw times across."
)
def _tile(g: Grid, nh: int, nw: int) -> Grid:
    nh, nw = int(nh), int(nw)
    if nh <= 0 or nw <= 0:
        raise DslRuntimeError(f"TILE counts must be positive, got {nh},{nw}", code="bad_dims")
    if g.shape[0] * nh > MAX_SIDE or g.shape[1] * nw > MAX_SIDE:
        raise DslLimitError(f"TILE would produce {g.shape[0] * nh}x{g.shape[1] * nw}")
    return _check_grid(np.tile(g, (nh, nw)))


@register("REPEAT_H", [Param("g", G), Param("n", I)], G, "Repeat the grid n times horizontally.")
def _repeat_h(g: Grid, n: int) -> Grid:
    return _tile(g, 1, n)


@register("REPEAT_V", [Param("g", G), Param("n", I)], G, "Repeat the grid n times vertically.")
def _repeat_v(g: Grid, n: int) -> Grid:
    return _tile(g, n, 1)


@register("CONCAT_H", [Param("a", G), Param("b", G)], G, "Join side by side. Heights must match.")
def _concat_h(a: Grid, b: Grid) -> Grid:
    if a.shape[0] != b.shape[0]:
        raise DslRuntimeError(
            f"CONCAT_H needs equal heights, got {a.shape[0]} and {b.shape[0]}", code="shape_mismatch"
        )
    return _check_grid(np.concatenate([a, b], axis=1))


@register("CONCAT_V", [Param("a", G), Param("b", G)], G, "Stack vertically. Widths must match.")
def _concat_v(a: Grid, b: Grid) -> Grid:
    if a.shape[1] != b.shape[1]:
        raise DslRuntimeError(
            f"CONCAT_V needs equal widths, got {a.shape[1]} and {b.shape[1]}", code="shape_mismatch"
        )
    return _check_grid(np.concatenate([a, b], axis=0))


# ---------------------------------------------------------------------------
# Relations / properties
# ---------------------------------------------------------------------------


@register("WIDTH", [Param("x", GRID_OR_OBJ)], I, "Column count of a grid, or bounding-box width of an object.")
def _width(x) -> int:
    return int(x.width) if isinstance(x, Obj) else int(x.shape[1])


@register("HEIGHT", [Param("x", GRID_OR_OBJ)], I, "Row count of a grid, or bounding-box height of an object.")
def _height(x) -> int:
    return int(x.height) if isinstance(x, Obj) else int(x.shape[0])


@register(
    "AREA",
    [Param("x", GRID_OR_OBJ)],
    I,
    "Cell count: h*w for a grid, number of occupied cells for an object.",
)
def _area(x) -> int:
    return int(x.size) if isinstance(x, Obj) else int(x.shape[0] * x.shape[1])


@register("COUNT", [Param("s", OS)], I, "Number of objects in the set.")
def _count(s: ObjectSet) -> int:
    return len(s)


@register("COLOR", [Param("o", O)], C, "Dominant colour of the object.")
def _color(o: Obj) -> int:
    return int(o.dominant_color())


@register(
    "CENTER",
    [Param("o", O)],
    O,
    "A single-cell object at the centre of the object's bounding box.",
)
def _center(o: Obj) -> Obj:
    if not o.cells:
        raise DslRuntimeError("CENTER on an empty object", code="empty_object")
    t, l, b, r = o.bbox
    return Obj.from_mapping({((t + b) // 2, (l + r) // 2): o.dominant_color()}, o.grid_shape)


@register(
    "TOUCHING",
    [Param("a", O), Param("b", O)],
    B,
    "True when the two disjoint objects have 4-adjacent cells.",
)
def _touching(a: Obj, b: Obj) -> bool:
    if a.cells & b.cells:
        return False
    return any(
        (r + dr, c + dc) in b.cells
        for r, c in a.cells
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


@register(
    "CONTAINS",
    [Param("outer", O), Param("inner", O)],
    B,
    "True when the outer object's bounding box strictly encloses the inner one's.",
)
def _contains(outer: Obj, inner: Obj) -> bool:
    if not outer.cells or not inner.cells:
        return False
    ot, ol, ob, orr = outer.bbox
    it, il, ib, ir = inner.bbox
    return ot < it and ol < il and ob > ib and orr > ir


@register(
    "ALIGNED_H",
    [Param("a", O), Param("b", O)],
    B,
    "True when the objects' row ranges overlap (they sit in the same horizontal band).",
)
def _aligned_h(a: Obj, b: Obj) -> bool:
    at, _, ab, _ = a.bbox
    bt, _, bb, _ = b.bbox
    return not (ab < bt or bb < at)


@register(
    "ALIGNED_V",
    [Param("a", O), Param("b", O)],
    B,
    "True when the objects' column ranges overlap (same vertical band).",
)
def _aligned_v(a: Obj, b: Obj) -> bool:
    _, al, _, ar = a.bbox
    _, bl, _, br = b.bbox
    return not (ar < bl or br < al)
