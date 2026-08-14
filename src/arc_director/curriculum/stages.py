"""Operator groups and the default curriculum ladder.

The action space is fixed for a whole run -- the policy's heads cannot change
width halfway through -- so a stage restricts operators with a *mask over the
full operator list*, never by building a smaller machine spec. Masked-out
operators keep their action indices and their embeddings; they simply cannot be
selected until the stage that unlocks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dsl.machine import HALT, MachineSpec

__all__ = ["OP_GROUPS", "group_ops", "Stage", "DEFAULT_LADDER", "ops_mask"]


#: Named operator bundles. A stage names groups; the ladder unlocks groups.
OP_GROUPS: Dict[str, Tuple[str, ...]] = {
    "geometry": (
        "ROTATE90", "ROTATE180", "ROTATE270", "FLIP_H", "FLIP_V", "TRANSPOSE", "COPY",
    ),
    "color": (
        "RECOLOR", "REPLACE_COLOR", "MAIN_COLOR", "MOST_COMMON_COLOR", "RAREST_COLOR",
        "COLOR_COUNT", "PALETTE_SIZE",
    ),
    "shape": (
        "CROP", "PAD", "TRIM", "COMPRESS", "TILE", "REPEAT_H", "REPEAT_V",
        "CONCAT_H", "CONCAT_V", "MIRROR_CONCAT", "UPSCALE", "DOWNSCALE",
        "PART", "SUBGRID", "EMPTY_GRID", "CANVAS_LIKE",
    ),
    "objects": (
        "COMPONENTS", "ALL_OBJECTS", "OBJECTS_OF_COLOR", "LARGEST", "SMALLEST",
        "NTH", "SORT_BY_SIZE", "MERGE", "COUNT", "FILTER_COLOR", "FILTER_SIZE",
        "FILTER_SHAPE", "FILTER_LARGER", "FILTER_SMALLER", "EXCLUDE_COLOR",
        "ODD_ONE_OUT", "MOST_COMMON_SHAPE", "BOUNDING_BOX", "OUTLINE", "COLOR",
        "FOREGROUND", "MASK_OF_COLOR", "OBJ_GRID",
    ),
    "paint": (
        "FILL", "STAMP", "ERASE", "PAINT_SET", "STAMP_SET", "ERASE_SET", "OVERLAY",
    ),
    "spatial": (
        "TRANSLATE", "TRANSLATE_OBJ", "MOVE_TO", "GRAVITY", "RAY", "SYMMETRIZE",
        "FRAME", "CENTER", "CELLWISE",
    ),
    "algebra": (
        "UNION", "INTERSECT", "DIFFERENCE", "MAP_RECOLOR", "MAP_MOVE", "MAP_BOX",
        "MAP_OUTLINE",
    ),
    "arith": (
        "ADD", "SUB", "MUL", "DIV", "MAX_INT", "MIN_INT", "HEIGHT", "WIDTH", "AREA",
        "EQ", "LT", "GT", "NOT", "AND", "OR", "IF_GRID", "IF_INT",
        "TOUCHING", "CONTAINS", "ALIGNED_H", "ALIGNED_V",
    ),
}


def group_ops(*groups: str) -> Tuple[str, ...]:
    """The union of several operator groups, always including HALT."""
    out: List[str] = [HALT]
    for name in groups:
        if name not in OP_GROUPS:
            raise KeyError(f"unknown operator group {name!r}; have {sorted(OP_GROUPS)}")
        out.extend(OP_GROUPS[name])
    return tuple(dict.fromkeys(out))


def ops_mask(spec: MachineSpec, names: Optional[Sequence[str]]) -> Optional[np.ndarray]:
    """Boolean mask over ``spec.op_names``. ``None`` means everything is legal."""
    if names is None:
        return None
    mask = np.zeros(spec.n_ops, dtype=bool)
    for name in names:
        idx = spec.op_index.get(name)
        if idx is not None:
            mask[idx] = True
    mask[spec.halt_index] = True
    return mask


@dataclass
class Stage:
    """One rung of the curriculum.

    ``promote_at`` is measured on the *held-out* demonstration when one exists,
    so a stage is only cleared by programs that generalise, not by programs
    that happened to fit the demonstrations they were scored on.

    It is measured on the **training** distribution -- full sampling
    temperature, with an entropy bonus actively pushing the policy off its
    argmax -- so it sits well below what the same policy does greedily. A rung-0
    checkpoint logging 0.60 during training scored 0.93 at temperature 0.5 and
    1.00 at 0.25 on fresh tasks. Thresholds are set against the training rate
    for that reason; read them as "clearly learned", not as accuracy.
    """

    name: str
    kind: str                      # "warmup" | "arc"
    groups: Optional[Tuple[str, ...]] = None   # None = every operator
    n_ops: Tuple[int, ...] = (1,)
    min_side: int = 4
    max_side: int = 8
    n_demos: int = 3
    max_steps: int = 6
    promote_at: float = 0.7
    promote_window: int = 300
    min_episodes: int = 600
    arc_filter: Optional[str] = None  # for ARC stages: "reachable" | "small" | None

    def op_names(self) -> Optional[Tuple[str, ...]]:
        return None if self.groups is None else group_ops(*self.groups)


#: The default ladder: three warm-up rungs of pure geometry and colour, then
#: shape and object mechanics, then everything, then real ARC. Sizes stay small
#: until the last two rungs so early episodes are cheap and the credit
#: assignment problem is short.
DEFAULT_LADDER: Tuple[Stage, ...] = (
    Stage(
        name="w0_geometry_1",
        kind="warmup",
        groups=("geometry",),
        n_ops=(1,),
        min_side=4, max_side=6, n_demos=3, max_steps=6,
        promote_at=0.55, min_episodes=400,
    ),
    Stage(
        name="w1_geometry_2",
        kind="warmup",
        groups=("geometry", "color"),
        n_ops=(1, 2),
        min_side=4, max_side=7, n_demos=3, max_steps=6,
        promote_at=0.45, min_episodes=800,
    ),
    Stage(
        name="w2_shape",
        kind="warmup",
        groups=("geometry", "color", "shape"),
        n_ops=(1, 2),
        min_side=4, max_side=8, n_demos=3, max_steps=8,
        promote_at=0.35, min_episodes=1200,
    ),
    Stage(
        name="w3_objects",
        kind="warmup",
        groups=("geometry", "color", "shape", "objects", "paint"),
        n_ops=(2, 3),
        min_side=5, max_side=10, n_demos=3, max_steps=10,
        promote_at=0.30, min_episodes=2000,
    ),
    Stage(
        name="w4_full",
        kind="warmup",
        groups=None,
        n_ops=(2, 3, 4),
        min_side=5, max_side=12, n_demos=3, max_steps=12,
        promote_at=0.25, min_episodes=3000,
    ),
    Stage(
        name="a0_arc_reachable",
        kind="arc",
        groups=None,
        max_steps=12,
        promote_at=0.20, min_episodes=4000,
        arc_filter="reachable",
    ),
    Stage(
        name="a1_arc_small",
        kind="arc",
        groups=None,
        max_steps=14,
        promote_at=0.15, min_episodes=8000,
        arc_filter="small",
    ),
    Stage(
        name="a2_arc_all",
        kind="arc",
        groups=None,
        max_steps=16,
        promote_at=1.01,          # terminal rung: never promotes
        min_episodes=10 ** 9,
        arc_filter=None,
    ),
)
