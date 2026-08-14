"""Print generated warm-up tasks so the curriculum can be checked by eye.

    python scripts/show_warmup.py --stage w1_geometry_2 --n 5

A curriculum you have never looked at is a curriculum you do not know the
difficulty of. This renders the grids as digits, next to the program that
produced them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_director.arc.grid import format_grid  # noqa: E402
from arc_director.curriculum.generator import GenConfig, generate_tasks  # noqa: E402
from arc_director.curriculum.stages import DEFAULT_LADDER, ops_mask  # noqa: E402
from arc_director.dsl.machine import MachineSpec  # noqa: E402


def side_by_side(a: np.ndarray, b: np.ndarray, gap: str = "   ->   ") -> str:
    left, right = format_grid(a).splitlines(), format_grid(b).splitlines()
    width = max(len(r) for r in left)
    height = max(len(left), len(right))
    left += [""] * (height - len(left))
    right += [""] * (height - len(right))
    return "\n".join(f"{l:<{width}}{gap}{r}" for l, r in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="w1_geometry_2")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="list stage names and exit")
    args = parser.parse_args()

    if args.list:
        for stage in DEFAULT_LADDER:
            print(f"{stage.name:20s} kind={stage.kind:7s} n_ops={stage.n_ops} groups={stage.groups}")
        return

    stage = next((s for s in DEFAULT_LADDER if s.name == args.stage), None)
    if stage is None or stage.kind != "warmup":
        raise SystemExit(f"no warm-up stage named {args.stage!r} (try --list)")

    spec = MachineSpec.build()
    cfg = GenConfig(
        n_ops=stage.n_ops[-1],
        n_demos=stage.n_demos,
        min_side=stage.min_side,
        max_side=stage.max_side,
    )
    tasks = generate_tasks(
        args.n, spec, cfg, seed=args.seed, allowed_ops=ops_mask(spec, stage.op_names())
    )
    print(f"stage {stage.name}: {len(tasks)} tasks, {stage.n_ops[-1]} operators each\n")
    for made in tasks:
        print("=" * 60)
        print(made.program)
        for pair in made.task.train:
            print("-" * 30)
            print(side_by_side(pair.input, pair.output))
        print()


if __name__ == "__main__":
    main()
