"""Find the ARC tasks this action space provably covers.

    python scripts/mine_reachable.py --out data/reachable.json --depth 3 --budget 6

Writes ``{"task_ids": [...], "programs": {...}}``. The first ARC rung of the
curriculum draws from that list, so a failure there is the agent's failure and
not the language's.

A hit is proof. A miss is not: the search is a bounded randomised beam, and the
report says so per task.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_director.curriculum.sources import load_arc_tasks  # noqa: E402
from arc_director.dsl.machine import MachineSpec  # noqa: E402
from arc_director.dsl.search import SearchConfig, search_task  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc-root", default="data/arc2",
        help="one root, or several comma-separated (tasks are de-duplicated by id)",
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--out", default="data/reachable.json")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--beam", type=int, default=64)
    parser.add_argument("--per-op", type=int, default=4, help="children per operator per node")
    parser.add_argument("--budget", type=float, default=6.0, help="seconds per task")
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = MachineSpec.build()
    tasks, seen = [], set()
    for root in args.arc_root.split(","):
        for task in load_arc_tasks(root.strip(), args.split):
            if task.task_id in seen:
                continue
            seen.add(task.task_id)
            tasks.append(task)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"searching {len(tasks)} tasks from {args.arc_root} [{args.split}]")

    cfg = SearchConfig(
        max_depth=args.depth,
        beam_width=args.beam,
        per_op_samples=args.per_op,
        timeout_s=args.budget,
        seed=args.seed,
    )

    solved, test_exact, programs, rows = [], [], {}, []
    start = time.time()
    for i, task in enumerate(tasks, 1):
        result = search_task(spec, task, cfg)
        rows.append(result.as_dict())
        if result.solved:
            solved.append(task.task_id)
            programs[task.task_id] = result.program
            if result.test_exact:
                test_exact.append(task.task_id)
        if i % 25 == 0 or i == len(tasks):
            print(
                f"  {i}/{len(tasks)}  demo-exact={len(solved)}  test-exact={len(test_exact)}  "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": f"{args.arc_root}/{args.split}",
                "search": vars(args),
                "n_tasks": len(tasks),
                "task_ids": sorted(test_exact),
                "demo_exact_ids": sorted(solved),
                "programs": programs,
                "detail": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\ndemonstration-exact: {len(solved)}/{len(tasks)} "
        f"({len(solved)/max(1,len(tasks)):.2%})\n"
        f"test-exact (certified): {len(test_exact)}/{len(tasks)} "
        f"({len(test_exact)/max(1,len(tasks)):.2%})\n"
        f"wrote {out}"
    )


if __name__ == "__main__":
    main()
