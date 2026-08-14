"""Summarise a training log.

    python scripts/inspect_run.py runs/warmup_small [--rows 15]

`gen` is the number that matters: exactness on the held-out demonstration, the
one a program cannot reach by fitting the demonstrations it was scored on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--rows", type=int, default=15)
    args = parser.parse_args()

    path = Path(args.run_dir) / "train_log.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit(f"{path} is empty")

    print(
        f"{'steps':>9} {'stage':<17}{'solve':>6}{'gen':>7}{'ret':>7}"
        f"{'goal_r':>8}{'H_w':>6}{'H_m':>6}{'halt':>6}{'stmts':>7}{'recon':>7}{'sps':>6}"
    )
    step = max(1, len(rows) // args.rows)
    for r in rows[::step] + [rows[-1]]:
        print(
            f"{r['env_steps']:>9,} {str(r.get('env/stage', '?')):<17}"
            f"{r.get('solve_rate', 0):>6.3f}{r.get('generalize_rate', 0):>7.3f}"
            f"{r.get('mean_return', 0):>+7.2f}{r['reward/goal_mean']:>+8.3f}"
            f"{r['policy/worker_entropy']:>6.2f}{r['policy/manager_entropy']:>6.2f}"
            f"{r.get('halt_rate', 0):>6.2f}{r.get('mean_statements', 0):>7.1f}"
            f"{r.get('goal_ae/recon', 0):>7.3f}{r['sps']:>6.0f}"
        )
    last = rows[-1]
    print(
        f"\n{last['episodes']:,} episodes, {last['env_steps']:,} env steps, "
        f"stage {last.get('env/stage')} "
        f"({last.get('env/stage_episodes', 0):,} episodes in stage, "
        f"rate {last.get('env/stage_rate', 0):.3f})"
    )


if __name__ == "__main__":
    main()
