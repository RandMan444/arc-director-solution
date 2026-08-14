"""One-click Director pipeline: generated DSL programs, then ARC-1/ARC-2.

The small-grid warm-up is intentionally a separate process.  It is much
cheaper than encoding every generated 4x4 grid on a 30x30 canvas.  When the
warm-up budget completes, its shape-compatible checkpoint is handed to the
full curriculum, which continues through the certified and then general ARC
rungs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_director.config import load_config  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _checkpoint(config: dict[str, Any]) -> Path:
    return _path(config.get("run_dir", "runs/dev")) / "checkpoint.pt"


def _run(script: str, args: Sequence[str]) -> int:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"[launch] {' '.join(command)}", flush=True)
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


def _dashboard_args(
    port: int, *, title: str, phase: str, index: int, total: int
) -> list[str]:
    return [
        "--dashboard-port",
        str(port),
        "--dashboard-title",
        title,
        "--dashboard-phase",
        phase,
        "--dashboard-phase-index",
        str(index),
        "--dashboard-phase-total",
        str(total),
    ]


def _arc_roots(config: dict[str, Any]) -> list[Path]:
    raw = config.get("curriculum", {}).get("arc_root", "data/arc2")
    values = [raw] if isinstance(raw, str) else list(raw)
    return [_path(value) for value in values]


def _prepare_arc_inputs(config: dict[str, Any]) -> int:
    """Make local datasets and reachability certificates before ARC starts."""
    roots = _arc_roots(config)
    missing = [
        root
        for root in roots
        if not all((root / split).is_dir() for split in ("training", "evaluation"))
    ]
    if missing:
        print(
            "[launch] ARC data is not in this checkout; looking for local copies "
            "before attempting a download",
            flush=True,
        )
        result = _run("prepare_arc_data.py", [])
        if result:
            return result
        missing = [
            root
            for root in roots
            if not all((root / split).is_dir() for split in ("training", "evaluation"))
        ]
        if missing:
            raise SystemExit(
                "ARC data preparation completed but these roots are still missing: "
                + ", ".join(str(path) for path in missing)
            )

    reachable_value = config.get("curriculum", {}).get("reachable_file")
    if not reachable_value:
        return 0
    reachable = _path(reachable_value)
    if reachable.is_file():
        try:
            if json.loads(reachable.read_text(encoding="utf-8")).get("task_ids"):
                return 0
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    print(
        "[launch] mining short DSL programs for the certified first ARC rung; "
        "this is a one-time, CPU-heavy preparation step",
        flush=True,
    )
    result = _run(
        "mine_reachable.py",
        [
            "--arc-root",
            ",".join(str(path) for path in roots),
            "--out",
            str(reachable),
            "--budget",
            "6",
        ],
    )
    if result:
        return result
    if not reachable.is_file():
        raise SystemExit(f"reachability mining completed without writing {reachable}")
    report = json.loads(reachable.read_text(encoding="utf-8"))
    if not report.get("task_ids"):
        raise SystemExit(
            "reachability mining found no test-exact DSL programs; refusing to "
            "start an empty certified ARC rung"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-config", default="configs/warmup_small.yaml")
    parser.add_argument("--arc-config", default="configs/full.yaml")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--fresh", action="store_true", help="refuse to overwrite checkpoints")
    start.add_argument("--resume", action="store_true", help="continue the furthest checkpoint")
    parser.add_argument("--dashboard-port", type=int, default=8321)
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="stop after generated-program warm-up",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    warmup_path = _path(args.warmup_config)
    arc_path = _path(args.arc_config)
    warmup_config = load_config(warmup_path)
    arc_config = load_config(arc_path)
    warmup_checkpoint = _checkpoint(warmup_config)
    arc_checkpoint = _checkpoint(arc_config)
    title = "ARC Director - DSL program synthesis"

    if args.fresh and (warmup_checkpoint.exists() or arc_checkpoint.exists()):
        existing = arc_checkpoint if arc_checkpoint.exists() else warmup_checkpoint
        raise SystemExit(
            f"fresh launch refused to overwrite {existing}; choose the RESUME launch button"
        )

    # Resume the furthest phase.  An ARC checkpoint already contains the
    # warm-started policy, optimizer, and curriculum state.
    if args.resume and arc_checkpoint.exists():
        if args.warmup_only:
            raise SystemExit("the ARC phase already exists; resume it without --warmup-only")
        preparation = _prepare_arc_inputs(arc_config)
        if preparation:
            return preparation
        return _run(
            "train.py",
            [
                "--config",
                str(arc_path),
                "--resume",
                str(arc_checkpoint),
                *_dashboard_args(
                    args.dashboard_port,
                    title=title,
                    phase="ARC-1 + ARC-2 curriculum",
                    index=2,
                    total=2,
                ),
            ],
        )

    if args.resume and not warmup_checkpoint.exists():
        raise SystemExit(
            "no checkpoint is available to resume; choose the FRESH launch button"
        )

    warmup_args = ["--config", str(warmup_path)]
    if args.resume:
        warmup_args.extend(["--resume", str(warmup_checkpoint)])
    warmup_args.extend(
        _dashboard_args(
            args.dashboard_port,
            title=title,
            phase="Self-generated DSL programs",
            index=1,
            total=2,
        )
    )
    result = _run("train.py", warmup_args)
    if result:
        return result
    if not warmup_checkpoint.is_file():
        raise SystemExit(
            f"warm-up exited successfully but did not write {warmup_checkpoint}"
        )
    if args.warmup_only:
        print(
            f"[launch] generated-program warm-up complete: {warmup_checkpoint}",
            flush=True,
        )
        return 0

    preparation = _prepare_arc_inputs(arc_config)
    if preparation:
        return preparation
    return _run(
        "train.py",
        [
            "--config",
            str(arc_path),
            "--resume",
            str(warmup_checkpoint),
            *_dashboard_args(
                args.dashboard_port,
                title=title,
                phase="ARC-1 + ARC-2 curriculum",
                index=2,
                total=2,
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
