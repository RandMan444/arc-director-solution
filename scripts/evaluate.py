"""Evaluate a checkpoint on ARC (or on generated warm-up tasks).

    python scripts/evaluate.py --config configs/full.yaml \
        --checkpoint runs/full/checkpoint.pt --split evaluation --attempts 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_director.config import build_spec, load_config  # noqa: E402
from arc_director.curriculum.generator import GenConfig, generate_tasks  # noqa: E402
from arc_director.curriculum.sources import load_arc_tasks  # noqa: E402
from arc_director.env.task_env import EnvConfig  # noqa: E402
from arc_director.models.agent import AgentConfig, DirectorAgent  # noqa: E402
from arc_director.train.evaluate import evaluate_tasks  # noqa: E402
from arc_director.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="evaluation", help="ARC split, or 'warmup'")
    parser.add_argument("--arc-root", default=None)
    parser.add_argument("--attempts", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid-side", type=int, default=None, help="override agent.grid_side")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    spec = build_spec(cfg)

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent_cfg = AgentConfig(**blob["agent_config"])
    if args.grid_side:
        # grid_side only controls how much of the padded grid is looked at, so a
        # checkpoint trained on small warm-up grids can be evaluated at 30.
        agent_cfg.grid_side = args.grid_side
    agent = DirectorAgent(agent_cfg, spec)
    agent.load_state_dict(blob["agent"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device).eval()

    env_cfg = EnvConfig(**cfg.get("env", {}))
    if args.split == "warmup":
        made = generate_tasks(args.limit or 100, spec, GenConfig(n_ops=2), seed=args.seed)
        tasks = [m.task for m in made]
    else:
        root_value = args.arc_root or cfg["curriculum"]["arc_root"]
        roots = (
            [part.strip() for part in root_value.split(",")]
            if isinstance(root_value, str)
            else list(root_value)
        )
        tasks, seen = [], set()
        for root in roots:
            source = Path(root).name.lower() or "arc"
            for task in load_arc_tasks(root, args.split, source=source):
                key = (task.source, task.task_id)
                if key not in seen:
                    seen.add(key)
                    tasks.append(task)
        if args.limit:
            tasks = tasks[: args.limit]

    print(f"evaluating {len(tasks)} tasks, {args.attempts} attempts each, on {device}")
    summary, results = evaluate_tasks(
        agent, spec, tasks, env_cfg, device,
        n_attempts=args.attempts, temperature=args.temperature, seed=args.seed, verbose=True,
    )

    print("\n" + json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {"summary": summary, "results": [r.as_dict() for r in results]}, indent=2
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
