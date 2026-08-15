"""Start or resume a Director run.

    python scripts/train.py --config configs/warmup_small.yaml
    python scripts/train.py --config configs/full.yaml \
        --resume runs/director_proper_warmup/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_director.config import build, load_config  # noqa: E402
from arc_director.dashboard import DashboardServer  # noqa: E402
from arc_director.utils.seed import set_seed  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="override the config's run_dir")
    parser.add_argument("--resume", default=None, help="checkpoint to load before training")
    parser.add_argument("--steps", type=int, default=None, help="override total env steps")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="serve a live dashboard on this port (0 chooses a free port)",
    )
    parser.add_argument("--dashboard-title", default="ARC Director")
    parser.add_argument("--dashboard-phase", default="Training")
    parser.add_argument("--dashboard-phase-index", type=int, default=1)
    parser.add_argument("--dashboard-phase-total", type=int, default=1)
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    set_seed(int(cfg.get("seed", 0)))

    spec, env, agent, trainer = build(cfg, run_dir=args.run_dir)
    n_params = sum(p.numel() for p in agent.parameters())
    print(
        f"machine: {spec.n_ops} operators, {spec.n_args} argument entries, "
        f"{spec.n_registers} registers"
    )
    print(f"agent:   {n_params/1e6:.2f}M parameters on {trainer.device}")
    print(f"stage:   {env.source.stats()}")

    (trainer.run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if args.resume:
        trainer.load(Path(args.resume))
        print(f"resumed from {args.resume} at update {trainer.updates}")

    dashboard = DashboardServer(
        trainer.run_dir,
        args.dashboard_title,
        port=args.dashboard_port,
        phase=args.dashboard_phase,
        phase_index=args.dashboard_phase_index,
        phase_total=args.dashboard_phase_total,
    )
    status = "finished"
    exit_code = 0
    try:
        dashboard.update(status="training")
        trainer.train(args.steps)
    except KeyboardInterrupt:
        status = "interrupted"
        exit_code = 130
        print("\ninterrupted; saving")
    except BaseException:
        status = "failed"
        raise
    finally:
        trainer.save(trainer.run_dir / "checkpoint.pt")
        dashboard.update(
            status=status,
            checkpoint=str(trainer.run_dir / "checkpoint.pt"),
            env_steps=trainer.env_steps,
            updates=trainer.updates,
        )
        dashboard.close()
        print(f"saved to {trainer.run_dir / 'checkpoint.pt'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
