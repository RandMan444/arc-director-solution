"""Start or resume a Director run.

    python scripts/train.py --config configs/warmup_small.yaml
    python scripts/train.py --config configs/full.yaml --resume runs/warmup_small/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc_director.config import build, load_config  # noqa: E402
from arc_director.utils.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="override the config's run_dir")
    parser.add_argument("--resume", default=None, help="checkpoint to load before training")
    parser.add_argument("--steps", type=int, default=None, help="override total env steps")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = parser.parse_args()

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

    try:
        trainer.train(args.steps)
    except KeyboardInterrupt:
        print("\ninterrupted; saving")
    finally:
        trainer.save(trainer.run_dir / "checkpoint.pt")
        print(f"saved to {trainer.run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
