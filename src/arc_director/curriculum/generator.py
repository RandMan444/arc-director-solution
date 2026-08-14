"""Warm-up task generation by random rollout of the machine itself.

Before the agent ever sees ARC it trains on tasks *we* generate, and the
cleanest way to generate them is to drive the same register machine the worker
will drive: sample a few random grids, take N random legal actions, and keep
the result as a task whose inputs are the sampled grids and whose outputs are
the resulting canvases.

That construction buys three things the predecessor's synthetic generator did
not:

* **Guaranteed reachability.** The task was produced by the action space, so a
  solution exists inside it, at exactly the advertised length. A failure to
  learn is a learning failure, never an expressivity failure -- which was the
  ambiguity that made the old ARC results so hard to read.
* **Curriculum by construction.** Program length, operator subset and grid size
  are all knobs on the generator, so "two operators, geometry only, 5x5 grids"
  is a one-line stage definition.
* **A ground-truth program.** Every generated task carries the statement
  sequence that solves it, which is what an optional behaviour-cloning warm
  start would need, and what makes the curriculum debuggable by eye.

Filtering is where the quality is. A random rollout is usually degenerate --
identity, blank output, or a program whose last statement is the only live one
-- so candidates are rejected unless the output is non-trivial, differs from
the input, varies across demonstrations, and depends on every statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..arc.dataset import ArcTask, Pair
from ..arc.grid import Grid, grids_equal
from ..dsl.machine import Machine, MachineSpec
from ..dsl.synthetic import random_grid

__all__ = ["GenConfig", "GeneratedTask", "generate_task", "generate_tasks"]


@dataclass
class GenConfig:
    """Knobs for one curriculum stage's task generator."""

    n_ops: int = 1
    n_demos: int = 3
    n_test: int = 1
    min_side: int = 4
    max_side: int = 8
    n_objects: Tuple[int, int] = (1, 4)
    palette: Tuple[int, int] = (2, 4)
    #: Probability of taking a register reference rather than a literal when
    #: both are legal. High values produce chained, input-dependent programs.
    ref_bias: float = 0.75
    #: Probability of preferring the most recent register of a bank.
    recency_bias: float = 0.7
    max_action_tries: int = 40

    def grid_kwargs(self) -> Dict[str, object]:
        return {
            "min_side": self.min_side,
            "max_side": self.max_side,
            "n_objects": self.n_objects,
            "palette": self.palette,
        }


@dataclass
class GeneratedTask:
    task: ArcTask
    program: str
    n_ops: int
    ops: Tuple[str, ...]


def _sample_args(
    rng: np.random.Generator,
    spec: MachineSpec,
    arg_masks: np.ndarray,
    op_idx: int,
    cfg: GenConfig,
) -> List[int]:
    """Choose one legal argument per parameter, biased toward register refs."""
    op_spec = spec.spec_of(op_idx)
    args: List[int] = [0] * spec.max_arity
    for p_idx in range(len(op_spec.params)):
        legal = np.flatnonzero(arg_masks[op_idx, p_idx])
        if legal.size == 0:  # pragma: no cover - guarded by the op mask
            return []
        weights = np.ones(legal.size, dtype=np.float64)
        for i, a in enumerate(legal):
            entry = spec.args[int(a)]
            if entry.kind == "reg":
                weights[i] = cfg.ref_bias * (
                    cfg.recency_bias if entry.reg_index == 0 else 1.0 - cfg.recency_bias
                )
            elif entry.kind == "default":
                weights[i] = 0.3 * (1.0 - cfg.ref_bias)
            else:
                weights[i] = 1.0 - cfg.ref_bias
        weights /= weights.sum()
        args[p_idx] = int(rng.choice(legal, p=weights))
    return args


def _interesting(inputs: Sequence[Grid], outputs: Sequence[Grid]) -> bool:
    """Reject the degenerate rollouts that dominate random program space."""
    if any(o.size <= 1 for o in outputs):
        return False
    if all(grids_equal(i, o) for i, o in zip(inputs, outputs)):
        return False                                   # identity in disguise
    if any(np.unique(o).size == 1 for o in outputs):
        return False                                   # blank / single colour
    if all(grids_equal(outputs[0], o) for o in outputs[1:]):
        return False                                   # constant, ignores input
    return True


def generate_task(
    rng: np.random.Generator,
    spec: MachineSpec,
    cfg: GenConfig,
    allowed_ops: Optional[np.ndarray] = None,
) -> Optional[GeneratedTask]:
    """Generate one task, or None when this attempt produced nothing usable."""
    n_grids = cfg.n_demos + cfg.n_test
    inputs = [random_grid(rng, **cfg.grid_kwargs()) for _ in range(n_grids)]
    machine = Machine(spec, inputs)

    ops_used: List[str] = []
    for _ in range(cfg.n_ops):
        arg_masks = machine.arg_masks()
        op_mask = machine.op_mask(arg_masks)
        op_mask[spec.halt_index] = False
        if allowed_ops is not None:
            op_mask &= allowed_ops
        legal_ops = np.flatnonzero(op_mask)
        if legal_ops.size == 0:
            return None

        for _ in range(cfg.max_action_tries):
            op_idx = int(rng.choice(legal_ops))
            args = _sample_args(rng, spec, arg_masks, op_idx, cfg)
            if not args:
                continue
            before = [c.copy() for c in machine.canvases()]
            result = machine.step(op_idx, args)
            if not result.ok:
                continue
            # A statement that leaves every canvas untouched *and* produces no
            # new non-grid value is wasted; retry instead of burning a slot.
            after = machine.canvases()
            if all(grids_equal(a, b) for a, b in zip(before, after)) and result.result_type is None:
                continue
            ops_used.append(spec.op_names[op_idx])
            break
        else:
            return None

    outputs = machine.canvases()
    if machine.live_statements() != cfg.n_ops:
        return None
    if not _interesting(inputs, outputs):
        return None

    demos = [Pair(inputs[i], outputs[i]) for i in range(cfg.n_demos)]
    tests = [Pair(inputs[i], outputs[i]) for i in range(cfg.n_demos, n_grids)]
    task = ArcTask(task_id="gen", source="warmup", train=demos, test=tests)
    return GeneratedTask(
        task=task,
        program=machine.program_text(),
        n_ops=cfg.n_ops,
        ops=tuple(ops_used),
    )


def generate_tasks(
    n: int,
    spec: MachineSpec,
    cfg: GenConfig,
    *,
    seed: int = 0,
    allowed_ops: Optional[np.ndarray] = None,
    max_attempts_per_task: int = 60,
    dedup: bool = True,
) -> List[GeneratedTask]:
    """Generate ``n`` distinct tasks. Used by scripts; training generates live."""
    rng = np.random.default_rng(seed)
    out: List[GeneratedTask] = []
    seen: set[str] = set()
    budget = n * max_attempts_per_task
    attempts = 0
    while len(out) < n and attempts < budget:
        attempts += 1
        task = generate_task(rng, spec, cfg, allowed_ops)
        if task is None:
            continue
        if dedup:
            if task.program in seen:
                continue
            seen.add(task.program)
        task.task.task_id = f"gen_{len(out):06d}"
        out.append(task)
    return out
