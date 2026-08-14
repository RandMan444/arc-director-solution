"""Task sources: where an episode's task comes from, and when to move on.

The environment asks its source for a task at every reset and reports the
outcome at every episode end. That is the whole interface, which makes the
curriculum swappable: a fixed ARC list, an endless warm-up generator, or the
staged ladder that walks from one to the other.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..arc.dataset import ArcTask
from ..dsl.machine import MachineSpec
from .generator import GenConfig, generate_task
from .stages import Stage, ops_mask

__all__ = [
    "TaskSource",
    "WarmupSource",
    "ArcSource",
    "CurriculumSource",
    "load_arc_tasks",
    "filter_arc_tasks",
]


class TaskSource:
    """Base class. ``allowed_ops`` and ``max_steps`` are read by the env."""

    allowed_ops: Optional[np.ndarray] = None
    max_steps: Optional[int] = None
    name: str = "source"

    def sample(self, rng: np.random.Generator) -> ArcTask:  # pragma: no cover - abstract
        raise NotImplementedError

    def report(self, info: Dict[str, object]) -> None:
        """Called once per finished episode with the env's terminal info."""

    def stats(self) -> Dict[str, object]:
        return {"source": self.name}


# ---------------------------------------------------------------------------
# Warm-up: endless generated tasks
# ---------------------------------------------------------------------------


class WarmupSource(TaskSource):
    """Generates a fresh task per episode from the machine itself.

    Generation is cheap (milliseconds) and never repeats, so there is nothing
    to memorise -- the only way to score is to read the demonstrations. A small
    cache absorbs the occasional run of rejected candidates without stalling
    the rollout.
    """

    def __init__(
        self,
        spec: MachineSpec,
        cfg: GenConfig,
        *,
        allowed_ops: Optional[np.ndarray] = None,
        n_ops_choices: Sequence[int] = (1,),
        max_steps: Optional[int] = None,
        name: str = "warmup",
        max_tries: int = 50,
    ) -> None:
        self.spec = spec
        self.cfg = cfg
        self.allowed_ops = allowed_ops
        self.n_ops_choices = tuple(n_ops_choices)
        self.max_steps = max_steps
        self.name = name
        self.max_tries = max_tries
        self.generated = 0
        self.rejected = 0
        self.last_program: Optional[str] = None

    def sample(self, rng: np.random.Generator) -> ArcTask:
        for _ in range(self.max_tries):
            self.cfg.n_ops = int(rng.choice(np.asarray(self.n_ops_choices)))
            made = generate_task(rng, self.spec, self.cfg, self.allowed_ops)
            if made is None:
                self.rejected += 1
                continue
            self.generated += 1
            self.last_program = made.program
            made.task.task_id = f"{self.name}_{self.generated:07d}"
            return made.task
        raise RuntimeError(
            f"{self.name}: no task generated in {self.max_tries} tries; the operator "
            "subset is probably too small for the requested program length"
        )

    def stats(self) -> Dict[str, object]:
        total = self.generated + self.rejected
        return {
            "source": self.name,
            "generated": self.generated,
            "reject_rate": round(self.rejected / total, 3) if total else 0.0,
        }


# ---------------------------------------------------------------------------
# Real ARC
# ---------------------------------------------------------------------------


def load_arc_tasks(root: str | Path, split: str = "training") -> List[ArcTask]:
    """Load every task in ``<root>/<split>``."""
    directory = Path(root) / split
    if not directory.is_dir():
        raise FileNotFoundError(f"no ARC split at {directory}")
    return [ArcTask.from_file(p, source=split) for p in sorted(directory.glob("*.json"))]


def filter_arc_tasks(
    tasks: Sequence[ArcTask],
    mode: Optional[str],
    *,
    reachable_ids: Optional[Iterable[str]] = None,
    max_side: int = 12,
) -> List[ArcTask]:
    """Select the ARC subset a stage should draw from.

    ``"reachable"`` keeps only tasks a bounded search has *proved* the DSL can
    express (see ``scripts/mine_reachable.py``). Those are the tasks where a
    failure is unambiguously the agent's, which makes them the right first
    contact with real ARC. ``"small"`` is the fallback heuristic tier: same
    input/output shape and small grids.
    """
    if mode is None:
        return list(tasks)
    if mode == "reachable":
        ids = set(reachable_ids or ())
        return [t for t in tasks if t.task_id in ids or t.uid in ids]
    if mode == "small":
        out = []
        for t in tasks:
            pairs = t.demo_pairs()
            if not pairs:
                continue
            if all(i.shape == o.shape for i, o in pairs) and all(
                max(i.shape) <= max_side for i, _ in pairs
            ):
                out.append(t)
        return out
    raise ValueError(f"unknown ARC filter {mode!r}")


class ArcSource(TaskSource):
    """Uniform sampling over a fixed task list.

    Augmentation happens in the environment, so a 200-task pool still yields a
    different observation every episode.
    """

    def __init__(
        self,
        tasks: Sequence[ArcTask],
        *,
        allowed_ops: Optional[np.ndarray] = None,
        max_steps: Optional[int] = None,
        name: str = "arc",
    ) -> None:
        if not tasks:
            raise ValueError(f"{name}: empty task pool")
        self.tasks = list(tasks)
        self.allowed_ops = allowed_ops
        self.max_steps = max_steps
        self.name = name
        self.solved_ids: set[str] = set()

    def sample(self, rng: np.random.Generator) -> ArcTask:
        return self.tasks[int(rng.integers(len(self.tasks)))]

    def report(self, info: Dict[str, object]) -> None:
        if info.get("generalized") or (
            info.get("solved_demos") and info.get("solved_heldout", True)
        ):
            self.solved_ids.add(str(info.get("task_id", "")).split("#")[0])

    def stats(self) -> Dict[str, object]:
        return {
            "source": self.name,
            "pool": len(self.tasks),
            "tasks_solved": len(self.solved_ids),
        }


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


class CurriculumSource(TaskSource):
    """Walks the stage ladder, promoting on a rolling generalisation rate.

    Promotion uses the held-out demonstration when the stage provides one. A
    stage clears only after ``min_episodes`` *and* a rolling rate above
    ``promote_at``, so a lucky streak early on cannot skip a rung.
    """

    def __init__(
        self,
        spec: MachineSpec,
        stages: Sequence[Stage],
        *,
        arc_tasks: Optional[Sequence[ArcTask]] = None,
        reachable_ids: Optional[Iterable[str]] = None,
        start_stage: int = 0,
        auto_promote: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("a curriculum needs at least one stage")
        self.spec = spec
        self.stages = list(stages)
        self.arc_tasks = list(arc_tasks or [])
        self.reachable_ids = list(reachable_ids or [])
        self.auto_promote = auto_promote
        self.index = int(start_stage)
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.recent: deque = deque(maxlen=self.stages[self.index].promote_window)
        self.history: List[Dict[str, object]] = []
        self._build()

    # -- stage plumbing --------------------------------------------------
    def _build(self) -> None:
        stage = self.stages[self.index]
        mask = ops_mask(self.spec, stage.op_names())
        if stage.kind == "warmup":
            cfg = GenConfig(
                n_ops=stage.n_ops[0],
                n_demos=stage.n_demos,
                n_test=1,
                min_side=stage.min_side,
                max_side=stage.max_side,
            )
            self.inner: TaskSource = WarmupSource(
                self.spec,
                cfg,
                allowed_ops=mask,
                n_ops_choices=stage.n_ops,
                max_steps=stage.max_steps,
                name=stage.name,
            )
        elif stage.kind == "arc":
            pool = filter_arc_tasks(
                self.arc_tasks, stage.arc_filter, reachable_ids=self.reachable_ids
            )
            if not pool:
                raise ValueError(
                    f"stage {stage.name!r} selected no ARC tasks with filter "
                    f"{stage.arc_filter!r}; run scripts/mine_reachable.py or change the ladder"
                )
            self.inner = ArcSource(
                pool, allowed_ops=mask, max_steps=stage.max_steps, name=stage.name
            )
        else:
            raise ValueError(f"unknown stage kind {stage.kind!r}")
        self.allowed_ops = self.inner.allowed_ops
        self.max_steps = self.inner.max_steps
        self.name = stage.name
        self.recent = deque(maxlen=stage.promote_window)
        self.episodes_in_stage = 0

    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    @property
    def rate(self) -> float:
        return float(np.mean(self.recent)) if self.recent else 0.0

    # -- TaskSource ------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> ArcTask:
        return self.inner.sample(rng)

    def report(self, info: Dict[str, object]) -> None:
        self.inner.report(info)
        # Promotion is earned on the held-out demonstration whenever there is
        # one. Fitting only the demonstrations that were scored is exactly the
        # failure mode a curriculum must not reward with an easier next rung.
        if info.get("has_heldout"):
            solved = bool(info.get("generalized", False))
        else:
            solved = bool(info.get("solved_demos"))
        self.recent.append(1.0 if solved else 0.0)
        self.episodes_in_stage += 1
        self.total_episodes += 1
        if self.auto_promote:
            self.maybe_promote()

    def maybe_promote(self) -> bool:
        stage = self.stage
        if self.index + 1 >= len(self.stages):
            return False
        if self.episodes_in_stage < stage.min_episodes:
            return False
        if len(self.recent) < stage.promote_window:
            return False
        if self.rate < stage.promote_at:
            return False
        self.history.append(
            {
                "stage": stage.name,
                "episodes": self.episodes_in_stage,
                "rate": round(self.rate, 4),
                "at_total_episode": self.total_episodes,
            }
        )
        self.index += 1
        self._build()
        return True

    def stats(self) -> Dict[str, object]:
        out = dict(self.inner.stats())
        out.update(
            stage=self.stage.name,
            stage_index=self.index,
            stage_episodes=self.episodes_in_stage,
            stage_rate=round(self.rate, 4),
            total_episodes=self.total_episodes,
        )
        return out

    def state_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "episodes_in_stage": self.episodes_in_stage,
            "total_episodes": self.total_episodes,
            "history": self.history,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.index = int(state.get("index", 0))
        self.history = list(state.get("history", []))
        self._build()
        self.episodes_in_stage = int(state.get("episodes_in_stage", 0))
        self.total_episodes = int(state.get("total_episodes", 0))
