"""Evaluation under the ARC protocol.

The rule ARC actually scores by: the demonstrations are all you may look at
when choosing an answer, and you get two attempts. That shapes this module.

    1. Run ``n_attempts`` episodes on the task, each with its own random
       rule-preserving augmentation.
    2. Keep only the attempts whose program is exact on *every* demonstration.
       That filter uses no test information at all.
    3. Map each surviving prediction back out of its augmentation and vote.
       Agreement across augmentations is evidence of a rule rather than a
       coincidence, which is why test-time augmentation is worth the compute.
    4. Score the top two distinct answers, reporting attempt 1 and attempt 2
       separately.

``pass@n`` is also reported: whether *any* attempt was exact on the test pair.
It is a useful learning signal and an honest upper bound on what better
candidate selection could buy, but it is not an ARC score -- it peeks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..arc.dataset import ArcTask
from ..arc.grid import Grid, grids_equal
from ..dsl.machine import MachineSpec
from ..env.task_env import EnvConfig
from ..env.vec import VecProgramEnv
from ..models.agent import DirectorAgent
from .rollout import to_tensors

__all__ = ["TaskResult", "evaluate_task", "evaluate_tasks", "SingleTaskSource"]


class SingleTaskSource:
    """A source that always returns the same task; used only for evaluation."""

    allowed_ops = None
    max_steps: Optional[int] = None
    name = "eval"

    def __init__(self, task: ArcTask, max_steps: Optional[int] = None) -> None:
        self.task = task
        self.max_steps = max_steps

    def sample(self, rng: np.random.Generator) -> ArcTask:
        return self.task

    def report(self, info: Dict[str, object]) -> None:
        return

    def stats(self) -> Dict[str, object]:
        return {"source": "eval", "task": self.task.task_id}


@dataclass
class TaskResult:
    task_id: str
    source: str
    attempts: int
    demo_fit: int                 # attempts exact on every demonstration
    exact_at_1: bool
    exact_at_2: bool
    pass_at_n: bool
    votes: int                    # size of the winning vote block
    program: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "attempts": self.attempts,
            "demo_fit": self.demo_fit,
            "exact_at_1": self.exact_at_1,
            "exact_at_2": self.exact_at_2,
            "pass_at_n": self.pass_at_n,
            "votes": self.votes,
            "program": self.program,
        }


def _key(grid: Grid) -> str:
    return f"{grid.shape[0]}x{grid.shape[1]}:" + "".join(str(int(v)) for v in grid.ravel())


@torch.no_grad()
def evaluate_task(
    agent: DirectorAgent,
    spec: MachineSpec,
    task: ArcTask,
    env_config: EnvConfig,
    device: torch.device,
    *,
    n_attempts: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
) -> TaskResult:
    """Run ``n_attempts`` augmented attempts at one task and score them.

    The policy's sampling is seeded from ``seed`` and the global torch RNG is
    put back afterwards, so an evaluation is reproducible and running one does
    not shift the training run's random stream.
    """
    cfg = EnvConfig(**{**env_config.__dict__, "holdout_demo": False, "report_answer": True})
    source = SingleTaskSource(task, max_steps=cfg.max_steps)
    env = VecProgramEnv(n_attempts, spec, source, cfg, seed=seed)

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        return _run_attempts(agent, spec, task, env, cfg, device, n_attempts, temperature)
    finally:
        torch.random.set_rng_state(rng_state)


def _run_attempts(
    agent: DirectorAgent,
    spec: MachineSpec,
    task: ArcTask,
    env: VecProgramEnv,
    cfg: EnvConfig,
    device: torch.device,
    n_attempts: int,
    temperature: float,
) -> TaskResult:

    obs = to_tensors(env.reset(), device)
    state = agent.initial_state(n_attempts, device)
    feedback = torch.zeros(n_attempts, 2, device=device)

    finished: List[dict] = [None] * n_attempts
    for _ in range(cfg.max_steps):
        out = agent.step(obs, state, feedback, temperature=temperature)
        next_obs, reward, done, infos = env.step(
            out.op.cpu().numpy(), out.args.cpu().numpy()
        )
        for i, (d, info) in enumerate(zip(done, infos)):
            if d and finished[i] is None:
                finished[i] = info
        done_t = torch.as_tensor(done, dtype=torch.float32, device=device)
        error = torch.as_tensor(
            np.array([float(i.get("error_code") is not None) for i in infos]),
            dtype=torch.float32, device=device,
        )
        feedback = torch.stack(
            [torch.as_tensor(reward, dtype=torch.float32, device=device), error], dim=-1
        ) * (1.0 - done_t).unsqueeze(-1)
        state = out.next_state.reset_where(done_t)
        obs = to_tensors(next_obs, device)
        if all(f is not None for f in finished):
            break

    attempts = [f for f in finished if f is not None]
    truth = [p.output for p in task.test if p.output is not None]
    target = truth[0] if truth else None

    fitting = [a for a in attempts if a.get("solved_demos") and a.get("answer") is not None]
    counts: Counter = Counter()
    by_key: Dict[str, Tuple[Grid, str]] = {}
    for a in fitting:
        answer = a["answer"]
        key = _key(answer)
        counts[key] += 1
        by_key.setdefault(key, (answer, a.get("program", "")))

    ranked = [by_key[k] for k, _ in counts.most_common(2)]
    exact_1 = bool(target is not None and ranked and grids_equal(ranked[0][0], target))
    exact_2 = bool(
        exact_1
        or (target is not None and len(ranked) > 1 and grids_equal(ranked[1][0], target))
    )
    pass_n = bool(
        target is not None
        and any(
            a.get("answer") is not None and grids_equal(a["answer"], target)
            for a in attempts
        )
    )

    return TaskResult(
        task_id=task.task_id,
        source=task.source,
        attempts=len(attempts),
        demo_fit=len(fitting),
        exact_at_1=exact_1,
        exact_at_2=exact_2,
        pass_at_n=pass_n,
        votes=counts.most_common(1)[0][1] if counts else 0,
        program=ranked[0][1] if ranked else None,
    )


def evaluate_tasks(
    agent: DirectorAgent,
    spec: MachineSpec,
    tasks: Sequence[ArcTask],
    env_config: EnvConfig,
    device: torch.device,
    *,
    n_attempts: int = 16,
    temperature: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Dict[str, float], List[TaskResult]]:
    """Evaluate a list of tasks and return ``(summary, per-task results)``."""
    was_training = agent.training
    agent.eval()
    results: List[TaskResult] = []
    for i, task in enumerate(tasks):
        result = evaluate_task(
            agent, spec, task, env_config, device,
            n_attempts=n_attempts, temperature=temperature, seed=seed + i,
        )
        results.append(result)
        if verbose and (result.exact_at_2 or result.demo_fit):
            print(
                f"  {task.task_id}: demo_fit={result.demo_fit}/{result.attempts} "
                f"exact@1={result.exact_at_1} exact@2={result.exact_at_2}"
            )
    if was_training:
        agent.train()

    def rates(items: Sequence[TaskResult]) -> Dict[str, float]:
        n = max(1, len(items))
        return {
            "tasks": len(items),
            "demo_fit_rate": sum(r.demo_fit > 0 for r in items) / n,
            "exact_at_1": sum(r.exact_at_1 for r in items) / n,
            "exact_at_2": sum(r.exact_at_2 for r in items) / n,
            "pass_at_n": sum(r.pass_at_n for r in items) / n,
        }

    summary = {**rates(results), "n_attempts": n_attempts}
    sources = sorted({result.source for result in results if result.source})
    for source in sources:
        key = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_") or "arc"
        summary.update(
            {
                f"{key}/{metric}": value
                for metric, value in rates(
                    [result for result in results if result.source == source]
                ).items()
            }
        )
    return summary, results
