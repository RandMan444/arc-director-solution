"""Bounded search over the register machine, for certifying DSL coverage.

This answers one narrow question per task: *can this action space express a
program that is exact on every demonstration and on the labelled test pair,
inside a stated budget?* A hit is proof of coverage. A miss is not proof of
inexpressibility -- the search is bounded, and says so.

Why it matters here more than it did before. The predecessor project ran RL on
the full ARC pool with a DSL that could provably express about 2% of it, so a
flat learning curve had two indistinguishable explanations: the agent cannot
learn, or the answer is not in the language. Mining the reachable subset first
separates them. The reachable tasks become the first ARC rung of the
curriculum, where a failure is unambiguously the agent's.

The search is a randomised beam: expansion by exhaustive enumeration is
hopeless (98 operators times up to five arguments each), so each node is
expanded by sampling legal actions under the machine's own masks, and nodes are
ranked by mean shaping accuracy against the demonstration outputs.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..arc.dataset import ArcTask
from ..arc.grid import Grid, grids_equal
from ..arc.scoring import score_pair
from .machine import Machine, MachineSpec

__all__ = ["SearchConfig", "SearchResult", "search_task"]


@dataclass
class SearchConfig:
    max_depth: int = 3
    beam_width: int = 16
    #: Children generated per operator per node. When an operator's legal
    #: argument combinations number no more than this, *all* of them are tried;
    #: otherwise this many are sampled. Enumerating the small ones matters:
    #: every dihedral transform, every single-colour recolour and every
    #: connectivity choice is reachable in one step, and those are exactly the
    #: programs a short-budget search should never miss by bad luck.
    per_op_samples: int = 4
    timeout_s: float = 5.0
    seed: int = 0


@dataclass
class SearchResult:
    task_id: str
    solved: bool                 # exact on every demonstration
    test_exact: bool             # and on the labelled test pair
    program: Optional[str]
    depth: Optional[int]
    best_score: float
    nodes: int
    elapsed_s: float

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "solved": self.solved,
            "test_exact": self.test_exact,
            "program": self.program,
            "depth": self.depth,
            "best_score": round(self.best_score, 4),
            "nodes": self.nodes,
            "elapsed_s": round(self.elapsed_s, 3),
        }


def _score(machine: Machine, targets: Sequence[Grid]) -> Tuple[float, bool]:
    """Mean shaping accuracy over the demonstrations, and whether all are exact."""
    total = 0.0
    exact = True
    for i, target in enumerate(targets):
        canvas = machine.canvas(i)
        s = score_pair(canvas, target)
        total += (s.cell_accuracy + 2.0 * s.balanced_cell_accuracy) / 3.0
        exact = exact and s.exact
    return total / max(1, len(targets)), exact


def _expand(
    rng: np.random.Generator, spec: MachineSpec, machine: Machine, per_op: int
) -> List[Tuple[int, List[int]]]:
    """Candidate statements from one machine state.

    Every legal operator contributes: exhaustively when its legal argument
    combinations are few, by sampling when they are many. Giving each operator
    its own quota is what stops the operators with dozens of literal arguments
    from crowding out the argument-free ones.
    """
    arg_masks = machine.arg_masks()
    op_mask = machine.op_mask(arg_masks)
    op_mask[spec.halt_index] = False

    out: List[Tuple[int, List[int]]] = []
    for op in np.flatnonzero(op_mask):
        op = int(op)
        params = spec.spec_of(op).params
        choices = [np.flatnonzero(arg_masks[op, p]) for p in range(len(params))]
        if any(c.size == 0 for c in choices):
            continue
        total = int(np.prod([c.size for c in choices])) if choices else 1
        if total <= per_op:
            for combo in itertools.product(*choices):
                args = [0] * spec.max_arity
                args[: len(combo)] = [int(a) for a in combo]
                out.append((op, args))
        else:
            for _ in range(per_op):
                args = [0] * spec.max_arity
                for p, c in enumerate(choices):
                    args[p] = int(rng.choice(c))
                out.append((op, args))
    rng.shuffle(out)
    return out


def search_task(
    spec: MachineSpec,
    task: ArcTask,
    config: Optional[SearchConfig] = None,
) -> SearchResult:
    """Search one task. Demonstrations drive the search; the test pair only validates."""
    cfg = config or SearchConfig()
    rng = np.random.default_rng(cfg.seed)
    start = time.monotonic()

    demos = task.demo_pairs()
    if not demos:
        return SearchResult(task.task_id, False, False, None, None, 0.0, 0, 0.0)
    test = [p for p in task.test if p.output is not None]

    inputs = [i for i, _ in demos] + [p.input for p in test]
    targets = [o for _, o in demos]

    root = Machine(spec, inputs)
    base_score, base_exact = _score(root, targets)
    if base_exact:  # the identity already solves it
        return SearchResult(
            task.task_id, True, _test_exact(root, task, len(demos)),
            root.program_text(), 0, base_score, 1, time.monotonic() - start,
        )

    beam: List[Tuple[float, Machine]] = [(base_score, root)]
    best_score = base_score
    nodes = 1

    for depth in range(1, cfg.max_depth + 1):
        candidates: List[Tuple[float, Machine]] = []
        for _, node in beam:
            for action in _expand(rng, spec, node, cfg.per_op_samples):
                if time.monotonic() - start > cfg.timeout_s:
                    return SearchResult(
                        task.task_id, False, False, None, None, best_score, nodes,
                        time.monotonic() - start,
                    )
                child = node.clone()
                if not child.step(action[0], action[1]).ok:
                    continue
                nodes += 1
                score, exact = _score(child, targets)
                best_score = max(best_score, score)
                if exact:
                    return SearchResult(
                        task.task_id, True, _test_exact(child, task, len(demos)),
                        child.program_text(), depth, score, nodes,
                        time.monotonic() - start,
                    )
                candidates.append((score, child))
        if not candidates:
            break
        # Deduplicate by behaviour: two programs that leave every canvas
        # identical are the same node for search purposes.
        seen: Dict[str, bool] = {}
        unique: List[Tuple[float, Machine]] = []
        for score, child in sorted(candidates, key=lambda c: -c[0]):
            key = "|".join(
                f"{c.shape[0]}x{c.shape[1]}:" + c.tobytes().hex() for c in child.canvases()
            )
            if key in seen:
                continue
            seen[key] = True
            unique.append((score, child))
            if len(unique) >= cfg.beam_width:
                break
        beam = unique

    return SearchResult(
        task.task_id, False, False, None, None, best_score, nodes, time.monotonic() - start
    )


def _test_exact(machine: Machine, task: ArcTask, n_demos: int) -> bool:
    """Validate a demonstration-exact program against the labelled test pairs."""
    labelled = [p for p in task.test if p.output is not None]
    if not labelled:
        return False
    return all(
        grids_equal(machine.canvas(n_demos + i), pair.output)
        for i, pair in enumerate(labelled)
    )
