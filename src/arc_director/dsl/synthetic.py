"""Synthetic (grid, grid, program) generation for the DSL warm-start.

Plan section 7. The goal is *not* to teach ARC solutions -- it is to teach
syntax, operator semantics, and grounding between grid changes and program
operations.

Generation strategy
-------------------
Programs are grown **type-directed and executed incrementally**: a candidate
statement is only kept if it succeeds on *every* one of the K sampled input
grids. This prunes dead branches at the point they are created instead of
throwing away whole programs at the end, which is what makes the yield high
enough to be useful.

The result is a synthetic ARC task -- several demonstration pairs plus a test
pair, all produced by one known program -- so synthetic data flows through
exactly the same :class:`ArcTask` pipeline as real ARC data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..arc.dataset import ArcTask, Pair
from ..arc.grid import BACKGROUND, NUM_COLORS, Grid, grids_equal, to_lists
from .errors import DslError
from .executor import exec_statement, execute
from .operators import OPERATORS, OpSpec, Param
from .parser import Assign, BoolLit, Call, IntLit, Program, VarRef, parse
from .types import DslType

__all__ = ["random_grid", "SyntheticTask", "generate_task", "generate_dataset", "save_dataset"]


# ---------------------------------------------------------------------------
# Random grids
# ---------------------------------------------------------------------------


def random_grid(
    rng: np.random.Generator,
    *,
    min_side: int = 4,
    max_side: int = 12,
    n_objects: Tuple[int, int] = (1, 5),
    palette: Tuple[int, int] = (2, 5),
) -> Grid:
    """A random grid with object structure.

    Pure uniform noise would make ``COMPONENTS`` degenerate (one giant blob or
    900 single cells), so grids are built by stamping a handful of rectangles
    and small blobs onto a background. That gives the object operators
    something meaningful to find.
    """
    h = int(rng.integers(min_side, max_side + 1))
    w = int(rng.integers(min_side, max_side + 1))
    grid = np.full((h, w), BACKGROUND, dtype=np.int8)

    n_colors = int(rng.integers(palette[0], palette[1] + 1))
    colors = rng.choice(np.arange(1, NUM_COLORS), size=min(n_colors, NUM_COLORS - 1), replace=False)

    count = int(rng.integers(n_objects[0], n_objects[1] + 1))
    for _ in range(count):
        color = int(rng.choice(colors))
        if rng.random() < 0.6:
            # Solid rectangle.
            oh = int(rng.integers(1, max(2, min(4, h))))
            ow = int(rng.integers(1, max(2, min(4, w))))
            r = int(rng.integers(0, max(1, h - oh + 1)))
            c = int(rng.integers(0, max(1, w - ow + 1)))
            grid[r : r + oh, c : c + ow] = color
        else:
            # Random walk blob, which produces non-rectangular shapes.
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            for _ in range(int(rng.integers(2, 7))):
                grid[r, c] = color
                dr, dc = ((0, 1), (1, 0), (0, -1), (-1, 0))[int(rng.integers(4))]
                r = int(np.clip(r + dr, 0, h - 1))
                c = int(np.clip(c + dc, 0, w - 1))

    if (grid == BACKGROUND).all():
        grid[int(rng.integers(h)), int(rng.integers(w))] = int(rng.choice(colors))
    return grid


# ---------------------------------------------------------------------------
# Literal sampling
# ---------------------------------------------------------------------------

# Ranges keyed by parameter name. Chosen so generated programs stay inside the
# 30x30 ARC limit and produce visible, learnable effects.
_INT_RANGES: Dict[str, Tuple[int, int]] = {
    "n": (1, 3),
    "nh": (1, 3),
    "nw": (1, 3),
    "dr": (-3, 3),
    "dc": (-3, 3),
    "h": (2, 10),
    "w": (2, 10),
    "size": (1, 6),
}


def _sample_int(rng: np.random.Generator, param: Param) -> int:
    if param.choices:
        return int(rng.choice(np.asarray(param.choices)))
    lo, hi = _INT_RANGES.get(param.name, (0, 5))
    return int(rng.integers(lo, hi + 1))


def _sample_color(rng: np.random.Generator, grids: Sequence[Grid]) -> int:
    """Prefer colours that actually occur, so REPLACE_COLOR etc. do something."""
    if grids and rng.random() < 0.75:
        present = sorted({int(c) for g in grids for c in np.unique(g)})
        if present:
            return int(rng.choice(np.asarray(present)))
    return int(rng.integers(0, NUM_COLORS))


# ---------------------------------------------------------------------------
# Program growth
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    """Generation state for one program being grown.

    ``envs`` holds one live execution environment per sampled input grid, so a
    candidate statement costs one operator call per grid rather than a full
    re-run of the program so far.
    """

    envs: List[Dict[str, object]] = field(default_factory=list)
    types: Dict[str, DslType] = field(default_factory=dict)
    statements: List[Assign] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=dict)
    last_target: Optional[str] = None

    def vars_of(self, spec) -> List[str]:
        targets = spec if isinstance(spec, tuple) else (spec,)
        out = []
        for name, t in self.types.items():
            for want in targets:
                if t == want or {t, want} == {DslType.COLOR, DslType.INTEGER}:
                    out.append(name)
                    break
        return out

    def fresh(self, dsl_type: DslType) -> str:
        prefix = {
            DslType.GRID: "g",
            DslType.OBJECT: "o",
            DslType.OBJECT_SET: "s",
            DslType.INTEGER: "n",
            DslType.COLOR: "k",
            DslType.BOOL: "b",
        }[dsl_type]
        idx = self.counters.get(prefix, 0)
        self.counters[prefix] = idx + 1
        return f"{prefix}{idx}"


_VALUE_TYPES = (DslType.GRID, DslType.OBJECT, DslType.OBJECT_SET)

# Probability that a statement is forced to consume the previous statement's
# result. Chaining is what produces COMPONENTS -> LARGEST -> CROP instead of a
# pile of independent one-liners, and it means almost nothing is dead code.
_CHAIN_P = 0.85


def _consumes(spec: OpSpec, dsl_type: DslType) -> bool:
    """True when some parameter of ``spec`` can accept a value of ``dsl_type``."""
    for param in spec.params:
        targets = param.type if isinstance(param.type, tuple) else (param.type,)
        for t in targets:
            if t == dsl_type or {t, dsl_type} == {DslType.COLOR, DslType.INTEGER}:
                return True
    return False


def _satisfiable(spec: OpSpec, frame: _Frame) -> bool:
    """True when every required value-typed parameter has a candidate variable."""
    for param in spec.params:
        if not param.required:
            continue
        targets = param.type if isinstance(param.type, tuple) else (param.type,)
        if any(t in _VALUE_TYPES for t in targets) and not frame.vars_of(param.type):
            return False
    return True


def _candidate_args(
    rng: np.random.Generator,
    spec: OpSpec,
    frame: _Frame,
    grids: Sequence[Grid],
    *,
    chain_from: Optional[str] = None,
) -> Optional[Tuple[List, List[Tuple[str, object]]]]:
    """Pick concrete arguments for an operator, or None if unsatisfiable.

    When ``chain_from`` names a variable, the first compatible value-typed
    parameter is bound to it so the new statement extends the chain.
    """
    positional: List = []
    keyword: List[Tuple[str, object]] = []
    chained = chain_from is None

    for param in spec.params:
        # Optional params are supplied only sometimes, so defaults get exercised too.
        if not param.required and rng.random() < 0.5:
            continue

        target_types = param.type if isinstance(param.type, tuple) else (param.type,)
        wants_value_type = any(t in _VALUE_TYPES for t in target_types)

        options = frame.vars_of(param.type)

        if wants_value_type:
            # Grids, Objects and ObjectSets can only come from variables.
            if not options:
                return None
            if not chained and chain_from in options:
                arg = VarRef(chain_from, 0)
                chained = True
            else:
                arg = VarRef(str(rng.choice(options)), 0)
        elif DslType.BOOL in target_types:
            arg = BoolLit(bool(rng.integers(2)), 0)
        else:
            # Integers and Colours may come from a variable or a literal.
            # Referencing a variable is what produces genuinely relational
            # programs (`n0 = COUNT(s0)` then `EMPTY_GRID(n0, n0)`), so prefer
            # the chain variable when it fits, and otherwise mix the two.
            use_var = bool(options) and (
                (not chained and chain_from in options) or rng.random() < 0.35
            )
            if use_var and not chained and chain_from in options:
                arg = VarRef(chain_from, 0)
                chained = True
            elif use_var:
                arg = VarRef(str(rng.choice(options)), 0)
            elif param.choices:
                arg = IntLit(_sample_int(rng, param), 0)
            elif DslType.COLOR in target_types:
                arg = IntLit(_sample_color(rng, grids), 0)
            else:
                arg = IntLit(_sample_int(rng, param), 0)

        if param.required:
            positional.append(arg)
        else:
            keyword.append((param.name, arg))

    if chain_from is not None and not chained:
        return None
    return positional, keyword


def _try_statement(
    rng: np.random.Generator,
    frame: _Frame,
    grids: Sequence[Grid],
    op_name: str,
    *,
    chain_from: Optional[str] = None,
) -> bool:
    """Attempt to append one statement. Returns True if it executed everywhere."""
    spec = OPERATORS[op_name]
    picked = _candidate_args(rng, spec, frame, grids, chain_from=chain_from)
    if picked is None:
        return False
    positional, keyword = picked

    call = Call(op=op_name, args=tuple(positional), kwargs=tuple(keyword), line=len(frame.statements) + 1)

    # Static return type. Polymorphic ops need the concrete arg types.
    arg_types: List[DslType] = []
    for param, arg in zip(spec.params, positional):
        if isinstance(arg, VarRef):
            arg_types.append(frame.types[arg.name])
        elif isinstance(arg, BoolLit):
            arg_types.append(DslType.BOOL)
        else:
            arg_types.append(DslType.INTEGER)
    ret_type = spec.return_type(tuple(arg_types))

    target = frame.fresh(ret_type)
    stmt = Assign(target=target, call=call, line=call.line)

    # Run the new statement against every sampled input. All must succeed or
    # it is dropped. Values are committed only after every environment agrees,
    # so a partial failure leaves no trace.
    values = []
    for env, env_grid in zip(frame.envs, grids):
        ok, value, _ = exec_statement(stmt, env, env_grid)
        if not ok:
            frame.counters[target[0]] -= 1  # release the name we reserved
            return False
        values.append(value)

    for env, value in zip(frame.envs, values):
        env[target] = value
    frame.statements.append(stmt)
    frame.types[target] = ret_type
    frame.last_target = target
    return True


def _prune(statements: Sequence[Assign], ret: str) -> Tuple[Assign, ...]:
    """Dead-code elimination: keep only statements the RETURN value depends on.

    Without this, generated programs carry statements whose results are never
    used. That inflates the AST-size measure and, worse, teaches the model
    during SFT that emitting unused lines is normal.
    """
    live = {ret}
    kept: List[Assign] = []
    for stmt in reversed(statements):
        if stmt.target not in live:
            continue
        kept.append(stmt)
        for arg in list(stmt.call.args) + [a for _, a in stmt.call.kwargs]:
            if isinstance(arg, VarRef):
                live.add(arg.name)
    return tuple(reversed(kept))


def _grow_program(
    rng: np.random.Generator,
    grids: Sequence[Grid],
    *,
    n_ops: int,
    max_attempts: int = 80,
) -> Optional[Program]:
    """Grow a program whose *live* operator count (excluding INPUT) is ``n_ops``.

    Operator choice is restricted to those currently satisfiable, and biased
    toward those that consume the previous statement's result. Growth stops
    when the dead-code-eliminated program reaches the requested size, so the
    returned program contains no unused statements.
    """
    frame = _Frame(envs=[{} for _ in grids])
    # Every program starts by binding the input.
    frame.statements.append(Assign(target="g0", call=Call("INPUT", (), (), 1), line=1))
    frame.types["g0"] = DslType.GRID
    frame.counters["g"] = 1
    frame.last_target = "g0"
    for env, g in zip(frame.envs, grids):
        env["g0"] = g.copy()

    growable = [n for n in OPERATORS if n != "INPUT"]
    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        chain_from = frame.last_target
        chain = chain_from is not None and rng.random() < _CHAIN_P
        if chain:
            want = frame.types[chain_from]
            pool = [n for n in growable if _consumes(OPERATORS[n], want) and _satisfiable(OPERATORS[n], frame)]
            if not pool:
                chain = False
        if not chain:
            chain_from = None
            pool = [n for n in growable if _satisfiable(OPERATORS[n], frame)]
        if not pool:
            return None

        _try_statement(rng, frame, grids, str(rng.choice(pool)), chain_from=chain_from)

        # A program is complete when *some* grid variable, after dead-code
        # elimination, is the root of exactly `n_ops` live operators. Every
        # grid variable is considered, not just the newest: a single statement
        # can make several earlier ones live at once (FILL revives the whole
        # COMPONENTS -> SMALLEST chain), so insisting on the latest variable
        # would discard most object-flavoured programs.
        grid_vars = [n for n, t in frame.types.items() if t == DslType.GRID and n != "g0"]
        for ret in reversed(grid_vars):
            live = _prune(frame.statements, ret)
            if sum(1 for s in live if s.call.op != "INPUT") == n_ops:
                return Program(
                    statements=live,
                    ret=ret,
                    types={s.target: frame.types[s.target] for s in live},
                )
    return None


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTask:
    """A synthetic ARC task with a known ground-truth program."""

    program_text: str
    train: List[Tuple[Grid, Grid]]
    test: List[Tuple[Grid, Grid]]
    n_ops: int
    ops: Tuple[str, ...]

    def to_arc_task(self, task_id: str, source: str = "synthetic") -> ArcTask:
        return ArcTask(
            task_id=task_id,
            source=source,
            train=[Pair(i, o) for i, o in self.train],
            test=[Pair(i, o) for i, o in self.test],
        )

    def as_dict(self, task_id: str = "") -> dict:
        return {
            "task_id": task_id,
            "program": self.program_text,
            "n_ops": self.n_ops,
            "ops": list(self.ops),
            "train": [{"input": to_lists(i), "output": to_lists(o)} for i, o in self.train],
            "test": [{"input": to_lists(i), "output": to_lists(o)} for i, o in self.test],
        }


def generate_task(
    rng: np.random.Generator,
    *,
    n_ops: int = 2,
    n_demos: int = 3,
    n_test: int = 1,
    grid_kwargs: Optional[dict] = None,
    require_nontrivial: bool = True,
) -> Optional[SyntheticTask]:
    """Generate one synthetic task, or None if this attempt produced nothing usable."""
    grid_kwargs = grid_kwargs or {}
    n_grids = n_demos + n_test
    inputs = [random_grid(rng, **grid_kwargs) for _ in range(n_grids)]

    program = _grow_program(rng, inputs, n_ops=n_ops)
    if program is None:
        return None

    outputs: List[Grid] = []
    for g in inputs:
        result = execute(program, g)
        if not result.ok or result.grid is None:
            return None
        outputs.append(result.grid)

    if require_nontrivial:
        # Reject identity programs: they teach the model that copying is the
        # answer, which is a trap it will happily fall into during RL.
        if all(grids_equal(i, o) for i, o in zip(inputs, outputs)):
            return None
        # Reject blank / single-colour outputs, which carry no structure.
        if any(len(np.unique(o)) == 1 for o in outputs):
            return None

    text = program.to_text()
    # Round-trip through the real parser so every saved example is guaranteed
    # to be parseable by the same code path the policy's output will hit.
    try:
        reparsed = parse(text, extract=False)
    except DslError:
        return None
    if reparsed.to_text() != text:
        return None

    return SyntheticTask(
        program_text=text,
        train=list(zip(inputs[:n_demos], outputs[:n_demos])),
        test=list(zip(inputs[n_demos:], outputs[n_demos:])),
        n_ops=n_ops,
        ops=program.ops_used,
    )


def generate_dataset(
    n: int,
    *,
    seed: int = 0,
    n_ops_choices: Sequence[int] = (1, 2, 3),
    n_demos: int = 3,
    n_test: int = 1,
    max_attempts_per_task: int = 40,
    dedup: bool = True,
    grid_kwargs: Optional[dict] = None,
) -> List[SyntheticTask]:
    """Generate ``n`` distinct synthetic tasks.

    ``dedup`` drops repeat program texts, which keeps the SFT set from
    collapsing onto the handful of operators that are easiest to satisfy.
    """
    rng = np.random.default_rng(seed)
    tasks: List[SyntheticTask] = []
    seen: set[str] = set()
    attempts = 0
    budget = n * max_attempts_per_task

    while len(tasks) < n and attempts < budget:
        attempts += 1
        n_ops = int(rng.choice(np.asarray(n_ops_choices)))
        task = generate_task(
            rng, n_ops=n_ops, n_demos=n_demos, n_test=n_test, grid_kwargs=grid_kwargs
        )
        if task is None:
            continue
        if dedup:
            if task.program_text in seen:
                continue
            seen.add(task.program_text)
        tasks.append(task)
    return tasks


def save_dataset(tasks: Iterable[SyntheticTask], path: str | Path) -> int:
    """Write tasks as JSONL. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks):
            fh.write(json.dumps(task.as_dict(task_id=f"syn_{i:06d}")) + "\n")
            count += 1
    return count
