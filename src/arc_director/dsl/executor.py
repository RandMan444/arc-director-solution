"""Program execution.

Interprets the parsed AST directly. There is no ``eval``, no ``exec``, no
import of model-supplied text, and no filesystem or network access -- the only
things an operator can touch are numpy arrays we handed it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..arc.grid import Grid
from .errors import DslError, DslLimitError, DslRuntimeError
from .operators import OPERATORS, OpSpec
from .parser import Assign, BoolLit, IntLit, Program, VarRef, parse
from .types import DslType, type_of

__all__ = ["ExecResult", "execute", "run_program", "run_on_pairs", "DEFAULT_TIMEOUT_S"]

DEFAULT_TIMEOUT_S = 2.0


@dataclass
class ExecResult:
    """Outcome of running one program on one input grid."""

    ok: bool
    grid: Optional[Grid] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_line: Optional[int] = None
    steps: int = 0

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "error_line": self.error_line,
            "steps": self.steps,
        }


def _resolve(arg, env: Dict[str, Any], line: int):
    if isinstance(arg, VarRef):
        if arg.name not in env:
            raise DslRuntimeError(f"{arg.name!r} is not defined", line=line, code="undefined_name")
        return env[arg.name]
    if isinstance(arg, BoolLit):
        return arg.value
    if isinstance(arg, IntLit):
        return arg.value
    raise DslRuntimeError(f"cannot resolve argument {arg!r}", line=line, code="bad_arg")


def _call(stmt: Assign, env: Dict[str, Any], input_grid: Grid):
    call = stmt.call
    spec: OpSpec = OPERATORS[call.op]

    if call.op == "INPUT":
        return input_grid.copy()

    kwargs: Dict[str, Any] = {}
    for param, arg in zip(spec.params, call.args):
        kwargs[param.name] = _resolve(arg, env, call.line)
    for name, arg in call.kwargs:
        kwargs[name] = _resolve(arg, env, call.line)

    try:
        return spec.fn(**kwargs)
    except DslError as exc:
        # Operator implementations raise without knowing where they were called
        # from. Attach the statement's line so the feedback can point at it.
        if exc.line is None:
            exc.line = call.line
        raise
    except (ValueError, TypeError, IndexError, KeyError, ZeroDivisionError, OverflowError) as exc:
        raise DslRuntimeError(
            f"{call.op} failed: {type(exc).__name__}: {exc}", line=call.line, code="operator_failed"
        ) from exc
    except MemoryError as exc:  # pragma: no cover - defensive
        raise DslLimitError(f"{call.op} ran out of memory", line=call.line) from exc


def exec_statement(
    stmt: Assign, env: Dict[str, Any], input_grid: Grid
) -> tuple[bool, Any, Optional[DslError]]:
    """Run a single statement against a live environment without mutating it.

    Returns ``(ok, value, error)``. The caller decides whether to commit the
    value into ``env``. Used by the synthetic generator to extend a program
    incrementally instead of re-running it from the top for every candidate.
    """
    try:
        value = _call(stmt, env, input_grid)
    except DslError as exc:
        return False, None, exc
    if type_of(value) is None:
        return (
            False,
            None,
            DslRuntimeError(
                f"{stmt.call.op} produced a non-DSL value of type {type(value).__name__}",
                line=stmt.line,
                code="bad_return_value",
            ),
        )
    return True, value, None


def execute(
    program: Program,
    input_grid: Grid,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    require_grid: bool = True,
) -> ExecResult:
    """Run a parsed program against one input grid.

    Returns an :class:`ExecResult` instead of raising: execution failure is
    normal, expected RL feedback, not an exceptional condition.

    ``require_grid=False`` allows the RETURN target to be any DSL value. Only
    the synthetic generator uses this, to probe partially-grown programs whose
    latest value is an ObjectSet or an Integer. Real programs must return a Grid.
    """
    env: Dict[str, Any] = {}
    deadline = time.monotonic() + timeout_s
    steps = 0

    try:
        for stmt in program.statements:
            if time.monotonic() > deadline:
                raise DslLimitError(
                    f"execution exceeded {timeout_s}s", line=stmt.line, code="timeout"
                )
            value = _call(stmt, env, input_grid)
            if type_of(value) is None:
                raise DslRuntimeError(
                    f"{stmt.call.op} produced a non-DSL value of type {type(value).__name__}",
                    line=stmt.line,
                    code="bad_return_value",
                )
            env[stmt.target] = value
            steps += 1

        if program.ret not in env:
            raise DslRuntimeError(
                f"RETURN value {program.ret!r} was never assigned", code="undefined_name"
            )
        result = env[program.ret]
        if isinstance(result, np.ndarray):
            return ExecResult(
                ok=True, grid=np.ascontiguousarray(result, dtype=np.int8), steps=steps
            )
        if require_grid:
            raise DslRuntimeError(
                f"RETURN value {program.ret!r} is not a Grid", code="return_not_grid"
            )
        return ExecResult(ok=True, grid=None, steps=steps)

    except DslError as exc:
        return ExecResult(
            ok=False,
            error_code=exc.code,
            error_message=exc.message,
            error_line=exc.line,
            steps=steps,
        )


def run_program(
    text: str, input_grid: Grid, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> tuple[Optional[Program], ExecResult]:
    """Parse then execute. Parse failures come back as an :class:`ExecResult` too."""
    try:
        program = parse(text)
    except DslError as exc:
        return None, ExecResult(
            ok=False, error_code=exc.code, error_message=exc.message, error_line=exc.line
        )
    return program, execute(program, input_grid, timeout_s=timeout_s)


def run_on_pairs(
    program: Program,
    inputs: Sequence[Grid],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> List[ExecResult]:
    """Run one program across every demonstration input."""
    return [execute(program, g, timeout_s=timeout_s) for g in inputs]
