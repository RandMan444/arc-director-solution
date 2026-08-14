"""Line-oriented DSL parser.

Grammar (v0)::

    program := statement+ return
    statement := IDENT '=' call
    return    := 'RETURN' IDENT
    call      := OPNAME | OPNAME '(' arg (',' arg)* ')'
    arg       := IDENT | INT | 'true' | 'false' | IDENT '=' (IDENT|INT|bool)

Deliberately **not** supported: nested calls. ``LARGEST(COMPONENTS(g))`` must be
split across two lines. Single static assignment keeps each intermediate value
named, which makes the AST size a meaningful complexity measure and makes
per-step credit assignment tractable. The parser emits a dedicated
``nested_call`` error for this so the policy gets an actionable correction
rather than a generic syntax failure.

There is no ``eval``/``exec`` anywhere in this module or the executor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .errors import DslSyntaxError, DslTypeError
from .operators import OPERATORS, OpSpec, Param, is_assignable
from .types import DslType

__all__ = [
    "Program",
    "Assign",
    "Call",
    "VarRef",
    "IntLit",
    "BoolLit",
    "parse",
    "extract_program_block",
    "MAX_STATEMENTS",
]

MAX_STATEMENTS = 24

_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_OPNAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_INT_RE = re.compile(r"^-?\d+$")
_ASSIGN_RE = re.compile(r"^([^=]+?)\s*=\s*(.+)$")
_CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?$")
_RETURN_RE = re.compile(r"^RETURN\s+(.+)$", re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarRef:
    name: str
    line: int


@dataclass(frozen=True)
class IntLit:
    value: int
    line: int


@dataclass(frozen=True)
class BoolLit:
    value: bool
    line: int


Arg = Union[VarRef, IntLit, BoolLit]


@dataclass(frozen=True)
class Call:
    op: str
    args: Tuple[Arg, ...]
    kwargs: Tuple[Tuple[str, Arg], ...]
    line: int

    @property
    def n_args(self) -> int:
        return len(self.args) + len(self.kwargs)


@dataclass(frozen=True)
class Assign:
    target: str
    call: Call
    line: int


@dataclass(frozen=True)
class Program:
    statements: Tuple[Assign, ...]
    ret: str
    source: str = ""
    types: Dict[str, DslType] = field(default_factory=dict)

    @property
    def ast_nodes(self) -> int:
        """Complexity proxy used by the reward's length penalty.

        One node per statement, one per argument, one for the RETURN.
        """
        return sum(1 + s.call.n_args for s in self.statements) + 1

    @property
    def ops_used(self) -> Tuple[str, ...]:
        return tuple(s.call.op for s in self.statements)

    def op_multiset_key(self) -> str:
        """Order-insensitive fingerprint, used to measure program-family diversity."""
        return "|".join(sorted(self.ops_used))

    def to_text(self) -> str:
        lines = []
        for s in self.statements:
            parts = [_arg_text(a) for a in s.call.args]
            parts += [f"{k}={_arg_text(v)}" for k, v in s.call.kwargs]
            call = s.call.op if not parts else f"{s.call.op}({', '.join(parts)})"
            lines.append(f"{s.target} = {call}")
        lines.append(f"RETURN {self.ret}")
        return "\n".join(lines)


def _arg_text(a: Arg) -> str:
    if isinstance(a, VarRef):
        return a.name
    if isinstance(a, BoolLit):
        return "true" if a.value else "false"
    return str(a.value)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_program_block(text: str) -> str:
    """Pull the DSL out of a model completion.

    Prefers a fenced code block; otherwise keeps the lines from the first
    assignment or RETURN onward, dropping any prose the model prepended. This
    is lenient on purpose: the reward already penalises invalid programs, and
    we do not want to score a correct program as invalid because the model said
    "Here is my answer:" first.
    """
    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()

    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _RETURN_RE.match(stripped) or (_ASSIGN_RE.match(stripped) and "==" not in stripped):
            start = i
            break
    if start is None:
        return text.strip()
    return "\n".join(lines[start:]).strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_arg(token: str, line: int) -> Arg:
    token = token.strip()
    if not token:
        raise DslSyntaxError("empty argument", line=line, code="empty_arg")
    low = token.lower()
    if low in ("true", "false"):
        return BoolLit(low == "true", line)
    if _INT_RE.match(token):
        return IntLit(int(token), line)
    if _IDENT_RE.match(token):
        return VarRef(token, line)
    if "(" in token or ")" in token:
        raise DslSyntaxError(
            f"nested calls are not allowed in DSL v0: {token!r}. Assign the inner "
            "call to its own variable on a previous line.",
            line=line,
            code="nested_call",
        )
    raise DslSyntaxError(
        f"cannot parse argument {token!r}; expected a variable name, an integer, "
        "true or false",
        line=line,
        code="bad_arg",
    )


def _split_args(text: str, line: int) -> List[str]:
    if "(" in text or ")" in text:
        raise DslSyntaxError(
            "nested calls are not allowed in DSL v0. Assign the inner call to its "
            "own variable on a previous line.",
            line=line,
            code="nested_call",
        )
    return [part for part in text.split(",")]


def _parse_call(text: str, line: int) -> Call:
    match = _CALL_RE.match(text.strip())
    if not match:
        raise DslSyntaxError(f"cannot parse expression {text!r}", line=line, code="bad_expression")
    op, arg_text = match.group(1), match.group(2)

    if not _OPNAME_RE.match(op):
        raise DslSyntaxError(
            f"operator names are UPPERCASE; got {op!r}", line=line, code="bad_operator_name"
        )
    if op not in OPERATORS:
        raise DslSyntaxError(f"unknown operator {op!r}", line=line, code="unknown_operator")

    args: List[Arg] = []
    kwargs: List[Tuple[str, Arg]] = []
    if arg_text is not None and arg_text.strip():
        seen_kw: set[str] = set()
        for raw in _split_args(arg_text, line):
            piece = raw.strip()
            if not piece:
                raise DslSyntaxError("empty argument", line=line, code="empty_arg")
            kw_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", piece)
            if kw_match and _IDENT_RE.match(kw_match.group(1)):
                name, value = kw_match.group(1), kw_match.group(2)
                if name in seen_kw:
                    raise DslSyntaxError(
                        f"duplicate keyword argument {name!r}", line=line, code="duplicate_kwarg"
                    )
                seen_kw.add(name)
                kwargs.append((name, _parse_arg(value, line)))
            else:
                if kwargs:
                    raise DslSyntaxError(
                        "positional argument after keyword argument",
                        line=line,
                        code="arg_order",
                    )
                args.append(_parse_arg(piece, line))
    return Call(op=op, args=tuple(args), kwargs=tuple(kwargs), line=line)


def _bind_and_check(call: Call, env: Dict[str, DslType]) -> DslType:
    """Type check a call against its declared signature; return its result type."""
    spec: OpSpec = OPERATORS[call.op]
    params: Sequence[Param] = spec.params

    if len(call.args) > len(params):
        raise DslTypeError(
            f"{call.op} takes at most {len(params)} arguments, got {len(call.args)}",
            line=call.line,
            code="too_many_args",
        )

    by_name: Dict[str, Arg] = {}
    for param, arg in zip(params, call.args):
        by_name[param.name] = arg
    param_names = {p.name for p in params}
    for name, arg in call.kwargs:
        if name not in param_names:
            raise DslTypeError(
                f"{call.op} has no parameter {name!r}; expected one of "
                f"{sorted(param_names)}",
                line=call.line,
                code="unknown_kwarg",
            )
        if name in by_name:
            raise DslTypeError(
                f"{call.op} got multiple values for {name!r}", line=call.line, code="duplicate_arg"
            )
        by_name[name] = arg

    arg_types: List[DslType] = []
    for param in params:
        if param.name not in by_name:
            if param.required:
                raise DslTypeError(
                    f"{call.op} is missing required argument {param.name!r}. "
                    f"Signature: {spec.signature()}",
                    line=call.line,
                    code="missing_arg",
                )
            continue

        arg = by_name[param.name]
        if isinstance(arg, VarRef):
            if arg.name not in env:
                raise DslTypeError(
                    f"{arg.name!r} is not defined yet", line=call.line, code="undefined_name"
                )
            actual = env[arg.name]
        elif isinstance(arg, BoolLit):
            actual = DslType.BOOL
        else:
            actual = DslType.INTEGER

        if not is_assignable(actual, param.type):
            raise DslTypeError(
                f"{call.op} argument {param.name!r} expects {param.type_names()}, "
                f"got {actual}",
                line=call.line,
                code="type_mismatch",
            )
        if param.choices and isinstance(arg, IntLit) and arg.value not in param.choices:
            raise DslTypeError(
                f"{call.op} argument {param.name!r} must be one of {list(param.choices)}, "
                f"got {arg.value}",
                line=call.line,
                code="bad_choice",
            )
        arg_types.append(actual)

    return spec.return_type(arg_types)


def parse(text: str, *, extract: bool = True, max_statements: int = MAX_STATEMENTS) -> Program:
    """Parse and type check a DSL program.

    Raises :class:`DslSyntaxError` or :class:`DslTypeError`. Never executes
    anything and never touches the filesystem.
    """
    source = extract_program_block(text) if extract else text
    raw_lines = source.splitlines()

    statements: List[Assign] = []
    env: Dict[str, DslType] = {}
    ret_name: Optional[str] = None
    ret_line = 0

    for lineno, raw in enumerate(raw_lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        if ret_name is not None:
            raise DslSyntaxError(
                "no statements may follow RETURN", line=lineno, code="code_after_return"
            )

        ret_match = _RETURN_RE.match(line)
        if ret_match:
            name = ret_match.group(1).strip()
            if not _IDENT_RE.match(name):
                raise DslSyntaxError(
                    f"RETURN expects a variable name, got {name!r}",
                    line=lineno,
                    code="bad_return_target",
                )
            ret_name, ret_line = name, lineno
            continue

        assign_match = _ASSIGN_RE.match(line)
        if not assign_match:
            raise DslSyntaxError(
                f"expected `name = OPERATOR(...)` or `RETURN name`, got {line!r}",
                line=lineno,
                code="bad_statement",
            )

        target = assign_match.group(1).strip()
        if not _IDENT_RE.match(target):
            raise DslSyntaxError(
                f"invalid variable name {target!r}; use lowercase letters, digits "
                "and underscores, starting with a letter",
                line=lineno,
                code="bad_variable_name",
            )
        if target in env:
            raise DslSyntaxError(
                f"{target!r} is already defined; every variable is assigned exactly once",
                line=lineno,
                code="redefined_name",
            )
        if len(statements) >= max_statements:
            raise DslSyntaxError(
                f"program exceeds {max_statements} statements", line=lineno, code="too_long"
            )

        call = _parse_call(assign_match.group(2), lineno)
        env[target] = _bind_and_check(call, env)
        statements.append(Assign(target=target, call=call, line=lineno))

    if not statements:
        raise DslSyntaxError("program has no statements", line=1, code="empty_program")
    if ret_name is None:
        raise DslSyntaxError(
            "program must end with `RETURN <variable>`",
            line=len(raw_lines),
            code="missing_return",
        )
    if ret_name not in env:
        raise DslTypeError(
            f"RETURN refers to undefined variable {ret_name!r}",
            line=ret_line,
            code="undefined_name",
        )
    if env[ret_name] != DslType.GRID:
        raise DslTypeError(
            f"RETURN must yield a Grid, but {ret_name!r} is a {env[ret_name]}",
            line=ret_line,
            code="return_not_grid",
        )

    return Program(statements=tuple(statements), ret=ret_name, source=source, types=dict(env))
