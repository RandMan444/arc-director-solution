"""The register machine: the DSL rewritten as a discrete action space.

The predecessor project had a language model emit whole programs as text. A
Director worker instead takes one action per environment step, so the DSL has
to become a *machine* that a policy can drive one statement at a time.

Shape of one action
-------------------
An action is factored into ``1 + MAX_ARITY`` discrete choices:

    op          which operator to run (or HALT)
    arg_0..4    one entry of a shared argument vocabulary per parameter

The argument vocabulary is a single flat list so every argument head shares an
embedding table:

    register references   "the newest Grid", "the second-newest ObjectSet", ...
    integer literals      a curated ladder that covers colours, directions,
                          connectivity, small counts and grid coordinates
    booleans              true / false
    DEFAULT               use the operator's declared default (optional params)

Every head is masked against what is actually legal *right now*: an operator is
offered only when each of its required parameters has at least one legal
argument, and an argument entry is offered only when its type is assignable to
the parameter and, for a register reference, that register is occupied. A
sampled action is therefore always type-correct by construction; the only
failures left are genuine runtime ones (``LARGEST`` of an empty set), which are
informative and are reported rather than crashing.

Registers are recency-indexed rings per type
--------------------------------------------
There is no destination-register head. A statement's result is pushed onto the
ring for its return type, so index 0 always means "the value this type most
recently took". This removes a whole action head, keeps the reference space
tiny, and makes ``Grid[0]`` a natural *canvas*: it is the working grid, it is
what the goal-reaching reward is measured against, and it is the answer when
the episode ends.

One machine, many contexts
--------------------------
A candidate program must work on every demonstration, so the machine holds one
value bank per *context* -- each demonstration input, plus the test input --
and executes each statement against all of them at once. A statement that
raises on any context is rejected wholesale and leaves no trace, which keeps
the register *types* identical across contexts. That invariant is what lets one
mask be computed for the whole machine instead of one per demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..arc.grid import Grid
from .errors import DslError
from .operators import OPERATORS, OpSpec, Param, is_assignable
from .types import DslType, type_of

__all__ = [
    "MachineSpec",
    "Machine",
    "StepResult",
    "HALT",
    "DEFAULT_LAYOUT",
    "INT_LITERALS",
    "MAX_ARITY",
]

HALT = "HALT"

#: Register ring depth per type. COLOR shares the INTEGER bank because the DSL
#: treats the two as mutually assignable.
DEFAULT_LAYOUT: Tuple[Tuple[DslType, int], ...] = (
    (DslType.GRID, 4),
    (DslType.OBJECT, 2),
    (DslType.OBJECT_SET, 2),
    (DslType.INTEGER, 2),
    (DslType.BOOL, 1),
)

#: Integer literals the policy can name directly. Everything else has to be
#: computed (``HEIGHT``, ``COUNT``, ``ADD``) and referenced through a register,
#: which is exactly the behaviour we want to encourage: input-dependent
#: programs generalise, hard-coded coordinates do not.
INT_LITERALS: Tuple[int, ...] = (
    -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 12, 15, 20, 25, 29, 30,
)

MAX_ARITY = 5

_NAME_PREFIX = {
    DslType.GRID: "g",
    DslType.OBJECT: "o",
    DslType.OBJECT_SET: "s",
    DslType.INTEGER: "n",
    DslType.BOOL: "b",
}


def _bank_type(t: DslType) -> DslType:
    """The register bank a value of type ``t`` lives in."""
    return DslType.INTEGER if t == DslType.COLOR else t


def _type_compatible(entry: "ArgEntry", param: Param) -> bool:
    """Legality of an argument entry ignoring register occupancy.

    Occupancy is the only part that changes as a program is written, so
    everything else is precomputed into ``MachineSpec.static_arg``.
    """
    targets = param.type if isinstance(param.type, tuple) else (param.type,)
    if entry.kind == "default":
        return not param.required
    if entry.kind == "reg":
        return any(is_assignable(entry.reg_type, t) for t in targets)
    if entry.kind == "bool":
        return DslType.BOOL in targets
    if not any(t in (DslType.INTEGER, DslType.COLOR) for t in targets):
        return False
    if param.choices:
        return entry.value in param.choices
    return True


# ---------------------------------------------------------------------------
# Argument vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArgEntry:
    """One entry of the flat argument vocabulary."""

    kind: str  # "reg" | "int" | "bool" | "default"
    reg_type: Optional[DslType] = None
    reg_index: int = 0
    value: Any = None

    def label(self) -> str:
        if self.kind == "reg":
            return f"{_NAME_PREFIX[self.reg_type]}[{self.reg_index}]"
        if self.kind == "int":
            return str(self.value)
        if self.kind == "bool":
            return "true" if self.value else "false"
        return "DEFAULT"


@dataclass(frozen=True)
class MachineSpec:
    """The immutable description of an action space.

    Built once and shared by the environment, the policy and the searcher, so
    action indices mean the same thing everywhere.
    """

    op_names: Tuple[str, ...]
    args: Tuple[ArgEntry, ...]
    layout: Tuple[Tuple[DslType, int], ...]
    max_arity: int

    # -- derived tables, filled in __post_init__ -------------------------
    op_index: Dict[str, int] = field(default_factory=dict, repr=False)
    op_arity: Tuple[int, ...] = field(default=(), repr=False)
    reg_slots: Tuple[Tuple[DslType, int], ...] = field(default=(), repr=False)
    # Static legality tables. Type compatibility never changes, so it is
    # computed once here and the per-step mask becomes two numpy operations
    # instead of ~18,000 Python comparisons.
    static_arg: np.ndarray = field(default=None, repr=False)   # (n_ops, arity, n_args)
    param_valid: np.ndarray = field(default=None, repr=False)  # (n_ops, arity)
    arg_is_reg: np.ndarray = field(default=None, repr=False)   # (n_args,)
    arg_reg_bank: np.ndarray = field(default=None, repr=False) # (n_args,) bank ordinal, -1 if not a reg
    arg_reg_index: np.ndarray = field(default=None, repr=False)
    bank_order: Tuple[DslType, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "op_index", {n: i for i, n in enumerate(self.op_names)})
        arity = []
        for name in self.op_names:
            arity.append(0 if name == HALT else len(OPERATORS[name].params))
        object.__setattr__(self, "op_arity", tuple(arity))
        slots = []
        for t, depth in self.layout:
            for i in range(depth):
                slots.append((t, i))
        object.__setattr__(self, "reg_slots", tuple(slots))

        banks = tuple(t for t, _ in self.layout)
        object.__setattr__(self, "bank_order", banks)
        bank_of = {t: i for i, t in enumerate(banks)}

        n_args = len(self.args)
        is_reg = np.zeros(n_args, dtype=bool)
        reg_bank = np.full(n_args, -1, dtype=np.int64)
        reg_idx = np.zeros(n_args, dtype=np.int64)
        for i, entry in enumerate(self.args):
            if entry.kind == "reg":
                is_reg[i] = True
                reg_bank[i] = bank_of[_bank_type(entry.reg_type)]
                reg_idx[i] = entry.reg_index
        object.__setattr__(self, "arg_is_reg", is_reg)
        object.__setattr__(self, "arg_reg_bank", reg_bank)
        object.__setattr__(self, "arg_reg_index", reg_idx)

        n_ops = len(self.op_names)
        static = np.zeros((n_ops, self.max_arity, n_args), dtype=bool)
        valid = np.zeros((n_ops, self.max_arity), dtype=bool)
        for op_idx, name in enumerate(self.op_names):
            if name == HALT:
                continue
            for p_idx, param in enumerate(OPERATORS[name].params):
                valid[op_idx, p_idx] = True
                for a_idx, entry in enumerate(self.args):
                    static[op_idx, p_idx, a_idx] = _type_compatible(entry, param)
        object.__setattr__(self, "static_arg", static)
        object.__setattr__(self, "param_valid", valid)

    # -- construction ----------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        ops: Optional[Sequence[str]] = None,
        layout: Sequence[Tuple[DslType, int]] = DEFAULT_LAYOUT,
        int_literals: Sequence[int] = INT_LITERALS,
        max_arity: int = MAX_ARITY,
    ) -> "MachineSpec":
        """Build a spec.

        ``ops`` restricts the operator set, which is how the warm-up curriculum
        hands the worker a three-operator world before it sees all of ARC.
        ``INPUT`` is never an action: the machine binds the input grid itself.
        """
        if ops is None:
            names = sorted(n for n in OPERATORS if n != "INPUT")
        else:
            names = []
            for n in ops:
                if n == "INPUT":
                    continue
                if n not in OPERATORS:
                    raise KeyError(f"unknown operator {n!r}")
                if n not in names:
                    names.append(n)
            names.sort()
        for n in names:
            if len(OPERATORS[n].params) > max_arity:
                raise ValueError(f"operator {n} has arity > max_arity={max_arity}")

        entries: List[ArgEntry] = []
        for t, depth in layout:
            for i in range(depth):
                entries.append(ArgEntry(kind="reg", reg_type=t, reg_index=i))
        for v in int_literals:
            entries.append(ArgEntry(kind="int", value=int(v)))
        entries.append(ArgEntry(kind="bool", value=True))
        entries.append(ArgEntry(kind="bool", value=False))
        entries.append(ArgEntry(kind="default"))

        return cls(
            op_names=tuple(names) + (HALT,),
            args=tuple(entries),
            layout=tuple(layout),
            max_arity=max_arity,
        )

    # -- sizes -----------------------------------------------------------
    @property
    def n_ops(self) -> int:
        return len(self.op_names)

    @property
    def n_args(self) -> int:
        return len(self.args)

    @property
    def halt_index(self) -> int:
        return self.op_index[HALT]

    @property
    def n_registers(self) -> int:
        return len(self.reg_slots)

    def spec_of(self, op_idx: int) -> Optional[OpSpec]:
        name = self.op_names[op_idx]
        return None if name == HALT else OPERATORS[name]

    def describe(self, op_idx: int, arg_idx: Sequence[int]) -> str:
        """Render an action the way it will appear in the program text."""
        name = self.op_names[op_idx]
        if name == HALT:
            return HALT
        spec = OPERATORS[name]
        parts = []
        for param, idx in zip(spec.params, arg_idx):
            entry = self.args[int(idx)]
            if entry.kind == "default":
                continue
            parts.append(f"{param.name}={entry.label()}")
        return f"{name}({', '.join(parts)})" if parts else name


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """What happened when one statement was attempted."""

    ok: bool
    halted: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    line: Optional[str] = None
    result_type: Optional[DslType] = None


class Machine:
    """A typed register machine executing one statement at a time.

    Parameters
    ----------
    spec:
        The action space.
    inputs:
        One input grid per context. Context 0 is conventionally the first
        demonstration; the environment decides what the rest are.
    """

    def __init__(self, spec: MachineSpec, inputs: Sequence[Grid]) -> None:
        if not inputs:
            raise ValueError("a machine needs at least one input context")
        self.spec = spec
        self.n_ctx = len(inputs)
        self.depth: Dict[DslType, int] = {t: d for t, d in spec.layout}

        # values[ctx][type] is a newest-first list; names[type] mirrors it.
        self.values: List[Dict[DslType, List[Any]]] = [
            {t: [] for t in self.depth} for _ in range(self.n_ctx)
        ]
        self.names: Dict[DslType, List[str]] = {t: [] for t in self.depth}
        self.counters: Dict[DslType, int] = {t: 0 for t in self.depth}
        self.lines: List[str] = []
        # Provenance, for liveness analysis: what each statement wrote and
        # which earlier values it read. The warm-up generator uses this to
        # reject programs with dead statements.
        self.stmt_targets: List[str] = []
        self.stmt_refs: List[List[str]] = []
        self.n_statements = 0
        self.halted = False

        for ctx, grid in enumerate(inputs):
            self.values[ctx][DslType.GRID].append(np.ascontiguousarray(grid, dtype=np.int8))
        self.names[DslType.GRID].append("g0")
        self.counters[DslType.GRID] = 1
        self.lines.append("g0 = INPUT")

    # -- state -----------------------------------------------------------
    def clone(self) -> "Machine":
        """A copy that can be stepped independently.

        Operators never mutate their arguments (every one of them copies before
        writing), so the value rings can be copied shallowly. Search relies on
        this being cheap.
        """
        other = Machine.__new__(Machine)
        other.spec = self.spec
        other.n_ctx = self.n_ctx
        other.depth = self.depth
        other.values = [{t: list(ring) for t, ring in ctx.items()} for ctx in self.values]
        other.names = {t: list(v) for t, v in self.names.items()}
        other.counters = dict(self.counters)
        other.lines = list(self.lines)
        other.stmt_targets = list(self.stmt_targets)
        other.stmt_refs = [list(r) for r in self.stmt_refs]
        other.n_statements = self.n_statements
        other.halted = self.halted
        return other

    def occupancy(self) -> Dict[DslType, int]:
        """How many registers of each type currently hold a value."""
        return {t: len(self.names[t]) for t in self.depth}

    def canvas(self, ctx: int) -> Grid:
        """The working grid of one context -- the newest Grid register."""
        return self.values[ctx][DslType.GRID][0]

    def canvases(self) -> List[Grid]:
        return [self.canvas(i) for i in range(self.n_ctx)]

    def program_text(self) -> str:
        return "\n".join(self.lines + [f"RETURN {self.names[DslType.GRID][0]}"])

    def live_statements(self) -> int:
        """How many statements the returned canvas actually depends on.

        Backward liveness from the RETURN name. A random rollout that writes
        four statements but returns a grid depending on one of them has really
        written a one-statement program, and the warm-up curriculum must not
        label it as four.
        """
        live = {self.names[DslType.GRID][0]}
        count = 0
        for target, refs in zip(reversed(self.stmt_targets), reversed(self.stmt_refs)):
            if target not in live:
                continue
            count += 1
            live.update(refs)
        return count

    def register_summary(self) -> np.ndarray:
        """Per-slot occupancy flags, in ``spec.reg_slots`` order.

        The policy sees this so it knows which references are live without
        having to infer it from the mask alone.
        """
        occ = self.occupancy()
        return np.array(
            [1.0 if idx < occ[t] else 0.0 for t, idx in self.spec.reg_slots],
            dtype=np.float32,
        )

    # -- masks -----------------------------------------------------------
    def _arg_allowed(self, entry: ArgEntry, param: Param, occ: Dict[DslType, int]) -> bool:
        """Scalar legality check, used to guard an incoming action."""
        if entry.kind == "reg" and entry.reg_index >= occ[_bank_type(entry.reg_type)]:
            return False
        return _type_compatible(entry, param)

    def _available(self) -> np.ndarray:
        """Per-argument-entry availability: literals always, registers if filled."""
        spec = self.spec
        counts = np.array([len(self.names[t]) for t in spec.bank_order], dtype=np.int64)
        avail = np.ones(spec.n_args, dtype=bool)
        reg = spec.arg_is_reg
        avail[reg] = spec.arg_reg_index[reg] < counts[spec.arg_reg_bank[reg]]
        return avail

    def arg_mask(self, op_idx: int, param_idx: int) -> np.ndarray:
        """Legal argument entries for one parameter of one operator."""
        return self.spec.static_arg[op_idx, param_idx] & self._available()

    def arg_masks(self) -> np.ndarray:
        """``(n_ops, max_arity, n_args)`` legality tensor for the current state.

        Static type compatibility comes from the spec; the only thing that
        changes between statements is which registers are occupied.
        """
        return self.spec.static_arg & self._available()[None, None, :]

    def op_mask(self, arg_masks: Optional[np.ndarray] = None) -> np.ndarray:
        """Operators every one of whose parameters is satisfiable.

        HALT has no parameters, so it falls out of the same expression as
        always-legal.
        """
        if arg_masks is None:
            arg_masks = self.arg_masks()
        satisfied = arg_masks.any(axis=2)                 # (n_ops, max_arity)
        return (~self.spec.param_valid | satisfied).all(axis=1)

    # -- execution -------------------------------------------------------
    def _resolve(self, entry: ArgEntry, param: Param, ctx: int) -> Any:
        if entry.kind == "reg":
            return self.values[ctx][entry.reg_type][entry.reg_index]
        if entry.kind == "default":
            return param.default
        return entry.value

    def step(self, op_idx: int, arg_idx: Sequence[int]) -> StepResult:
        """Attempt one statement against every context.

        The statement is committed only when it succeeds everywhere, so a
        failure leaves the machine exactly as it was and the policy can try
        something else on the next step.
        """
        if self.halted:
            return StepResult(ok=False, halted=True, error_code="already_halted")

        name = self.spec.op_names[op_idx]
        if name == HALT:
            self.halted = True
            return StepResult(ok=True, halted=True, line=HALT)

        spec = OPERATORS[name]
        entries = [self.spec.args[int(arg_idx[i])] for i in range(len(spec.params))]

        # Guard the arguments even though the caller is supposed to have
        # sampled under the mask: a bug here would otherwise surface as a
        # confusing runtime error deep inside an operator.
        occ = self.occupancy()
        for param, entry in zip(spec.params, entries):
            if not self._arg_allowed(entry, param, occ):
                return StepResult(
                    ok=False,
                    error_code="illegal_argument",
                    error_message=f"{entry.label()} is not legal for {name}.{param.name}",
                )

        results: List[Any] = []
        for ctx in range(self.n_ctx):
            kwargs = {
                param.name: self._resolve(entry, param, ctx)
                for param, entry in zip(spec.params, entries)
            }
            try:
                value = spec.fn(**kwargs)
            except DslError as exc:
                return StepResult(ok=False, error_code=exc.code, error_message=exc.message)
            except (
                ValueError, TypeError, IndexError, KeyError, ZeroDivisionError,
                OverflowError, MemoryError,
            ) as exc:
                return StepResult(
                    ok=False,
                    error_code="operator_failed",
                    error_message=f"{name} failed: {type(exc).__name__}: {exc}",
                )
            if type_of(value) is None:
                return StepResult(
                    ok=False,
                    error_code="bad_return_value",
                    error_message=f"{name} produced {type(value).__name__}",
                )
            results.append(value)

        # Every context agreed on a type, or the operator is broken.
        ret_type = type_of(results[0])
        if any(type_of(v) != ret_type for v in results[1:]):
            return StepResult(
                ok=False,
                error_code="type_divergence",
                error_message=f"{name} returned different types across demonstrations",
            )

        bank = _bank_type(ret_type)
        depth = self.depth[bank]
        for ctx, value in enumerate(results):
            ring = self.values[ctx][bank]
            ring.insert(0, value)
            del ring[depth:]

        target = f"{_NAME_PREFIX[bank]}{self.counters[bank]}"
        self.counters[bank] += 1
        self.names[bank].insert(0, target)
        del self.names[bank][depth:]

        parts = []
        refs: List[str] = []
        for param, entry in zip(spec.params, entries):
            if entry.kind == "default":
                continue
            if entry.kind == "reg":
                ring_names = self.names[_bank_type(entry.reg_type)]
                # The result was just pushed, so a reference to the bank we
                # wrote has shifted by one position.
                offset = 1 if _bank_type(entry.reg_type) == bank else 0
                pos = entry.reg_index + offset
                label = ring_names[pos] if pos < len(ring_names) else entry.label()
                refs.append(label)
            else:
                label = entry.label()
            parts.append(f"{param.name}={label}")
        call = f"{name}({', '.join(parts)})" if parts else name
        self.lines.append(f"{target} = {call}")
        self.stmt_targets.append(target)
        self.stmt_refs.append(refs)
        self.n_statements += 1

        return StepResult(ok=True, line=self.lines[-1], result_type=ret_type)
