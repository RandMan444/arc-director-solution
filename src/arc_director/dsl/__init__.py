"""The DSL: v0 core operators, the v2 extensions, and the register machine.

Importing this package is what registers every operator, so
``from ..dsl import OPERATORS`` always sees the full set.
"""

from . import operators, ops_v2  # noqa: F401  (import for the registration side effect)
from .errors import DslError, DslLimitError, DslRuntimeError, DslSyntaxError, DslTypeError
from .machine import (
    DEFAULT_LAYOUT,
    HALT,
    INT_LITERALS,
    MAX_ARITY,
    Machine,
    MachineSpec,
    StepResult,
)
from .operators import OPERATORS, OpSpec, Param, get_op, op_names
from .types import DslType, Obj, ObjectSet, type_of

__all__ = [
    "OPERATORS",
    "OpSpec",
    "Param",
    "get_op",
    "op_names",
    "DslType",
    "Obj",
    "ObjectSet",
    "type_of",
    "Machine",
    "MachineSpec",
    "StepResult",
    "HALT",
    "MAX_ARITY",
    "DEFAULT_LAYOUT",
    "INT_LITERALS",
    "DslError",
    "DslLimitError",
    "DslRuntimeError",
    "DslSyntaxError",
    "DslTypeError",
]
