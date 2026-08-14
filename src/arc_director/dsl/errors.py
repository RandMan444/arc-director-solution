"""DSL error hierarchy.

Every failure carries a short machine-readable ``code`` in addition to its
message. The environment surfaces the code (not the prose) to the policy, so
feedback stays compact and stable across refactors.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["DslError", "DslSyntaxError", "DslTypeError", "DslRuntimeError", "DslLimitError"]


class DslError(Exception):
    code = "dsl_error"

    def __init__(self, message: str, *, line: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.line = line
        if code:
            self.code = code

    def __str__(self) -> str:
        where = f" (line {self.line})" if self.line is not None else ""
        return f"{self.message}{where}"

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "line": self.line}


class DslSyntaxError(DslError):
    """The text is not a well-formed program."""

    code = "syntax_error"


class DslTypeError(DslError):
    """The program parses but is not type-consistent."""

    code = "type_error"


class DslRuntimeError(DslError):
    """An operator failed on the actual values it was given."""

    code = "runtime_error"


class DslLimitError(DslRuntimeError):
    """A resource limit was hit (grid too large, too many steps)."""

    code = "limit_exceeded"
