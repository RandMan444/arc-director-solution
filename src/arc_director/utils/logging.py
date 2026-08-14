"""Run logging.

Plan section 20: metrics go to TensorBoard *and* to a plain JSONL file, so
analysis never depends on a hosted service or on a running process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

__all__ = ["RunLogger", "read_jsonl"]


class RunLogger:
    """Append-only JSONL logger with an optional TensorBoard mirror.

    Each :meth:`log` call writes one flushed line, so a run that is killed
    mid-training still leaves a complete record up to that point.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        tensorboard: bool = False,
        filename: str = "metrics.jsonl",
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self._writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:
                self._writer = None

    def log(self, step: int, /, **metrics: Any) -> None:
        # Dashboard summaries deliberately carry their own step. Making this
        # argument positional-only lets callers forward those summaries, and
        # the explicit argument remains the authoritative timeline value.
        metrics.pop("step", None)
        record: Dict[str, Any] = {"step": step, **metrics}
        self._fh.write(json.dumps(record, default=_encode) + "\n")
        if self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)

    def log_dict(self, step: int, payload: Dict[str, Any], prefix: str = "") -> None:
        """Log a nested dict, flattening it with ``/`` separators."""
        self.log(step, **_flatten(payload, prefix))

    def close(self) -> None:
        self._fh.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _flatten(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def _encode(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return str(obj)


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
