"""ARC task loading and the committed train/dev split manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .augment import Augmentation, sample_augmentation
from .grid import Grid, to_grid, to_lists

__all__ = ["Pair", "ArcTask", "ArcDataset", "SplitManifest", "make_split"]


@dataclass(frozen=True)
class Pair:
    """One demonstration or test input/output pair."""

    input: Grid
    output: Optional[Grid]  # None for held-out test pairs with hidden answers

    @property
    def has_output(self) -> bool:
        return self.output is not None


@dataclass
class ArcTask:
    """A single ARC task: demonstration pairs plus one or more test pairs."""

    task_id: str
    source: str
    train: List[Pair]
    test: List[Pair]

    # -- construction ---------------------------------------------------
    @classmethod
    def from_json(cls, data: dict, task_id: str, source: str = "") -> "ArcTask":
        def read(section: str) -> List[Pair]:
            out = []
            for item in data.get(section, []):
                inp = to_grid(item["input"])
                if section == "train" and ("output" not in item or item["output"] is None):
                    raise ValueError(f"task {task_id} has a demonstration without an output")
                out_grid = (
                    to_grid(item["output"])
                    if "output" in item and item["output"] is not None
                    else None
                )
                out.append(Pair(inp, out_grid))
            return out

        train, test = read("train"), read("test")
        if not train:
            raise ValueError(f"task {task_id} has no demonstration pairs")
        if not test:
            raise ValueError(f"task {task_id} has no test pairs")
        return cls(task_id=task_id, source=source, train=train, test=test)

    @classmethod
    def from_file(cls, path: Path, source: str = "") -> "ArcTask":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_json(data, task_id=path.stem, source=source or path.parent.name)

    def to_json(self) -> dict:
        def dump(pairs: Sequence[Pair]) -> list:
            items = []
            for p in pairs:
                d = {"input": to_lists(p.input)}
                if p.output is not None:
                    d["output"] = to_lists(p.output)
                items.append(d)
            return items

        return {"train": dump(self.train), "test": dump(self.test)}

    # -- properties -----------------------------------------------------
    @property
    def uid(self) -> str:
        """Globally unique key, since ARC1 and ARC2 can share task stems."""
        return f"{self.source}__{self.task_id}" if self.source else self.task_id

    @property
    def num_demos(self) -> int:
        return len(self.train)

    def demo_pairs(self) -> List[Tuple[Grid, Grid]]:
        """Demonstration pairs with outputs guaranteed present."""
        return [(p.input, p.output) for p in self.train if p.output is not None]

    # -- augmentation ---------------------------------------------------
    def augmented(self, aug: Augmentation) -> "ArcTask":
        """Return a new task with ``aug`` applied consistently everywhere."""

        def tf(p: Pair) -> Pair:
            return Pair(
                aug.apply(p.input),
                aug.apply(p.output) if p.output is not None else None,
            )

        return ArcTask(
            task_id=f"{self.task_id}#{aug.key()}",
            source=self.source,
            train=[tf(p) for p in self.train],
            test=[tf(p) for p in self.test],
        )

    def sample_augmented(
        self, rng: np.random.Generator, *, reorder_demos: bool = True, **kwargs
    ) -> "ArcTask":
        """Draw a random rule-preserving variant of this task.

        ``reorder_demos`` shuffles demonstration order, which the LLM sees as a
        different prompt while the underlying rule is untouched.
        """
        task = self.augmented(sample_augmentation(rng, **kwargs))
        if reorder_demos and len(task.train) > 1:
            order = rng.permutation(len(task.train))
            task.train = [task.train[i] for i in order]
        return task

    def leave_one_out(self) -> Iterator["ArcTask"]:
        """Re-root the task by promoting each demonstration to be the test pair.

        Yields ``num_demos`` variants (only when at least 2 demos remain), each
        a legitimate task with the same hidden rule. Cheap extra supervision
        that costs no new labelling.
        """
        if self.num_demos < 3:
            return
        for i in range(self.num_demos):
            held = self.train[i]
            rest = self.train[:i] + self.train[i + 1 :]
            yield ArcTask(
                task_id=f"{self.task_id}@loo{i}",
                source=self.source,
                train=rest,
                test=[held],
            )


class ArcDataset:
    """A collection of tasks loaded from one or more folders."""

    def __init__(self, tasks: Sequence[ArcTask]):
        self.tasks: List[ArcTask] = list(tasks)
        self._by_uid: Dict[str, ArcTask] = {t.uid: t for t in self.tasks}
        if len(self._by_uid) != len(self.tasks):
            raise ValueError("duplicate task uids in dataset")

    @classmethod
    def from_folders(cls, folders: Dict[str, str | Path]) -> "ArcDataset":
        """Load ``{source_name: folder}`` into one dataset, sorted for determinism."""
        tasks: List[ArcTask] = []
        for source, folder in folders.items():
            path = Path(folder)
            if not path.is_dir():
                raise FileNotFoundError(f"ARC folder not found: {path}")
            for file in sorted(path.glob("*.json")):
                tasks.append(ArcTask.from_file(file, source=source))
        return cls(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> ArcTask:
        return self.tasks[idx]

    def __iter__(self) -> Iterator[ArcTask]:
        return iter(self.tasks)

    @property
    def uids(self) -> List[str]:
        return [t.uid for t in self.tasks]

    def by_uid(self, uid: str) -> ArcTask:
        return self._by_uid[uid]

    def subset(self, uids: Sequence[str]) -> "ArcDataset":
        return ArcDataset([self._by_uid[u] for u in uids])

    def counts_by_source(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in self.tasks:
            counts[t.source] = counts.get(t.source, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Split manifest
# ---------------------------------------------------------------------------


@dataclass
class SplitManifest:
    """A committed, reproducible train/dev split over the ARC *training* tasks.

    The official public evaluation set is deliberately not represented here.
    Development decisions are made against ``dev`` only (plan section 4).
    """

    seed: int
    source: str
    train: List[str]
    dev: List[str]
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum:
            self.checksum = self.compute_checksum()

    def compute_checksum(self) -> str:
        payload = json.dumps(
            {"seed": self.seed, "source": self.source, "train": self.train, "dev": self.dev},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def verify(self) -> None:
        if self.checksum != self.compute_checksum():
            raise ValueError("split manifest checksum mismatch - the file was edited")
        if len(set(self.train)) != len(self.train) or len(set(self.dev)) != len(self.dev):
            raise ValueError("split manifest contains duplicate task ids")
        overlap = set(self.train) & set(self.dev)
        if overlap:
            raise ValueError(f"train/dev overlap: {sorted(overlap)[:5]}")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "seed": self.seed,
                    "source": self.source,
                    "checksum": self.checksum,
                    "n_train": len(self.train),
                    "n_dev": len(self.dev),
                    "train": self.train,
                    "dev": self.dev,
                },
                fh,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not data.get("checksum"):
            raise ValueError("split manifest has no checksum")
        if "n_train" in data and int(data["n_train"]) != len(data["train"]):
            raise ValueError("split manifest n_train does not match its task list")
        if "n_dev" in data and int(data["n_dev"]) != len(data["dev"]):
            raise ValueError("split manifest n_dev does not match its task list")
        m = cls(
            seed=data["seed"],
            source=data["source"],
            train=data["train"],
            dev=data["dev"],
            checksum=data["checksum"],
        )
        m.verify()
        return m


def make_split(
    dataset: ArcDataset,
    *,
    n_dev: int = 100,
    seed: int = 20260811,
    source: str = "",
    dev_pool_uids: Optional[Sequence[str]] = None,
) -> SplitManifest:
    """Deterministic split, optionally drawing dev from an eligible UID pool."""
    uids = sorted(dataset.uids)
    if n_dev < 0:
        raise ValueError(f"n_dev must be non-negative, got {n_dev}")
    if n_dev >= len(uids):
        raise ValueError(f"n_dev={n_dev} but dataset only has {len(uids)} tasks")
    candidates = sorted(dev_pool_uids) if dev_pool_uids is not None else uids
    unknown = sorted(set(candidates) - set(uids))
    if unknown:
        raise ValueError(f"dev pool contains unknown task ids: {unknown[:5]}")
    if n_dev > len(candidates):
        raise ValueError(f"n_dev={n_dev} but dev pool only has {len(candidates)} tasks")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    dev = sorted(candidates[i] for i in order[:n_dev])
    dev_set = set(dev)
    train = sorted(uid for uid in uids if uid not in dev_set)
    return SplitManifest(
        seed=seed, source=source or ",".join(sorted(dataset.counts_by_source())), train=train, dev=dev
    )
