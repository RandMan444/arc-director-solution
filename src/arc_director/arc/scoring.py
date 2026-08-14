"""Scoring. Exact match is the metric that counts; the rest shapes the reward.

ARC is pixel-exact, so :func:`exact_match` is the only number reported as
"solve rate". Everything else here exists to build the structured executor
feedback (plan section 8) and the shaped reward (plan section 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .grid import Grid, bounding_box, grids_equal

__all__ = ["PairScore", "score_pair", "TaskScore", "score_task", "exact_match"]


def exact_match(pred: Optional[Grid], target: Grid) -> bool:
    return pred is not None and grids_equal(pred, target)


@dataclass
class PairScore:
    """How a predicted grid compares to one target grid."""

    exact: bool
    shape_correct: bool
    cell_accuracy: float
    balanced_cell_accuracy: float
    pred_shape: Optional[tuple]
    target_shape: tuple
    mismatched_colors: Dict[int, int] = field(default_factory=dict)
    mismatch_bbox: Optional[List[int]] = None

    def as_dict(self) -> dict:
        return {
            "exact": self.exact,
            "shape_correct": self.shape_correct,
            "cell_accuracy": round(self.cell_accuracy, 4),
            "balanced_cell_accuracy": round(self.balanced_cell_accuracy, 4),
            "pred_shape": list(self.pred_shape) if self.pred_shape else None,
            "target_shape": list(self.target_shape),
            "mismatched_colors": self.mismatched_colors,
            "mismatch_bbox": self.mismatch_bbox,
        }


def score_pair(pred: Optional[Grid], target: Grid) -> PairScore:
    """Compare a prediction against a target.

    When shapes disagree the cell accuracy is computed over the *overlapping*
    region and then scaled by the area ratio. That keeps the signal continuous:
    a prediction that is right but one row too tall scores well above noise,
    while a wildly mis-sized one is pushed toward zero. A hard 0.0 on any shape
    mismatch would flatten most of the early learning signal.
    """
    target_shape = tuple(int(x) for x in target.shape)

    if pred is None:
        return PairScore(
            exact=False,
            shape_correct=False,
            cell_accuracy=0.0,
            balanced_cell_accuracy=0.0,
            pred_shape=None,
            target_shape=target_shape,
        )

    pred_shape = tuple(int(x) for x in pred.shape)
    shape_correct = pred_shape == target_shape

    if shape_correct:
        eq = pred == target
        accuracy = float(eq.mean())
        mismatched: Dict[int, int] = {}
        if not eq.all():
            wrong_targets = target[~eq]
            values, counts = np.unique(wrong_targets, return_counts=True)
            mismatched = {int(v): int(c) for v, c in zip(values, counts)}
        box = bounding_box(~eq)
        return PairScore(
            exact=bool(eq.all()),
            shape_correct=True,
            cell_accuracy=accuracy,
            balanced_cell_accuracy=_balanced_cell_accuracy(pred, target),
            pred_shape=pred_shape,
            target_shape=target_shape,
            mismatched_colors=mismatched,
            mismatch_bbox=list(box) if box else None,
        )

    # Shapes differ: score the overlap, discounted by how much area we missed.
    h = min(pred_shape[0], target_shape[0])
    w = min(pred_shape[1], target_shape[1])
    overlap_matches = int((pred[:h, :w] == target[:h, :w]).sum())
    pred_area = pred_shape[0] * pred_shape[1]
    target_area = target_shape[0] * target_shape[1]
    # Denominator is the larger area, so extra predicted cells are penalised too.
    accuracy = overlap_matches / max(pred_area, target_area)
    return PairScore(
        exact=False,
        shape_correct=False,
        cell_accuracy=float(accuracy),
        balanced_cell_accuracy=_balanced_cell_accuracy(pred, target),
        pred_shape=pred_shape,
        target_shape=target_shape,
    )


@dataclass
class TaskScore:
    """Aggregate of per-demonstration scores for one candidate program."""

    pair_scores: List[PairScore]

    @property
    def n(self) -> int:
        return len(self.pair_scores)

    @property
    def all_exact(self) -> bool:
        return self.n > 0 and all(s.exact for s in self.pair_scores)

    @property
    def exact_fraction(self) -> float:
        return sum(s.exact for s in self.pair_scores) / self.n if self.n else 0.0

    @property
    def shape_fraction(self) -> float:
        return sum(s.shape_correct for s in self.pair_scores) / self.n if self.n else 0.0

    @property
    def mean_cell_accuracy(self) -> float:
        return sum(s.cell_accuracy for s in self.pair_scores) / self.n if self.n else 0.0

    @property
    def mean_balanced_cell_accuracy(self) -> float:
        """Macro-average colors so abundant background cannot dominate reward."""
        return (
            sum(s.balanced_cell_accuracy for s in self.pair_scores) / self.n
            if self.n
            else 0.0
        )

    @property
    def shaping_accuracy(self) -> float:
        """Dense progress signal used for within-episode improvement."""
        return (self.mean_cell_accuracy + 2.0 * self.mean_balanced_cell_accuracy) / 3.0

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n,
            "all_exact": self.all_exact,
            "exact_fraction": round(self.exact_fraction, 4),
            "shape_fraction": round(self.shape_fraction, 4),
            "mean_cell_accuracy": round(self.mean_cell_accuracy, 4),
            "mean_balanced_cell_accuracy": round(self.mean_balanced_cell_accuracy, 4),
            "pairs": [s.as_dict() for s in self.pair_scores],
        }


def score_task(preds: Sequence[Optional[Grid]], targets: Sequence[Grid]) -> TaskScore:
    if len(preds) != len(targets):
        raise ValueError(f"got {len(preds)} predictions for {len(targets)} targets")
    return TaskScore([score_pair(p, t) for p, t in zip(preds, targets)])


def _balanced_cell_accuracy(pred: Grid, target: Grid) -> float:
    """Per-target-color recall with an extra-area penalty.

    A blank prediction can score highly under raw pixel accuracy because ARC
    grids contain much background. Giving every target color equal weight
    preserves continuous credit without letting background count hundreds of
    times more than a small object.
    """
    ph, pw = pred.shape
    th, tw = target.shape
    h, w = min(ph, th), min(pw, tw)
    correct = np.zeros(target.shape, dtype=bool)
    correct[:h, :w] = pred[:h, :w] == target[:h, :w]
    recalls = [float(correct[target == color].mean()) for color in np.unique(target)]
    target_area, pred_area = int(target.size), int(pred.size)
    extra_area_penalty = target_area / max(target_area, pred_area)
    return float(np.mean(recalls) * extra_area_penalty)
