"""The program-synthesis environment.

One episode is one attempt to write a DSL program for one task. The agent
drives a :class:`~arc_director.dsl.machine.Machine` that holds a value bank for
every demonstration at once, so a statement is only accepted when it runs on
all of them -- the same discipline a human program synthesiser follows.

How the 1-10 demonstrations problem is handled
----------------------------------------------
An ARC task carries between 2 and 10 demonstration pairs (97% have <= 5) and
1-4 test inputs. Three mechanisms cover that variability:

1. **Contexts, not concatenation.** Every demonstration is its own machine
   context. The statement is executed on all of them; the reward is the mean
   over the visible ones. Adding a demonstration adds a context, it does not
   change the action space.

2. **A masked set encoder.** The policy sees a padded stack of context slots
   plus a mask, and pools over them permutation-invariantly (models/epn.py).
   Two demonstrations and six demonstrations are the same computation with a
   different mask, exactly the way the episodic-memory transformer in the A2C
   EPN work handled a variable-length memory.

3. **Resampling.** Tasks with more demonstrations than ``max_demos`` have a
   fresh subset drawn each episode, so nothing is permanently discarded; over
   training the agent sees every pair.

The test input is carried as an extra context so its statements are executed
too -- a program that crashes on the test input is worthless -- but it never
contributes to reward, and its output is never in the observation.

Held-out demonstration
----------------------
With ``holdout_demo`` one demonstration is withheld from the reward and used
only as a metric. It is the honest in-training answer to "did the agent find
the rule or overfit the demonstrations it was scored on", and it is available
long before test accuracy moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..arc.augment import sample_augmentation
from ..arc.dataset import ArcTask
from ..arc.grid import BACKGROUND, MAX_SIDE, NUM_COLORS, Grid, grids_equal
from ..arc.scoring import score_pair
from ..dsl.machine import Machine, MachineSpec

__all__ = ["EnvConfig", "ProgramEnv", "Observation", "ROLE_DEMO", "ROLE_HELDOUT", "ROLE_TEST"]

ROLE_DEMO = 0
ROLE_HELDOUT = 1
ROLE_TEST = 2
N_ROLES = 3

#: Length of the per-context feature vector built by ``_pair_features``.
PAIR_FEATURE_DIM = 3 + N_ROLES + 6 + 4 + 3


@dataclass
class EnvConfig:
    """Everything about episode shape and reward.

    The defaults are tuned for the warm-up curriculum: short episodes, dense
    shaping, a solve bonus large enough to dominate the shaping total.
    """

    max_demos: int = 4
    min_visible_demos: int = 2
    include_test_ctx: bool = True
    holdout_demo: bool = True
    max_steps: int = 12
    augment: bool = True
    permute_colors: bool = True

    # reward
    step_cost: float = 0.02
    error_penalty: float = 0.10
    shaping_scale: float = 2.0
    solve_bonus: float = 5.0
    heldout_bonus: float = 0.0  # kept out of reward by default; see module docs
    length_bonus: float = 0.0   # optional reward for short solving programs

    #: Attach the de-augmented test prediction to the terminal info. Off during
    #: training (it would pin a grid per episode in the log buffer); the
    #: evaluator turns it on, because that is the moment the answer exists.
    report_answer: bool = False

    def __post_init__(self) -> None:
        if self.max_demos < 1:
            raise ValueError("max_demos must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @property
    def n_slots(self) -> int:
        """Padded context count: visible demos + held-out demo + test input."""
        return self.max_demos + 2


@dataclass
class Observation:
    """One environment observation, as plain numpy.

    Kept as arrays rather than tensors so the environment stays torch-free and
    can be pickled into worker processes later.
    """

    grids: np.ndarray        # (S, 3, 30, 30) int8: input, canvas, target
    grid_shapes: np.ndarray  # (S, 3, 2) int16
    slot_mask: np.ndarray    # (S,) float32
    pair_feats: np.ndarray   # (S, PAIR_FEATURE_DIM) float32
    reg_occ: np.ndarray      # (n_registers,) float32
    scalars: np.ndarray      # (SCALAR_DIM,) float32
    last_action: np.ndarray  # (1 + max_arity,) int64
    op_mask: np.ndarray      # (n_ops,) bool
    arg_mask: np.ndarray     # (n_ops, max_arity, n_args) bool

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "grids": self.grids,
            "grid_shapes": self.grid_shapes,
            "slot_mask": self.slot_mask,
            "pair_feats": self.pair_feats,
            "reg_occ": self.reg_occ,
            "scalars": self.scalars,
            "last_action": self.last_action,
            "op_mask": self.op_mask,
            "arg_mask": self.arg_mask,
        }


SCALAR_DIM = 8


def _pad_grid(g: Optional[Grid]) -> Tuple[np.ndarray, Tuple[int, int]]:
    out = np.zeros((MAX_SIDE, MAX_SIDE), dtype=np.int8)
    if g is None:
        return out, (0, 0)
    h, w = int(g.shape[0]), int(g.shape[1])
    out[:h, :w] = g
    return out, (h, w)


def _color_hist(g: Grid) -> np.ndarray:
    counts = np.bincount(np.asarray(g, dtype=np.int64).ravel(), minlength=NUM_COLORS)
    return counts.astype(np.float32) / max(1, g.size)


@dataclass
class _Context:
    """One machine context: an input grid, a role, and maybe a target."""

    input: Grid
    target: Optional[Grid]
    role: int


class ProgramEnv:
    """Single-episode program synthesis over one ARC (or synthetic) task.

    The environment is deliberately *not* vectorised internally; batching lives
    in :mod:`arc_director.env.vec`, which keeps this class simple enough to
    step by hand in a notebook.
    """

    def __init__(
        self,
        spec: MachineSpec,
        task_source,
        config: Optional[EnvConfig] = None,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.source = task_source
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)

        self.task: Optional[ArcTask] = None
        self.augmentation = None
        self.contexts: List[_Context] = []
        self.machine: Optional[Machine] = None
        self.steps = 0
        self.prev_score = 0.0
        self.last_action = np.zeros(1 + spec.max_arity, dtype=np.int64)
        self.last_error: Optional[str] = None
        self.episode_return = 0.0

    # -- episode lifecycle -----------------------------------------------
    def reset(self, task: Optional[ArcTask] = None) -> Observation:
        task = task if task is not None else self.source.sample(self.rng)
        self.augmentation = None
        if self.cfg.augment:
            self.augmentation = sample_augmentation(
                self.rng, permute_colors=self.cfg.permute_colors
            )
            task = task.augmented(self.augmentation)
        self.task = task

        demos = [(p.input, p.output) for p in task.train if p.output is not None]
        order = self.rng.permutation(len(demos))
        demos = [demos[i] for i in order]

        n_visible = min(self.cfg.max_demos, len(demos))
        heldout: Optional[Tuple[Grid, Grid]] = None
        if self.cfg.holdout_demo:
            if len(demos) > n_visible:
                # Spare demonstrations exist; hold one out for free.
                heldout = demos[n_visible]
            elif n_visible > self.cfg.min_visible_demos:
                # Otherwise borrow one, but never drop below the floor: a rule
                # induced from a single pair is guesswork, not generalisation.
                n_visible -= 1
                heldout = demos[n_visible]

        contexts = [_Context(i, o, ROLE_DEMO) for i, o in demos[:n_visible]]
        if heldout is not None:
            contexts.append(_Context(heldout[0], heldout[1], ROLE_HELDOUT))
        if self.cfg.include_test_ctx and task.test:
            pair = task.test[int(self.rng.integers(len(task.test)))]
            contexts.append(_Context(pair.input, pair.output, ROLE_TEST))

        self.contexts = contexts
        self.machine = Machine(self.spec, [c.input for c in contexts])
        # A curriculum stage may shorten the episode and restrict the operator
        # set; both are read from the source so the ladder stays in one place.
        self.max_steps = int(getattr(self.source, "max_steps", None) or self.cfg.max_steps)
        self.allowed_ops = getattr(self.source, "allowed_ops", None)
        self.steps = 0
        self.last_action[:] = 0
        self.last_error = None
        self.episode_return = 0.0
        self.prev_score = self._score()
        self.initial_score = self.prev_score
        return self.observe()

    # -- reward ----------------------------------------------------------
    def _visible(self) -> List[int]:
        return [i for i, c in enumerate(self.contexts) if c.role == ROLE_DEMO]

    def _score(self) -> float:
        """Mean shaping accuracy of the canvas against the visible targets."""
        idx = self._visible()
        if not idx:
            return 0.0
        total = 0.0
        for i in idx:
            s = score_pair(self.machine.canvas(i), self.contexts[i].target)
            total += (s.cell_accuracy + 2.0 * s.balanced_cell_accuracy) / 3.0
        return total / len(idx)

    def _solved(self, role: int) -> bool:
        idx = [i for i, c in enumerate(self.contexts) if c.role == role]
        if not idx:
            return False
        return all(
            self.contexts[i].target is not None
            and grids_equal(self.machine.canvas(i), self.contexts[i].target)
            for i in idx
        )

    # -- stepping --------------------------------------------------------
    def step(self, op_idx: int, arg_idx: Sequence[int]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.machine is None:
            raise RuntimeError("call reset() before step()")

        result = self.machine.step(int(op_idx), [int(a) for a in arg_idx])
        self.steps += 1
        self.last_action[0] = int(op_idx)
        self.last_action[1:] = [int(a) for a in arg_idx][: self.spec.max_arity]
        self.last_error = result.error_code

        reward = -self.cfg.step_cost
        if not result.ok:
            reward -= self.cfg.error_penalty

        score = self._score()
        reward += self.cfg.shaping_scale * (score - self.prev_score)
        self.prev_score = score

        done = bool(result.halted) or self.steps >= self.max_steps
        info: Dict[str, Any] = {"error_code": result.error_code, "score": score}

        if done:
            solved = self._solved(ROLE_DEMO)
            if solved:
                reward += self.cfg.solve_bonus
                if self.cfg.length_bonus:
                    slack = max(0, self.max_steps - self.machine.n_statements)
                    reward += self.cfg.length_bonus * slack / self.max_steps
            heldout_ok = self._solved(ROLE_HELDOUT)
            if solved and heldout_ok and self.cfg.heldout_bonus:
                reward += self.cfg.heldout_bonus
            has_heldout = any(c.role == ROLE_HELDOUT for c in self.contexts)
            info.update(
                solved_demos=solved,
                solved_heldout=heldout_ok,
                has_heldout=has_heldout,
                solved_test=self._solved(ROLE_TEST),
                generalized=bool(solved and (heldout_ok or not has_heldout)),
                n_statements=self.machine.n_statements,
                halted=bool(result.halted),
                program=self.machine.program_text(),
                task_id=self.task.task_id if self.task else "",
                final_score=score,
                score_gain=score - self.initial_score,
                episode_return=self.episode_return + reward,
            )
            if self.cfg.report_answer:
                info["answer"] = self.answer()
            self.source.report(info)

        self.episode_return += reward
        return self.observe(), float(reward), done, info

    # -- observation -----------------------------------------------------
    def _pair_features(self, slot: int) -> np.ndarray:
        c = self.contexts[slot]
        canvas = self.machine.canvas(slot)
        feats = np.zeros(PAIR_FEATURE_DIM, dtype=np.float32)
        at = 0
        feats[at] = 1.0; at += 1                                   # slot is real
        feats[at] = 1.0 if c.target is not None and c.role != ROLE_TEST else 0.0; at += 1
        # Whether this pair preserves grid size: 65% of ARC-2 pairs do, and it
        # is the single most useful bit for deciding between "edit the grid"
        # and "build a new one".
        same_size = (
            c.target is not None and c.role != ROLE_TEST and c.input.shape == c.target.shape
        )
        feats[at] = float(same_size); at += 1
        feats[at + c.role] = 1.0; at += N_ROLES
        feats[at] = c.input.shape[0] / MAX_SIDE; at += 1
        feats[at] = c.input.shape[1] / MAX_SIDE; at += 1
        feats[at] = canvas.shape[0] / MAX_SIDE; at += 1
        feats[at] = canvas.shape[1] / MAX_SIDE; at += 1
        target = c.target if c.role != ROLE_TEST else None
        feats[at] = (target.shape[0] / MAX_SIDE) if target is not None else 0.0; at += 1
        feats[at] = (target.shape[1] / MAX_SIDE) if target is not None else 0.0; at += 1
        if target is not None:
            s = score_pair(canvas, target)
            feats[at] = float(s.exact); at += 1
            feats[at] = float(s.shape_correct); at += 1
            feats[at] = s.cell_accuracy; at += 1
            feats[at] = s.balanced_cell_accuracy; at += 1
        else:
            at += 4
        feats[at] = float(grids_equal(canvas, c.input)); at += 1
        feats[at] = float(np.unique(canvas).size) / NUM_COLORS; at += 1
        feats[at] = float((canvas != BACKGROUND).mean()); at += 1
        return feats

    def observe(self) -> Observation:
        slots = self.cfg.n_slots
        grids = np.zeros((slots, 3, MAX_SIDE, MAX_SIDE), dtype=np.int8)
        shapes = np.zeros((slots, 3, 2), dtype=np.int16)
        slot_mask = np.zeros(slots, dtype=np.float32)
        feats = np.zeros((slots, PAIR_FEATURE_DIM), dtype=np.float32)

        for i, c in enumerate(self.contexts[:slots]):
            canvas = self.machine.canvas(i)
            target = c.target if c.role != ROLE_TEST else None
            for j, g in enumerate((c.input, canvas, target)):
                padded, shape = _pad_grid(g)
                grids[i, j] = padded
                shapes[i, j] = shape
            slot_mask[i] = 1.0
            feats[i] = self._pair_features(i)

        arg_mask = self.machine.arg_masks()
        op_mask = self.machine.op_mask(arg_mask)
        if self.allowed_ops is not None:
            op_mask &= self.allowed_ops
            op_mask[self.spec.halt_index] = True   # HALT is never withheld

        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        scalars[0] = self.steps / self.max_steps
        scalars[1] = 1.0 - self.steps / self.max_steps
        scalars[2] = self.machine.n_statements / self.max_steps
        scalars[3] = self.prev_score
        scalars[4] = float(self.last_error is not None)
        scalars[5] = len(self._visible()) / self.cfg.max_demos
        scalars[6] = float(self.machine.halted)
        scalars[7] = self.prev_score - self.initial_score

        return Observation(
            grids=grids,
            grid_shapes=shapes,
            slot_mask=slot_mask,
            pair_feats=feats,
            reg_occ=self.machine.register_summary(),
            scalars=scalars,
            last_action=self.last_action.copy(),
            op_mask=op_mask,
            arg_mask=arg_mask,
        )

    # -- evaluation helper ------------------------------------------------
    def predict_test(self) -> Optional[Grid]:
        """The canvas of the test context, i.e. the agent's answer."""
        for i, c in enumerate(self.contexts):
            if c.role == ROLE_TEST:
                return self.machine.canvas(i)
        return None

    def answer(self) -> Optional[Grid]:
        """The test prediction mapped back out of augmented space.

        Episodes run on an augmented copy of the task, so a raw prediction is
        in the wrong frame. Inverting here is what lets several augmented
        attempts vote on one answer.
        """
        prediction = self.predict_test()
        if prediction is None:
            return None
        if self.augmentation is None:
            return prediction
        return self.augmentation.invert(prediction)
