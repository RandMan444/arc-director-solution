"""Environment: observation shape, reward shaping, held-out demo, augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from arc_director.arc.dataset import ArcTask, Pair
from arc_director.arc.grid import grids_equal
from arc_director.curriculum.generator import GenConfig, generate_task
from arc_director.dsl.machine import MachineSpec
from arc_director.dsl.types import DslType
from arc_director.env.task_env import (
    PAIR_FEATURE_DIM,
    ROLE_DEMO,
    ROLE_HELDOUT,
    ROLE_TEST,
    EnvConfig,
    ProgramEnv,
)
from arc_director.env.vec import VecProgramEnv


class FixedSource:
    allowed_ops = None
    max_steps = None

    def __init__(self, task: ArcTask) -> None:
        self.task = task
        self.reports = []

    def sample(self, rng):
        return self.task

    def report(self, info):
        self.reports.append(info)

    def stats(self):
        return {"source": "fixed"}


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.build()


def rotate_task(n_demos: int = 4) -> ArcTask:
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(n_demos + 1):
        g = rng.integers(0, 4, size=(4, 4)).astype(np.int8)
        pairs.append(Pair(g, np.ascontiguousarray(np.rot90(g, k=-1))))
    return ArcTask("rot", "test", train=pairs[:-1], test=[pairs[-1]])


def arg_index(spec, kind, **kwargs):
    for i, entry in enumerate(spec.args):
        if entry.kind == kind and all(getattr(entry, k) == v for k, v in kwargs.items()):
            return i
    raise AssertionError(kind)


def test_observation_shapes(spec):
    cfg = EnvConfig(max_demos=3, augment=False)
    env = ProgramEnv(spec, FixedSource(rotate_task()), cfg)
    obs = env.reset()
    slots = cfg.n_slots
    assert obs.grids.shape == (slots, 3, 30, 30)
    assert obs.pair_feats.shape == (slots, PAIR_FEATURE_DIM)
    assert obs.op_mask.shape == (spec.n_ops,)
    assert obs.arg_mask.shape == (spec.n_ops, spec.max_arity, spec.n_args)
    assert obs.slot_mask.sum() == len(env.contexts)


def test_roles_and_holdout(spec):
    env = ProgramEnv(spec, FixedSource(rotate_task(4)), EnvConfig(max_demos=3, augment=False))
    env.reset()
    roles = [c.role for c in env.contexts]
    assert roles.count(ROLE_DEMO) == 3
    assert roles.count(ROLE_HELDOUT) == 1
    assert roles.count(ROLE_TEST) == 1


def test_holdout_never_starves_the_visible_set(spec):
    """A two-demonstration task keeps both; a rule from one pair is a guess."""
    env = ProgramEnv(
        spec,
        FixedSource(rotate_task(2)),
        EnvConfig(max_demos=4, min_visible_demos=2, augment=False),
    )
    env.reset()
    assert sum(c.role == ROLE_DEMO for c in env.contexts) == 2
    assert sum(c.role == ROLE_HELDOUT for c in env.contexts) == 0


def test_test_target_is_never_observable(spec):
    env = ProgramEnv(spec, FixedSource(rotate_task()), EnvConfig(augment=False))
    obs = env.reset()
    test_slot = [i for i, c in enumerate(env.contexts) if c.role == ROLE_TEST][0]
    assert (obs.grids[test_slot, 2] == 0).all()
    assert (obs.grid_shapes[test_slot, 2] == 0).all()


def test_solving_pays_and_identity_does_not(spec):
    """Rotating solves the task; doing nothing costs the step price."""
    env = ProgramEnv(spec, FixedSource(rotate_task()), EnvConfig(augment=False, max_steps=4))
    env.reset()
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    _, r1, done, _ = env.step(spec.op_index["ROTATE90"], [g0] + [0] * (spec.max_arity - 1))
    assert not done and r1 > 0
    _, r2, done, info = env.step(spec.halt_index, [0] * spec.max_arity)
    assert done and info["solved_demos"] and info["solved_heldout"] and info["solved_test"]
    assert r2 > 1.0

    env2 = ProgramEnv(spec, FixedSource(rotate_task()), EnvConfig(augment=False, max_steps=4))
    env2.reset()
    _, r, done, info = env2.step(spec.halt_index, [0] * spec.max_arity)
    assert done and not info["solved_demos"] and r < 0


def test_error_is_penalised_and_state_survives(spec):
    env = ProgramEnv(spec, FixedSource(rotate_task()), EnvConfig(augment=False))
    env.reset()
    before = env.machine.canvas(0).copy()
    # MAIN_COLOR on a blank grid is a runtime error, not a type error.
    blank = spec.op_index["CANVAS_LIKE"]
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    default = arg_index(spec, "default")
    env.step(blank, [g0, default] + [0] * (spec.max_arity - 2))
    _, reward, _, info = env.step(
        spec.op_index["MAIN_COLOR"], [arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)]
        + [0] * (spec.max_arity - 1)
    )
    assert info["error_code"] == "blank_grid"
    assert reward < 0
    assert grids_equal(env.machine.canvas(0), np.zeros_like(before))


def test_answer_is_mapped_out_of_augmented_space(spec):
    """``answer()`` must undo the episode's augmentation exactly."""
    env = ProgramEnv(spec, FixedSource(rotate_task()), EnvConfig(augment=True, max_steps=4))
    env.reset()
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    env.step(spec.op_index["FLIP_H"], [g0] + [0] * (spec.max_arity - 1))
    assert grids_equal(env.augmentation.apply(env.answer()), env.predict_test())


@pytest.mark.parametrize("seed", range(8))
def test_augmentation_preserves_solvability(spec, seed):
    """An augmented rotation task is still solved by *some* single dihedral op.

    This is the property the whole augmentation scheme rests on: conjugating a
    rule by a bijection leaves a task of the same difficulty. If it failed, the
    endless-data story would be quietly training the agent on noise.
    """
    dihedral = ["ROTATE90", "ROTATE180", "ROTATE270", "FLIP_H", "FLIP_V", "TRANSPOSE", "COPY"]
    solved = []
    for name in dihedral:
        env = ProgramEnv(
            spec, FixedSource(rotate_task()), EnvConfig(augment=True, max_steps=4), seed=seed
        )
        env.reset()
        g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
        env.step(spec.op_index[name], [g0] + [0] * (spec.max_arity - 1))
        _, _, _, info = env.step(spec.halt_index, [0] * spec.max_arity)
        if info["solved_demos"]:
            solved.append(name)
            assert info["solved_test"], f"{name} fits the demos but not the test pair"
    assert len(solved) == 1, f"expected exactly one solving operator, got {solved}"


def test_source_is_told_about_every_episode(spec):
    source = FixedSource(rotate_task())
    env = ProgramEnv(spec, source, EnvConfig(augment=False, max_steps=2))
    env.reset()
    for _ in range(2):
        env.step(spec.halt_index, [0] * spec.max_arity)
        env.reset()
    assert len(source.reports) == 2


def test_vec_env_autoresets(spec):
    env = VecProgramEnv(4, spec, FixedSource(rotate_task()), EnvConfig(augment=False, max_steps=2))
    env.reset()
    ops = np.full(4, spec.halt_index)
    args = np.zeros((4, spec.max_arity), dtype=np.int64)
    obs, reward, done, infos = env.step(ops, args)
    assert done.all()
    assert obs["grids"].shape[0] == 4
    assert all("final_observation" in i for i in infos)
    assert env.episode_count == 4


def test_generated_task_is_solvable_by_its_own_program(spec):
    """The warm-up generator's contract: the task is reachable at the stated length."""
    rng = np.random.default_rng(3)
    made = None
    while made is None:
        made = generate_task(rng, spec, GenConfig(n_ops=2, n_demos=3, max_side=6))
    assert made.task.num_demos == 3
    assert len(made.program.splitlines()) == 2 + made.n_ops  # INPUT + ops + RETURN
