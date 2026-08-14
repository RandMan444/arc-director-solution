"""The register machine: masks, execution, provenance, program text."""

from __future__ import annotations

import numpy as np
import pytest

from arc_director.dsl.machine import HALT, Machine, MachineSpec
from arc_director.dsl.operators import OPERATORS
from arc_director.dsl.types import DslType


@pytest.fixture(scope="module")
def spec() -> MachineSpec:
    return MachineSpec.build()


@pytest.fixture
def grid() -> np.ndarray:
    g = np.zeros((5, 5), dtype=np.int8)
    g[1:3, 1:3] = 3
    g[4, 0] = 7
    return g


def arg_index(spec: MachineSpec, kind: str, **kwargs) -> int:
    for i, entry in enumerate(spec.args):
        if entry.kind != kind:
            continue
        if all(getattr(entry, k) == v for k, v in kwargs.items()):
            return i
    raise AssertionError(f"no {kind} entry matching {kwargs}")


def test_spec_covers_every_operator(spec):
    assert spec.n_ops == len(OPERATORS)  # every op except INPUT, plus HALT
    assert "INPUT" not in spec.op_names
    assert spec.op_names[spec.halt_index] == HALT


def test_a_wrongly_typed_argument_is_refused(spec, grid):
    """The guard catches an action that did not come from the mask."""
    m = Machine(spec, [grid])
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    result = m.step(spec.op_index["COMPONENTS"], [g0, g0, g0])   # a Grid as `conn`
    assert not result.ok and result.error_code == "illegal_argument"


def test_masks_only_offer_occupied_registers(spec, grid):
    m = Machine(spec, [grid])
    masks = m.arg_masks()
    for i, entry in enumerate(spec.args):
        if entry.kind == "reg" and entry.reg_type != DslType.GRID:
            assert not masks[:, :, i].any(), f"{entry.label()} offered before it exists"
        if entry.kind == "reg" and entry.reg_type == DslType.GRID and entry.reg_index > 0:
            assert not masks[:, :, i].any()


def test_op_mask_and_arg_mask_agree(spec, grid):
    m = Machine(spec, [grid])
    masks = m.arg_masks()
    ops = m.op_mask(masks)
    for op_idx in np.flatnonzero(ops):
        op_spec = spec.spec_of(int(op_idx))
        if op_spec is None:
            continue
        for p in range(len(op_spec.params)):
            assert masks[op_idx, p].any(), f"{spec.op_names[op_idx]} param {p} unsatisfiable"


def test_sampled_actions_always_execute(spec, grid):
    """Under the mask, only genuine runtime errors may fail -- never type errors."""
    rng = np.random.default_rng(0)
    bad_codes = {"illegal_argument", "bad_return_value", "type_divergence"}
    for _ in range(60):
        m = Machine(spec, [grid, grid[:3, :3].copy()])
        for _ in range(6):
            masks = m.arg_masks()
            ops = m.op_mask(masks)
            ops[spec.halt_index] = False
            op = int(rng.choice(np.flatnonzero(ops)))
            args = [0] * spec.max_arity
            for p in range(len(spec.spec_of(op).params)):
                args[p] = int(rng.choice(np.flatnonzero(masks[op, p])))
            result = m.step(op, args)
            assert result.error_code not in bad_codes, (spec.op_names[op], result.error_message)


def test_failed_statement_leaves_no_trace(spec):
    """LARGEST of an empty set must fail without disturbing the machine."""
    m = Machine(spec, [np.zeros((3, 3), dtype=np.int8)])
    default = arg_index(spec, "default")
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    assert m.step(spec.op_index["COMPONENTS"], [g0, default, default]).ok
    before_names = {t: list(v) for t, v in m.names.items()}
    before_lines = list(m.lines)

    result = m.step(spec.op_index["LARGEST"], [arg_index(spec, "reg", reg_type=DslType.OBJECT_SET, reg_index=0)])
    assert not result.ok
    assert {t: list(v) for t, v in m.names.items()} == before_names
    assert m.lines == before_lines


def test_statement_applies_to_every_context(spec, grid):
    other = np.rot90(grid).copy()
    m = Machine(spec, [grid, other])
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    assert m.step(spec.op_index["ROTATE90"], [g0]).ok
    assert np.array_equal(m.canvas(0), np.rot90(grid, k=-1))
    assert np.array_equal(m.canvas(1), np.rot90(other, k=-1))


def test_ring_registers_shift(spec, grid):
    m = Machine(spec, [grid])
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    m.step(spec.op_index["ROTATE90"], [g0])
    m.step(spec.op_index["FLIP_H"], [g0])
    # Slot 0 is the newest; the original input has slid to slot 2.
    assert np.array_equal(m.values[0][DslType.GRID][2], grid)
    assert m.names[DslType.GRID][0] == "g2"


def test_liveness_counts_only_reachable_statements(spec, grid):
    m = Machine(spec, [grid])
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    g1 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=1)
    m.step(spec.op_index["ROTATE90"], [g0])   # live
    m.step(spec.op_index["FLIP_H"], [g1])     # reads the original, orphaning the rotate
    assert m.live_statements() == 1


def test_program_text_is_readable(spec, grid):
    m = Machine(spec, [grid])
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    m.step(spec.op_index["COMPONENTS"], [g0, arg_index(spec, "default"), arg_index(spec, "default")])
    m.step(spec.op_index["LARGEST"], [arg_index(spec, "reg", reg_type=DslType.OBJECT_SET, reg_index=0)])
    m.step(spec.op_index["CROP"], [arg_index(spec, "reg", reg_type=DslType.OBJECT, reg_index=0)])
    text = m.program_text().splitlines()
    assert text[0] == "g0 = INPUT"
    assert text[-1] == "RETURN g1"
    assert "LARGEST" in text[2]


def test_clone_is_independent(spec, grid):
    m = Machine(spec, [grid])
    g0 = arg_index(spec, "reg", reg_type=DslType.GRID, reg_index=0)
    other = m.clone()
    other.step(spec.op_index["ROTATE90"], [g0])
    assert m.n_statements == 0
    assert np.array_equal(m.canvas(0), grid)


def test_halt_is_always_legal(spec, grid):
    m = Machine(spec, [grid])
    assert m.op_mask()[spec.halt_index]
    assert m.step(spec.halt_index, [0] * spec.max_arity).halted


def test_restricted_spec_shrinks_the_action_space(grid):
    small = MachineSpec.build(ops=["ROTATE90", "FLIP_H"])
    assert small.n_ops == 3  # two operators plus HALT
    m = Machine(small, [grid])
    assert m.op_mask().all()
