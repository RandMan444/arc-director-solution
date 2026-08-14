"""The v2 operators: the mechanisms the ARC coverage audit said were missing."""

from __future__ import annotations

import numpy as np
import pytest

from arc_director.dsl import OPERATORS
from arc_director.dsl.errors import DslRuntimeError
from arc_director.dsl.ops_v2 import DIRECTIONS
from arc_director.dsl.types import Obj, ObjectSet


def run(name: str, *args, **kwargs):
    return OPERATORS[name].fn(*args, **kwargs)


def test_every_operator_declares_a_signature_and_doc():
    for name, spec in OPERATORS.items():
        assert spec.doc.strip(), f"{name} has no doc"
        assert spec.signature()
        assert len(spec.params) <= 5, f"{name} exceeds MAX_ARITY"


# -- palette ---------------------------------------------------------------


def test_colour_statistics():
    g = np.array([[0, 1, 1], [0, 1, 2], [0, 0, 2]], dtype=np.int8)
    assert run("MOST_COMMON_COLOR", g) == 0
    assert run("MAIN_COLOR", g) == 1
    assert run("RAREST_COLOR", g) == 2
    assert run("COLOR_COUNT", g, 1) == 3
    assert run("PALETTE_SIZE", g) == 3


def test_main_color_rejects_blank():
    with pytest.raises(DslRuntimeError):
        run("MAIN_COLOR", np.zeros((3, 3), dtype=np.int8))


# -- arithmetic and branching ----------------------------------------------


def test_arithmetic_and_branch():
    assert run("ADD", 2, 3) == 5
    assert run("SUB", 2, 3) == -1
    assert run("MUL", 4, 3) == 12
    assert run("DIV", 7, 2) == 3
    with pytest.raises(DslRuntimeError):
        run("DIV", 1, 0)
    assert run("LT", 1, 2) and not run("GT", 1, 2)
    assert run("IF_INT", True, 5, 9) == 5
    a = np.ones((2, 2), dtype=np.int8)
    b = np.zeros((2, 2), dtype=np.int8)
    assert np.array_equal(run("IF_GRID", False, a, b), b)


# -- scaling and splitting -------------------------------------------------


def test_upscale_downscale_roundtrip():
    g = np.array([[1, 2], [3, 4]], dtype=np.int8)
    up = run("UPSCALE", g, 3)
    assert up.shape == (6, 6)
    assert np.array_equal(run("DOWNSCALE", up, 3), g)


def test_part_and_subgrid():
    g = np.arange(12, dtype=np.int8).reshape(3, 4) % 10
    assert np.array_equal(run("PART", g, 2, 0), g[:, :2])
    assert np.array_equal(run("PART", g, 3, 2, vertical=True), g[2:3, :])
    assert np.array_equal(run("SUBGRID", g, 1, 1, 2, 2), g[1:3, 1:3])
    with pytest.raises(DslRuntimeError):
        run("SUBGRID", g, 2, 2, 5, 5)


def test_compress_and_trim():
    g = np.zeros((5, 5), dtype=np.int8)
    g[2, 2] = 4
    assert np.array_equal(run("COMPRESS", g), np.array([[4]], dtype=np.int8))
    assert run("TRIM", g, 1).shape == (3, 3)


def test_cellwise():
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.array([[1, 9], [3, 9]], dtype=np.int8)
    out = run("CELLWISE", a, b, 0)
    assert np.array_equal(out, np.array([[1, 0], [3, 0]], dtype=np.int8))


# -- geometry --------------------------------------------------------------


def test_gravity_stacks_against_the_wall():
    g = np.array([[1, 0], [0, 0], [2, 0]], dtype=np.int8)
    assert np.array_equal(run("GRAVITY", g, 1)[:, 0], np.array([0, 1, 2], dtype=np.int8))
    assert np.array_equal(run("GRAVITY", g, 0)[:, 0], np.array([1, 2, 0], dtype=np.int8))


def test_ray_draws_to_the_edge():
    g = np.zeros((4, 4), dtype=np.int8)
    o = Obj.from_mapping({(1, 1): 3}, (4, 4))
    out = run("RAY", g, o, 3, 5)     # rightwards
    assert np.array_equal(out[1], np.array([0, 0, 5, 5], dtype=np.int8))


def test_symmetrize_fills_only_holes():
    g = np.array([[1, 0], [0, 2]], dtype=np.int8)
    out = run("SYMMETRIZE", g)
    assert np.array_equal(out, np.array([[1, 1], [2, 2]], dtype=np.int8))


def test_mirror_concat_doubles_a_side():
    g = np.array([[1, 2]], dtype=np.int8)
    assert run("MIRROR_CONCAT", g, 3).shape == (1, 4)
    assert run("MIRROR_CONCAT", g, 1).shape == (2, 2)


# -- set algebra and mapping -----------------------------------------------


def _objects(grid):
    return run("COMPONENTS", grid, 4, True)


def test_object_algebra():
    a = Obj.from_mapping({(0, 0): 1, (0, 1): 1}, (3, 3))
    b = Obj.from_mapping({(0, 1): 2, (1, 1): 2}, (3, 3))
    assert run("UNION", a, b).size == 3
    assert run("INTERSECT", a, b).cells == frozenset({(0, 1)})
    assert run("DIFFERENCE", a, b).cells == frozenset({(0, 0)})
    with pytest.raises(DslRuntimeError):
        run("DIFFERENCE", a, a)


def test_sort_by_size_preserves_order_through_nth():
    g = np.zeros((5, 5), dtype=np.int8)
    g[0, 0] = 1
    g[2:5, 2:5] = 2
    objs = _objects(g)
    ordered = run("SORT_BY_SIZE", objs, True)
    assert run("NTH", ordered, 0).size == 9
    assert run("NTH", ordered, 1).size == 1


def test_odd_one_out():
    g = np.zeros((3, 8), dtype=np.int8)
    g[0, 0] = 1
    g[0, 2] = 1
    g[0, 4:6] = 1          # the different shape
    objs = _objects(g)
    assert run("ODD_ONE_OUT", objs).size == 2
    assert run("MOST_COMMON_SHAPE", objs).size == 1


def test_map_operators_are_vectorised():
    g = np.zeros((4, 4), dtype=np.int8)
    g[0, 0] = 1
    g[2, 2] = 1
    objs = _objects(g)
    recoloured = run("MAP_RECOLOR", objs, 7)
    assert all(o.dominant_color() == 7 for o in recoloured)
    moved = run("MAP_MOVE", objs, 1, 0)
    assert {o.bbox[0] for o in moved} == {1, 3}
    boxed = run("MAP_BOX", _objects(np.pad(np.ones((2, 2), dtype=np.int8), 1)))
    assert boxed.objects[0].size == 4


def test_painting_round_trip():
    g = np.zeros((3, 3), dtype=np.int8)
    o = Obj.from_mapping({(1, 1): 5, (1, 2): 6}, (3, 3))
    stamped = run("STAMP", g, o)
    assert stamped[1, 1] == 5 and stamped[1, 2] == 6
    assert (run("ERASE", stamped, o) == 0).all()


def test_mask_of_color_and_foreground():
    g = np.array([[0, 3], [3, 1]], dtype=np.int8)
    assert run("MASK_OF_COLOR", g, 3).size == 2
    assert run("FOREGROUND", g).size == 3
    with pytest.raises(DslRuntimeError):
        run("MASK_OF_COLOR", g, 9)
