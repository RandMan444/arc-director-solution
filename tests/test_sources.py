"""Reliability checks for long-running task sources."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import arc_director.curriculum.sources as sources
from arc_director.curriculum.generator import GenConfig
from arc_director.curriculum.sources import WarmupSource
from arc_director.dsl.machine import MachineSpec
from tests.test_env import rotate_task


def test_warmup_source_survives_a_rejection_streak_longer_than_the_old_limit(
    monkeypatch,
):
    calls = 0

    def delayed_success(*_args):
        nonlocal calls
        calls += 1
        if calls <= 75:
            return None
        return SimpleNamespace(task=rotate_task(3), program="RETURN g0")

    monkeypatch.setattr(sources, "generate_task", delayed_success)
    source = WarmupSource(
        MachineSpec.build(), GenConfig(), n_ops_choices=(2, 3), name="w3_objects"
    )

    task = source.sample(np.random.default_rng(0))

    assert task.task_id == "w3_objects_0000001"
    assert source.generated == 1
    assert source.rejected == 75
    assert source.rejection_streak == 0
    assert source.max_rejection_streak == 75


def test_warmup_source_still_detects_an_invalid_custom_stage(monkeypatch):
    monkeypatch.setattr(sources, "generate_task", lambda *_args: None)
    source = WarmupSource(MachineSpec.build(), GenConfig(), max_tries=3, name="invalid")

    with pytest.raises(RuntimeError, match="after 3 candidate rejections"):
        source.sample(np.random.default_rng(0))


def test_warmup_source_rejects_invalid_retry_configuration():
    spec = MachineSpec.build()
    with pytest.raises(ValueError, match="program length"):
        WarmupSource(spec, GenConfig(), n_ops_choices=())
    with pytest.raises(ValueError, match="positive"):
        WarmupSource(spec, GenConfig(), max_tries=0)
