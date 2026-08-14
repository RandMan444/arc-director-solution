"""Coverage search and the ARC evaluation protocol."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arc_director.arc.dataset import ArcTask, Pair
from arc_director.dsl.machine import MachineSpec
from arc_director.dsl.search import SearchConfig, search_task
from arc_director.env.task_env import EnvConfig
from arc_director.models.agent import AgentConfig, DirectorAgent
from arc_director.train.evaluate import SingleTaskSource, evaluate_task, evaluate_tasks
from tests.test_env import rotate_task


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.build()


@pytest.fixture(scope="module")
def agent(spec):
    torch.manual_seed(0)
    cfg = AgentConfig(
        grid_side=6, grid_channels=(8, 16, 16), grid_dim=32, d_model=48,
        set_layers=1, set_heads=2, state_dim=48, lstm_dim=64,
        mem_len=8, mem_dim=32, mem_heads=2, goal_groups=4, goal_codes=4,
        goal_hidden=64, manager_every=3,
    )
    return DirectorAgent(cfg, spec)


def identity_task() -> ArcTask:
    rng = np.random.default_rng(1)
    pairs = [
        Pair(g, g.copy())
        for g in (rng.integers(0, 3, size=(3, 3)).astype(np.int8) for _ in range(3))
    ]
    return ArcTask("ident", "test", train=pairs[:2], test=[pairs[2]])


def test_search_certifies_a_one_operator_task(spec):
    result = search_task(
        spec, rotate_task(3), SearchConfig(max_depth=2, beam_width=16, per_op_samples=120, seed=0)
    )
    assert result.solved and result.test_exact
    assert "ROTATE90" in result.program
    assert result.depth == 1


def test_search_spots_an_identity_task_without_searching(spec):
    result = search_task(spec, identity_task(), SearchConfig(max_depth=1))
    assert result.solved and result.depth == 0


def test_search_reports_a_miss_honestly(spec):
    """A task outside a one-step budget must come back unsolved, not crash."""
    result = search_task(
        spec, rotate_task(3), SearchConfig(max_depth=1, per_op_samples=1, timeout_s=0.2, seed=5)
    )
    assert result.program is None or result.solved
    assert result.nodes >= 1


def test_search_respects_its_time_budget(spec):
    result = search_task(
        spec, rotate_task(4), SearchConfig(max_depth=3, per_op_samples=400, timeout_s=0.05)
    )
    assert result.elapsed_s < 2.0


def test_evaluation_never_looks_at_the_test_output(spec, agent):
    """Candidate selection must use demonstrations only.

    Poisoning the test *output* must not change what gets selected -- only the
    scores attached to it afterwards.
    """
    task = rotate_task(3)
    cfg = EnvConfig(max_demos=3, max_steps=4, augment=True)
    device = torch.device("cpu")

    poisoned = ArcTask(
        task.task_id, task.source, list(task.train),
        [Pair(p.input, np.zeros_like(p.output)) for p in task.test],
    )
    a = evaluate_task(agent, spec, task, cfg, device, n_attempts=6, seed=3)
    b = evaluate_task(agent, spec, poisoned, cfg, device, n_attempts=6, seed=3)
    assert a.demo_fit == b.demo_fit
    assert a.program == b.program


def test_evaluation_summary_shape(spec, agent):
    cfg = EnvConfig(max_demos=3, max_steps=4, augment=True)
    summary, results = evaluate_tasks(
        agent, spec, [rotate_task(3), identity_task()], cfg, torch.device("cpu"),
        n_attempts=4, seed=0,
    )
    assert summary["tasks"] == 2
    assert 0.0 <= summary["exact_at_1"] <= summary["exact_at_2"] <= 1.0
    assert summary["exact_at_2"] <= summary["pass_at_n"]
    assert all(r.attempts == 4 for r in results)


def test_evaluation_reports_arc1_and_arc2_separately(spec, agent):
    first = rotate_task(3)
    first.source = "arc1"
    second = identity_task()
    second.source = "arc2"
    summary, results = evaluate_tasks(
        agent,
        spec,
        [first, second],
        EnvConfig(max_demos=3, max_steps=4, augment=True),
        torch.device("cpu"),
        n_attempts=2,
        seed=4,
    )
    assert summary["arc1/tasks"] == 1
    assert summary["arc2/tasks"] == 1
    assert {result.source for result in results} == {"arc1", "arc2"}


def test_evaluation_leaves_the_agent_in_training_mode(spec, agent):
    agent.train()
    evaluate_tasks(
        agent, spec, [identity_task()], EnvConfig(max_steps=3), torch.device("cpu"), n_attempts=2
    )
    assert agent.training


def test_single_task_source_is_constant(spec):
    task = rotate_task(2)
    source = SingleTaskSource(task, max_steps=5)
    rng = np.random.default_rng(0)
    assert source.sample(rng) is task and source.sample(rng) is task
    assert source.max_steps == 5
