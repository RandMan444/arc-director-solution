"""The trainer: replay fidelity, two-timescale credit, checkpoints, curriculum."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from arc_director.config import build, load_config
from arc_director.curriculum.sources import CurriculumSource
from arc_director.curriculum.stages import DEFAULT_LADDER, Stage, group_ops, ops_mask
from arc_director.dsl.machine import MachineSpec
from arc_director.train.director import (
    DirectorTrainer,
    RunningNorm,
    epiplexity_auc,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    cfg = load_config(CONFIG)
    run_dir = tmp_path_factory.mktemp("run")
    torch.manual_seed(0)
    return build(cfg, run_dir=str(run_dir))


def test_config_rejects_unknown_keys():
    cfg = load_config(CONFIG)
    cfg["train"]["learning_rate"] = 1e-3
    with pytest.raises(KeyError, match="learning_rate"):
        build(cfg)


def test_director_proper_rejects_a_hybrid_worker_contract(tmp_path):
    cfg = load_config(CONFIG)
    cfg["train"]["director_proper"] = True
    with pytest.raises(ValueError, match="director_proper contract violated"):
        build(cfg, run_dir=str(tmp_path / "invalid_director"))


def test_director_proper_is_one_step_and_worker_reward_is_goal_only(tmp_path):
    cfg = load_config(CONFIG)
    cfg["agent"].update(manager_every=1, worker_env_feedback=False)
    cfg["train"].update(
        director_proper=True,
        worker_goal_weight=1.0,
        worker_extrinsic_weight=0.0,
        worker_bootstrap_across_goals=False,
    )
    _, _, _, trainer = build(cfg, run_dir=str(tmp_path / "proper_director"))
    buffer = trainer.collect()
    assert buffer.tensor("manager_active").all(), "Director must act at every step"
    stats = trainer.update(buffer)
    assert stats["reward/worker_mean"] == pytest.approx(
        stats["reward/goal_mean"], abs=1e-7
    )


def test_config_rejects_an_encoder_too_small_for_the_ladder():
    cfg = load_config(CONFIG)
    cfg["agent"]["grid_side"] = 4      # the ladder generates 6x6 grids
    with pytest.raises(ValueError, match="grid_side"):
        build(cfg)


def test_config_rejects_memory_shorter_than_an_episode():
    cfg = load_config(CONFIG)
    cfg["agent"]["mem_len"] = 2
    with pytest.raises(ValueError, match="mem_len"):
        build(cfg)


def test_collect_then_update_runs_and_stays_finite(built):
    spec, env, agent, trainer = built
    buffer = trainer.collect()
    assert len(buffer) == trainer.cfg.n_steps
    stats = trainer.update(buffer)
    numbers = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    assert all(np.isfinite(v) for v in numbers.values()), stats
    assert stats["actions/top_op"] in trainer.spec.op_names
    assert 0.0 < stats["actions/top_op_share"] <= 1.0
    assert all(torch.isfinite(p).all() for p in agent.parameters())


def test_the_update_replays_exactly_what_was_acted_on(built):
    """The one invariant a recurrent on-policy trainer cannot get wrong.

    If the update's forward pass diverges from the acting pass -- a state reset
    in the wrong place, a stale feedback vector, a mismatched manager code --
    the gradient is computed for a policy that never ran, and nothing about the
    loss curve would say so.
    """
    spec, env, agent, trainer = built
    for _ in range(3):
        buffer = trainer.collect()
        trainer.update(buffer)
        assert trainer.last_replay_drift < 1e-4, trainer.last_replay_drift


def test_manager_targets_accumulate_to_the_window_boundary(built):
    """Hand-checked case: rewards 1,1,1,1 with a goal boundary at step 2."""
    _, _, _, trainer = built
    gamma = trainer.cfg.gamma
    rewards = torch.ones(4, 1)
    dones = torch.zeros(4, 1)
    values = torch.full((4, 1), 5.0)
    active = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    boot = torch.zeros(1)

    targets = trainer._manager_targets(rewards, dones, values, active, boot)
    # From step 0: r0 + gamma*(r1 + gamma*V(step 2)), because a new goal starts at 2.
    expected0 = 1 + gamma * (1 + gamma * 5.0)
    assert targets[0].item() == pytest.approx(expected0, rel=1e-5)
    # From step 2 the window runs to the end of the segment with no bootstrap.
    assert targets[2].item() == pytest.approx(1 + gamma * 1, rel=1e-5)


def test_episode_end_cuts_the_manager_window(built):
    _, _, _, trainer = built
    rewards = torch.ones(3, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    values = torch.full((3, 1), 9.0)
    active = torch.tensor([[1.0], [0.0], [1.0]])
    targets = trainer._manager_targets(rewards, dones, values, active, torch.zeros(1))
    gamma = trainer.cfg.gamma
    # Step 1 terminates, so nothing after it is credited to the goal set at 0.
    assert targets[0].item() == pytest.approx(1 + gamma * 1, rel=1e-5)


def test_gae_matches_a_hand_computation(built):
    _, _, _, trainer = built
    cfg = trainer.cfg
    rewards = torch.tensor([[1.0], [0.0]])
    values = torch.tensor([[0.5], [0.25]])
    dones = torch.zeros(2, 1)
    last = torch.tensor([0.0])
    adv, ret = trainer._gae(rewards, values, dones, last)
    delta1 = 0.0 + cfg.gamma * 0.0 - 0.25
    delta0 = 1.0 + cfg.gamma * 0.25 - 0.5
    assert adv[1].item() == pytest.approx(delta1, rel=1e-5)
    assert adv[0].item() == pytest.approx(
        delta0 + cfg.gamma * cfg.gae_lambda * delta1, rel=1e-5
    )
    assert torch.allclose(ret, adv + values)


def test_checkpoint_round_trip(built, tmp_path):
    spec, env, agent, trainer = built
    trainer.update(trainer.collect())
    path = tmp_path / "ckpt.pt"
    trainer.save(path)

    before = {k: v.clone() for k, v in agent.state_dict().items()}
    for p in agent.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    trainer.load(path)
    after = agent.state_dict()
    for key, value in before.items():
        assert torch.allclose(value, after[key]), key


def test_epiplexity_auc_is_the_sum_of_the_episode_delta_list():
    assert epiplexity_auc([0.4, -0.1, 0.25]) == pytest.approx(0.55)


def test_crossed_epiplexity_duel_runs_and_counts_both_forks(tmp_path):
    cfg = load_config(CONFIG)
    cfg["train"].update(
        n_envs=2,
        n_steps=4,
        epiplexity_duel=True,
        epiplexity_duel_updates=1,
    )
    _, env, agent, trainer = build(cfg, run_dir=str(tmp_path / "duel"))
    stats, episodes = trainer.duel_update()

    assert trainer.updates == 2
    assert trainer.env_steps == 2 * 2 * 4
    assert trainer.duel_rounds == 1
    assert stats["epiplexity/worker_winner"] in {"explore", "exploit"}
    assert stats["epiplexity/director_winner"] in {"explore", "exploit"}
    for key in (
        "epiplexity/worker_explore_auc",
        "epiplexity/worker_exploit_auc",
        "epiplexity/director_explore_auc",
        "epiplexity/director_exploit_auc",
    ):
        assert np.isfinite(stats[key]), (key, stats[key])
    assert isinstance(episodes, list)
    assert all(program_env.steps == 0 for program_env in env.envs)
    assert all(torch.isfinite(parameter).all() for parameter in agent.parameters())


def test_running_norm_tracks_mean_and_variance():
    norm = RunningNorm()
    data = torch.randn(5000) * 3.0 + 7.0
    for chunk in data.split(500):
        norm.update(chunk)
    assert norm.mean == pytest.approx(7.0, abs=0.3)
    assert norm.var**0.5 == pytest.approx(3.0, abs=0.3)


# -- curriculum ------------------------------------------------------------


def test_curriculum_promotes_only_after_the_floor_and_the_rate():
    spec = MachineSpec.build()
    stages = (
        Stage("a", "warmup", ("geometry",), (1,), max_side=6, promote_at=0.5,
              promote_window=10, min_episodes=20),
        Stage("b", "warmup", ("geometry", "color"), (2,), max_side=6,
              promote_at=1.01, min_episodes=10**9),
    )
    source = CurriculumSource(spec, stages)
    for _ in range(15):
        source.report({"solved_demos": True, "generalized": True, "has_heldout": True})
    assert source.index == 0, "promoted before the episode floor"
    for _ in range(10):
        source.report({"solved_demos": True, "generalized": True, "has_heldout": True})
    assert source.index == 1
    assert source.stage.name == "b"
    assert source.history[0]["stage"] == "a"


def test_curriculum_does_not_promote_on_demo_fit_alone():
    """Fitting the visible demonstrations is not evidence; generalising is."""
    spec = MachineSpec.build()
    stages = (
        Stage("a", "warmup", ("geometry",), (1,), max_side=6, promote_at=0.5,
              promote_window=10, min_episodes=10),
        Stage("b", "warmup", ("geometry",), (1,), max_side=6),
    )
    source = CurriculumSource(spec, stages)
    for _ in range(50):
        source.report(
            {"solved_demos": True, "generalized": False, "has_heldout": True}
        )
    assert source.index == 0


def test_stage_operator_masks_restrict_the_action_space():
    spec = MachineSpec.build()
    stage = DEFAULT_LADDER[0]
    mask = ops_mask(spec, stage.op_names())
    allowed = {spec.op_names[i] for i in np.flatnonzero(mask)}
    assert "ROTATE90" in allowed
    assert "HALT" in allowed
    assert "COMPONENTS" not in allowed
    assert len(allowed) == len(set(group_ops("geometry")))


def test_curriculum_state_round_trip():
    spec = MachineSpec.build()
    source = CurriculumSource(spec, DEFAULT_LADDER[:2])
    for _ in range(5):
        source.report({"solved_demos": False, "generalized": False})
    blob = source.state_dict()
    restored = CurriculumSource(spec, DEFAULT_LADDER[:2])
    restored.load_state_dict(blob)
    assert restored.index == source.index
    assert restored.total_episodes == source.total_episodes
    assert list(restored.recent) == list(source.recent)
    assert restored.rate == source.rate


def test_delta_goal_reward_pays_nothing_for_standing_still(built):
    """The reason `delta` is the default.

    With the absolute form the worker collects reward every step for being near
    a goal it is already at, which both rewards inaction and makes ending the
    episode costly -- so HALT never gets emitted. The delta form telescopes to
    (final - initial) similarity: no movement, no reward.
    """
    _, _, _, trainer = built
    goals = torch.randn(4, 3, 16)
    states = goals.clone()          # already at the goal
    trainer.cfg.worker_goal_reward = "delta"
    assert torch.allclose(
        trainer.goal_reward(goals, states, states), torch.zeros(4, 3), atol=1e-5
    )
    trainer.cfg.worker_goal_reward = "absolute"
    assert (trainer.goal_reward(goals, states, states) > 0.9).all()
    trainer.cfg.worker_goal_reward = "delta"


def test_delta_goal_reward_pays_for_approaching(built):
    _, _, _, trainer = built
    trainer.cfg.worker_goal_reward = "delta"
    goal = torch.tensor([[[1.0, 0.0]]])
    far = torch.tensor([[[0.0, 1.0]]])
    near = torch.tensor([[[0.9, 0.1]]])
    assert trainer.goal_reward(goal, far, near).item() > 0
    assert trainer.goal_reward(goal, near, far).item() < 0


def test_unknown_goal_reward_mode_is_refused(built):
    _, _, _, trainer = built
    trainer.cfg.worker_goal_reward = "cosine"
    with pytest.raises(ValueError, match="worker_goal_reward"):
        trainer.goal_reward(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
    trainer.cfg.worker_goal_reward = "delta"


def test_dev_holdout_is_disjoint_and_deterministic(tmp_path):
    """The internal dev split must never appear in a training pool.

    It is the only number in a run that is not measured on data the agent
    trains on, so an overlap would quietly turn it into a training metric.
    """
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "full.yaml")
    cfg["train"]["n_envs"] = 2
    cfg["curriculum"]["reachable_file"] = None
    cfg["curriculum"]["ladder"] = [
        {"name": "arc", "kind": "arc", "arc_filter": "small", "max_steps": 6}
    ]
    cfg["agent"] = {"grid_side": 30, "mem_len": 8}
    cfg["train"]["director_proper"] = False

    _, env, _, trainer = build(cfg, run_dir=str(tmp_path / "a"))
    dev_ids = {t.task_id for t in trainer.dev_tasks}
    pool_ids = {t.task_id for t in env.source.inner.tasks}
    assert len(dev_ids) == 100
    assert not (dev_ids & pool_ids)
    assert {t.source for t in trainer.dev_tasks[:40]} == {"arc1", "arc2"}

    _, _, _, again = build(cfg, run_dir=str(tmp_path / "b"))
    assert {t.task_id for t in again.dev_tasks} == dev_ids


def test_the_shipped_configs_are_checkpoint_compatible():
    """`--resume runs/warmup_small/checkpoint.pt --config full.yaml` must work.

    Only `grid_side` (how much of the padded grid the encoder looks at) and
    `manager_every` (a schedule, not a shape) may differ. Widening the network
    in one file and not the other silently breaks the documented resume path.
    """
    from arc_director.models.agent import AgentConfig, DirectorAgent

    root = Path(__file__).resolve().parents[1] / "configs"
    warm = AgentConfig(**load_config(root / "warmup_small.yaml")["agent"])
    full = AgentConfig(**load_config(root / "full.yaml")["agent"])
    spec = MachineSpec.build()
    a, b = DirectorAgent(warm, spec).state_dict(), DirectorAgent(full, spec).state_dict()
    assert a.keys() == b.keys()
    bad = [k for k in a if a[k].shape != b[k].shape]
    assert not bad, f"shape-incompatible parameters: {bad}"


def test_load_refuses_a_checkpoint_of_the_wrong_shape(built, tmp_path):
    from arc_director.models.agent import AgentConfig, DirectorAgent

    spec, _, agent, trainer = built
    path = tmp_path / "wide.pt"
    trainer.save(path)
    blob = torch.load(path, weights_only=False)
    wide = AgentConfig(**{**blob["agent_config"], "lstm_dim": agent.cfg.lstm_dim * 2})
    blob["agent"] = DirectorAgent(wide, spec).state_dict()
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="different network shape"):
        trainer.load(path)
