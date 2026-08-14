"""The networks: masking, permutation invariance, the goal space."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arc_director.dsl.machine import MachineSpec
from arc_director.env.task_env import EnvConfig
from arc_director.env.vec import VecProgramEnv
from arc_director.models.agent import AgentConfig, DirectorAgent
from arc_director.models.goal import GoalAutoencoder, max_cosine
from arc_director.train.rollout import to_tensors
from tests.test_env import FixedSource, rotate_task


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.build()


@pytest.fixture(scope="module")
def agent(spec):
    cfg = AgentConfig(
        grid_side=6, grid_channels=(8, 16, 16), grid_dim=32, d_model=48,
        set_layers=1, set_heads=2, state_dim=48, lstm_dim=64,
        mem_len=8, mem_dim=32, mem_heads=2, goal_groups=4, goal_codes=4,
        goal_hidden=64, manager_every=3,
    )
    torch.manual_seed(0)
    return DirectorAgent(cfg, spec)


def make_obs(spec, n_envs=4, max_demos=3, seed=0):
    env = VecProgramEnv(
        n_envs, spec, FixedSource(rotate_task()),
        EnvConfig(max_demos=max_demos, augment=False, max_steps=6), seed=seed,
    )
    return env, to_tensors(env.reset(), torch.device("cpu"))


# -- action masking --------------------------------------------------------


def test_sampled_actions_respect_the_masks(spec, agent):
    env, obs = make_obs(spec)
    state = agent.initial_state(4, torch.device("cpu"))
    feedback = torch.zeros(4, 2)
    for _ in range(5):
        with torch.no_grad():
            out = agent.step(obs, state, feedback)
        for b in range(4):
            op = int(out.op[b])
            assert obs["op_mask"][b, op], f"illegal operator {spec.op_names[op]}"
            for p in range(spec.max_arity):
                mask = obs["arg_mask"][b, op, p]
                if mask.any():
                    assert mask[int(out.args[b, p])], "illegal argument"
                else:
                    assert int(out.args[b, p]) == 0, "unused parameter must stay at 0"
        next_obs, *_ = env.step(out.op.numpy(), out.args.numpy())
        obs = to_tensors(next_obs, torch.device("cpu"))
        state = out.next_state


def test_reevaluating_an_action_reproduces_its_logprob(spec, agent):
    _, obs = make_obs(spec)
    state = agent.initial_state(4, torch.device("cpu"))
    feedback = torch.zeros(4, 2)
    with torch.no_grad():
        sampled = agent.step(obs, state, feedback)
        replayed = agent.step(
            obs, state, feedback,
            actions=(sampled.op, sampled.args),
            manager_action=sampled.manager_codes,
        )
    assert torch.allclose(sampled.worker_logp, replayed.worker_logp, atol=1e-5)
    assert torch.allclose(sampled.manager_logp, replayed.manager_logp, atol=1e-5)


# -- the demonstration set -------------------------------------------------


def test_trunk_is_permutation_invariant_over_demonstrations(spec, agent):
    """Demonstrations are a set. Reordering them must not change the state."""
    _, obs = make_obs(spec, n_envs=2)
    with torch.no_grad():
        base = agent.trunk(obs)
    order = torch.tensor([2, 0, 1, 3, 4])
    shuffled = dict(obs)
    for key in ("grids", "grid_shapes", "slot_mask", "pair_feats"):
        shuffled[key] = obs[key][:, order]
    with torch.no_grad():
        permuted = agent.trunk(shuffled)
    assert torch.allclose(base, permuted, atol=1e-5)


@pytest.mark.parametrize("n_demos", [2, 3, 5])
def test_variable_demonstration_counts_run_unchanged(spec, agent, n_demos):
    """Two demonstrations and five are the same computation with a different mask."""
    env, obs = make_obs(spec, max_demos=n_demos)
    state = agent.initial_state(4, torch.device("cpu"))
    with torch.no_grad():
        out = agent.step(obs, state, torch.zeros(4, 2))
    assert out.state_vec.shape == (4, agent.cfg.state_dim)
    assert torch.isfinite(out.state_vec).all()


def test_padding_slots_do_not_leak(spec, agent):
    """Changing a masked-out slot's contents must not change the state."""
    _, obs = make_obs(spec, max_demos=4)
    empty = (obs["slot_mask"][0] < 0.5).nonzero().flatten()
    assert empty.numel() > 0
    with torch.no_grad():
        base = agent.trunk(obs)
    poisoned = dict(obs)
    poisoned["grids"] = obs["grids"].clone()
    poisoned["grids"][:, empty[0]] = 7
    poisoned["pair_feats"] = obs["pair_feats"].clone()
    poisoned["pair_feats"][:, empty[0]] = 3.0
    with torch.no_grad():
        after = agent.trunk(poisoned)
    assert torch.allclose(base, after, atol=1e-5)


# -- the goal space --------------------------------------------------------


def test_max_cosine_bounds():
    x = torch.randn(16, 8)
    assert torch.allclose(max_cosine(x, x), torch.ones(16), atol=1e-5)
    assert (max_cosine(x, -x) < 0).all()
    # A magnitude mismatch is penalised even when the direction is perfect.
    assert (max_cosine(x, 0.5 * x) < max_cosine(x, x)).all()


def test_goal_autoencoder_round_trip():
    ae = GoalAutoencoder(state_dim=16, n_groups=4, n_codes=4, hidden=32)
    states = torch.randn(32, 16)
    recon, logits = ae(states)
    assert recon.shape == states.shape
    assert logits.shape == (32, 4, 4)

    codes = logits.argmax(-1)
    assert torch.allclose(ae.decode_indices(codes), recon, atol=1e-5)

    novelty = ae.novelty(states)
    assert novelty.shape == (32,) and (novelty >= 0).all()


def test_goal_autoencoder_learns_a_low_rank_state_space():
    """Reconstruction must improve; a goal space that cannot describe the states
    it is built from gives the manager nothing to point at."""
    torch.manual_seed(0)
    ae = GoalAutoencoder(state_dim=8, n_groups=4, n_codes=8, hidden=64, kl_weight=0.01)
    basis = torch.randn(6, 8)
    states = basis[torch.randint(0, 6, (256,))]
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    first = last = None
    for step in range(300):
        loss, stats = ae.loss(states)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = stats["goal_ae/recon"]
        last = stats["goal_ae/recon"]
    assert last < first * 0.5, (first, last)


def test_manager_acts_on_its_own_timescale(spec, agent):
    env, obs = make_obs(spec)
    state = agent.initial_state(4, torch.device("cpu"))
    active = []
    goals = []
    for _ in range(6):
        with torch.no_grad():
            out = agent.step(obs, state, torch.zeros(4, 2))
        active.append(bool(out.manager_active[0]))
        goals.append(out.goal[0].clone())
        next_obs, _, done, _ = env.step(out.op.numpy(), out.args.numpy())
        state = out.next_state
        obs = to_tensors(next_obs, torch.device("cpu"))
    assert active == [True, False, False, True, False, False]
    # The goal is held fixed between manager steps.
    assert torch.allclose(goals[0], goals[1]) and torch.allclose(goals[1], goals[2])
    assert not torch.allclose(goals[2], goals[3])


def test_the_goal_decoder_is_not_trained_by_the_policy(spec, agent):
    """The goal space must be shaped only by the autoencoder's own loss.

    If a policy gradient could reach the decoder, the manager would learn to
    move its goals toward wherever the worker already goes -- the hierarchical
    version of moving the goalposts.
    """
    env, obs = make_obs(spec)
    state = agent.initial_state(4, torch.device("cpu"))
    out = agent.step(obs, state, torch.zeros(4, 2))
    loss = out.worker_logp.sum() + out.manager_logp.sum() + out.manager_value_extr.sum()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    for name, p in agent.goal_ae.decoder.named_parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)), name
    agent.zero_grad(set_to_none=True)
