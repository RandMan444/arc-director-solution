"""The Director training loop.

Two policies, three reward channels, one shared trunk.

    worker    r = w_goal * max_cosine(goal, next state) + w_extr * env reward
    manager   r_extrinsic  = env reward, summed over the goal window
              r_exploration = goal-autoencoder reconstruction error (novelty)

Both are optimised with recurrent advantage actor-critic. A2C rather than PPO
on purpose: the reference implementation this architecture comes from (the A2C
EPN work) is A2C, recurrent PPO needs sequence-aware minibatching that would
double the size of this file, and nothing here is sample-limited -- the
environment runs at over a thousand steps a second. Swapping in PPO later means
replacing :meth:`DirectorTrainer._policy_loss` and keeping everything else.

Credit assignment across the two timescales
-------------------------------------------
The worker sees every step, so it gets ordinary GAE over the episode. Director
instead ends the worker's horizon at each goal change -- a worker cannot be
blamed for what happens after its goal was replaced -- and that is right when
the worker's reward is purely the goal reward. Here it also receives task
reward, and the solve bonus lands on the *last* step of the episode, so
truncating at goal boundaries would hide it from every statement written more
than ``manager_every`` steps earlier. See ``worker_bootstrap_across_goals``.

The manager acts once every ``manager_every`` steps, so its transitions are
``(state at goal k, code, sum of rewards over the window, state at goal k+1)``.
Windows are ragged -- an episode can end mid-window -- so the target is built by
one backward pass that accumulates discounted reward until it reaches the next
goal boundary and bootstraps there.
"""

from __future__ import annotations

import copy
import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dsl.machine import MachineSpec
from ..env.vec import VecProgramEnv
from ..models.agent import AgentState, DirectorAgent
from ..models.goal import max_cosine
from .rollout import RolloutBuffer, to_tensors

__all__ = ["TrainConfig", "DirectorTrainer", "epiplexity_auc"]

_MANAGER_PREFIXES = (
    "manager_lstm.",
    "manager_head.",
    "manager_value_extr.",
    "manager_value_expl.",
)


@dataclass
class TrainConfig:
    """Optimisation, reward mixing and run bookkeeping."""

    # rollout
    n_envs: int = 16
    n_steps: int = 32
    total_steps: int = 2_000_000

    # optimisation
    lr: float = 3e-4
    goal_lr: float = 1e-3
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # worker
    #: ``delta``   r = cos(g, s_next) - cos(g, s)   -- progress toward the goal
    #: ``absolute`` r = cos(g, s_next)              -- Director's own form
    #:
    #: Director uses the absolute form, and in this environment that is a trap.
    #: The goal decoder is trained on states the agent visits, so a manager can
    #: name a goal close to where the worker already is and the worker collects
    #: a large reward every step for changing nothing. Worse, because the reward
    #: is per step, ending the episode *costs* the worker its remaining goal
    #: reward -- so it learns never to emit HALT, which is precisely the action
    #: it needs for "the answer is what I have now". The delta form telescopes
    #: to (final - initial) similarity, so standing still pays nothing and
    #: stopping costs nothing. Measured: with the absolute form the halt rate
    #: collapses to 0.00 and the solve rate sits at the random baseline.
    worker_goal_reward: str = "delta"
    worker_goal_weight: float = 1.0
    worker_extrinsic_weight: float = 1.0
    worker_value_coef: float = 0.5
    worker_entropy_coef: float = 0.02
    worker_entropy_final: float = 0.005
    worker_entropy_anneal_steps: int = 500_000
    #: Whether the worker's return horizon crosses a goal change.
    #:
    #: Director says no: its worker exists only to reach the current goal, and
    #: what happens after the goal is replaced is not its business. That is
    #: correct *when the worker's reward is only the goal reward*. The moment
    #: task reward is mixed in (``worker_extrinsic_weight`` above), truncating
    #: at goal boundaries means the solve bonus -- which lands on the last step
    #: of the episode -- can only be credited to the last ``manager_every``
    #: statements. Everything earlier is invisible to it.
    #:
    #: Measured with truncation on and extrinsic weight 1.0: the solve rate
    #: climbed to 0.60 and then fell back to the random baseline as the halt
    #: rate went to zero, because no early statement was ever credited for the
    #: episode ending well. Set this False only together with
    #: ``worker_extrinsic_weight: 0.0``, which is the faithful Director setting.
    worker_bootstrap_across_goals: bool = True

    # manager
    manager_extrinsic_weight: float = 1.0
    manager_exploration_weight: float = 0.1
    manager_value_coef: float = 0.5
    manager_entropy_coef: float = 0.01

    # goal autoencoder
    goal_ae_coef: float = 1.0

    #: Enforce the one-step Director contract rather than silently allowing a
    #: hybrid hierarchy: the manager acts each step, the worker receives no
    #: environment reward, and worker credit ends when the next goal arrives.
    director_proper: bool = False

    # epiplexity population training
    #: Run two crossed treatments from the same snapshot:
    #:   low-temperature Director + high-temperature worker
    #:   high-temperature Director + low-temperature worker
    #: The two critics duel independently and their winning parameter slices
    #: are recombined.  Each score is the sum of same-target value-loss
    #: reductions over ``epiplexity_duel_updates`` -- the A2C EPN notebook's
    #: per-episode list/AUC semantics.
    epiplexity_duel: bool = False
    epiplexity_duel_updates: int = 4
    epiplexity_low_temperature: float = 0.75
    epiplexity_high_temperature: float = 1.50

    # bookkeeping
    log_every: int = 10
    checkpoint_every: int = 200
    #: Updates between held-out evaluations. 0 disables. Evaluation uses the
    #: full ARC protocol (demonstrations-only selection, two attempts) on the
    #: internal dev split, which no training pool ever contains.
    eval_every: int = 0
    eval_attempts: int = 8
    eval_tasks: int = 40
    seed: int = 0
    device: str = "auto"
    run_dir: str = "runs/dev"

    def __post_init__(self) -> None:
        if self.epiplexity_duel_updates < 1:
            raise ValueError("epiplexity_duel_updates must be >= 1")
        if self.epiplexity_low_temperature <= 0:
            raise ValueError("epiplexity_low_temperature must be positive")
        if self.epiplexity_high_temperature <= self.epiplexity_low_temperature:
            raise ValueError(
                "epiplexity_high_temperature must exceed epiplexity_low_temperature"
            )

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningNorm:
    """Scalar running mean/std, used to keep the two manager rewards comparable."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1)
        n = x.numel()
        if n == 0:
            return
        mean = float(x.mean())
        var = float(x.var(unbiased=False))
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / total
        m_a = self.var * self.count
        m_b = var * n
        self.var = (m_a + m_b + delta**2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var**0.5 + 1e-6)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


def _masked_normalize(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standardise over the entries ``mask`` selects, leaving the rest at zero."""
    count = mask.sum().clamp(min=1.0)
    mean = (x * mask).sum() / count
    var = (((x - mean) ** 2) * mask).sum() / count
    return ((x - mean) / (var.sqrt() + 1e-6)) * mask


def epiplexity_auc(loss_improvements: List[float]) -> float:
    """Sum a duel episode's same-target critic improvements.

    The reference A2C notebook appends ``loss_before - loss_after`` after each
    update chunk and calls the sum of that list the episode's epiplexity/AUC.
    Keeping this tiny operation named and tested prevents a tempting but wrong
    replacement with the final critic loss.
    """
    return float(sum(loss_improvements))


@dataclass
class _Replay:
    state_all: torch.Tensor
    final_state_vec: Optional[torch.Tensor]
    end_state: AgentState
    boot: Optional[Any]
    worker_logp: torch.Tensor
    worker_ent: torch.Tensor
    worker_v: torch.Tensor
    manager_logp: torch.Tensor
    manager_ent: torch.Tensor
    manager_v_extr: torch.Tensor
    manager_v_expl: torch.Tensor
    goals: torch.Tensor


@dataclass
class _TrainerSnapshot:
    agent: Dict[str, torch.Tensor]
    worker_optimizer: Dict[str, Any]
    manager_optimizer: Dict[str, Any]
    goal_optimizer: Dict[str, Any]
    expl_norm: Tuple[float, float, float]
    env: Dict[str, Any]
    obs: Dict[str, torch.Tensor]
    state: AgentState
    feedback: torch.Tensor
    segment_start_state: AgentState
    updates: int
    env_steps: int
    torch_rng: torch.Tensor
    cuda_rng: Optional[List[torch.Tensor]]
    numpy_rng: tuple
    python_rng: object


def _cpu_tree(value: Any) -> Any:
    """Deep-copy nested optimizer/model state, moving tensors off the GPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _copy_agent_state(
    state: AgentState, device: Optional[torch.device] = None
) -> AgentState:
    return AgentState(
        **{
            key: value.detach().to(device or torch.device("cpu")).clone()
            for key, value in vars(state).items()
        }
    )


class DirectorTrainer:
    """Owns the environments, the agent, the optimisers and the run directory."""

    def __init__(
        self,
        spec: MachineSpec,
        env: VecProgramEnv,
        agent: DirectorAgent,
        config: TrainConfig,
        dev_tasks: Optional[List] = None,
    ) -> None:
        self.spec = spec
        self.env = env
        self.dev_tasks = list(dev_tasks or [])
        self.cfg = config
        self.device = config.resolved_device()
        self.agent = agent.to(self.device)

        named_params = dict(self.agent.named_parameters())
        goal_names = {
            name for name in named_params if name.startswith("goal_ae.")
        }
        manager_names = {
            name for name in named_params if name.startswith(_MANAGER_PREFIXES)
        }
        self._worker_param_names = tuple(
            name for name in named_params if name not in goal_names | manager_names
        )
        self._manager_param_names = tuple(
            name for name in named_params if name in manager_names
        )
        self._goal_param_names = tuple(name for name in named_params if name in goal_names)
        self._worker_params = [named_params[name] for name in self._worker_param_names]
        self._manager_params = [named_params[name] for name in self._manager_param_names]
        goal_params = [named_params[name] for name in self._goal_param_names]
        if config.epiplexity_duel and self.agent.cfg.manager_grads_to_trunk:
            raise ValueError(
                "epiplexity_duel requires agent.manager_grads_to_trunk=false so "
                "Director and worker winners can be recombined without shared gradients"
            )
        if config.director_proper:
            violations = []
            if self.agent.cfg.manager_every != 1:
                violations.append("agent.manager_every must be 1")
            if self.agent.cfg.worker_env_feedback:
                violations.append("agent.worker_env_feedback must be false")
            if config.worker_extrinsic_weight != 0.0:
                violations.append("train.worker_extrinsic_weight must be 0")
            if config.worker_goal_weight <= 0.0:
                violations.append("train.worker_goal_weight must be positive")
            if config.worker_bootstrap_across_goals:
                violations.append("train.worker_bootstrap_across_goals must be false")
            if violations:
                raise ValueError(
                    "director_proper contract violated: " + "; ".join(violations)
                )
        self.worker_optimizer = torch.optim.Adam(
            self._worker_params, lr=config.lr, eps=1e-5
        )
        self.manager_optimizer = torch.optim.Adam(
            self._manager_params, lr=config.lr, eps=1e-5
        )
        self.goal_optimizer = torch.optim.Adam(goal_params, lr=config.goal_lr, eps=1e-5)

        self.expl_norm = RunningNorm()
        self.updates = 0
        self.env_steps = 0
        self.duel_rounds = 0
        self.worker_explore_wins = 0
        self.director_explore_wins = 0
        self.start_time = time.time()

        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"
        # Bounded: a long run finishes millions of episodes and each info dict
        # carries a program string.
        self.episode_stats: deque = deque(maxlen=2000)

        obs = self.env.reset()
        self._obs = to_tensors(obs, self.device)
        self._state = self.agent.initial_state(self.env.n_envs, self.device)
        self._feedback = torch.zeros(self.env.n_envs, 2, device=self.device)
        # The recurrent state the update pass replays from. It is the state the
        # last segment ended on, which is the state collection is about to
        # continue from -- keeping the two in step is what makes the replayed
        # log-probabilities match the ones that were actually acted on.
        self._segment_start_state = self.agent.initial_state(self.env.n_envs, self.device)

    # -- collection ------------------------------------------------------
    @torch.no_grad()
    def collect(
        self,
        *,
        worker_temperature: float = 1.0,
        manager_temperature: float = 1.0,
    ) -> RolloutBuffer:
        buffer = RolloutBuffer(
            device=self.device,
            worker_temperature=float(worker_temperature),
            manager_temperature=float(manager_temperature),
        )
        for _ in range(self.cfg.n_steps):
            out = self.agent.step(
                self._obs,
                self._state,
                self._feedback,
                worker_temperature=worker_temperature,
                manager_temperature=manager_temperature,
            )
            ops = out.op.cpu().numpy()
            args = out.args.cpu().numpy()

            next_obs, reward, done, infos = self.env.step(ops, args)
            buffer.add(
                obs=self._obs,
                op=out.op,
                args=out.args,
                codes=out.manager_codes,
                active=out.manager_active,
                feedback=self._feedback,
                reward=reward,
                done=done,
                info=infos,
                behaviour_logp=out.worker_logp,
            )

            done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device)
            error_flag = torch.as_tensor(
                np.array([float(i.get("error_code") is not None) for i in infos]),
                dtype=torch.float32,
                device=self.device,
            )
            self._feedback = torch.stack(
                [torch.as_tensor(reward, dtype=torch.float32, device=self.device), error_flag],
                dim=-1,
            ) * (1.0 - done_t).unsqueeze(-1)
            self._state = out.next_state.reset_where(done_t)
            self._obs = to_tensors(next_obs, self.device)
            self.env_steps += self.env.n_envs

        buffer.final_obs = self._obs
        buffer.final_feedback = self._feedback
        return buffer

    # -- rewards ---------------------------------------------------------
    def goal_reward(
        self, goals: torch.Tensor, states: torch.Tensor, next_states: torch.Tensor
    ) -> torch.Tensor:
        """The worker's goal-reaching reward. See ``TrainConfig.worker_goal_reward``."""
        reached = max_cosine(goals, next_states)
        mode = self.cfg.worker_goal_reward
        if mode == "delta":
            return reached - max_cosine(goals, states)
        if mode == "absolute":
            return reached
        raise ValueError(
            f"worker_goal_reward must be 'delta' or 'absolute', got {mode!r}"
        )

    # -- returns ---------------------------------------------------------
    def _gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard GAE. ``dones[t]`` cuts the bootstrap after step ``t``."""
        t_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_value)
        for t in reversed(range(t_steps)):
            next_value = last_value if t == t_steps - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * not_done - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * not_done * gae
            advantages[t] = gae
        return advantages, advantages + values

    def _manager_targets(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        active: torch.Tensor,
        last_value: torch.Tensor,
    ) -> torch.Tensor:
        """Discounted reward to the end of the goal window, then bootstrap.

        Walking backwards, ``carry`` holds the value of continuing from the
        next step. It is reset to zero at an episode end, and replaced by the
        manager's own value estimate wherever a new goal starts -- which is
        exactly where the current manager decision stops being responsible.
        """
        t_steps = rewards.shape[0]
        targets = torch.zeros_like(rewards)
        carry = last_value
        for t in reversed(range(t_steps)):
            if t == t_steps - 1:
                cont = last_value
            else:
                cont = torch.where(active[t + 1] > 0.5, values[t + 1], carry)
            cont = cont * (1.0 - dones[t])
            carry = rewards[t] + self.cfg.gamma * cont
            targets[t] = carry
        return targets

    def _replay_segment(
        self,
        buffer: RolloutBuffer,
        start_state: AgentState,
        *,
        bootstrap: bool,
    ) -> _Replay:
        """Replay stored actions, optionally producing boundary bootstrap values."""
        t_steps = len(buffer)
        n_envs = self.env.n_envs
        state_all = self.agent.trunk(buffer.stacked_obs()).view(t_steps, n_envs, -1)

        ops = buffer.tensor("ops")
        args = buffer.tensor("args")
        codes = buffer.tensor("manager_codes")
        feedback = buffer.tensor("feedback")
        dones = buffer.done_tensor()

        state = start_state
        worker_logp, worker_ent, worker_v = [], [], []
        manager_logp, manager_ent, manager_v_extr, manager_v_expl = [], [], [], []
        goals = []
        for t in range(t_steps):
            out = self.agent.step(
                buffer.obs[t],
                state,
                feedback[t],
                actions=(ops[t], args[t]),
                manager_action=codes[t],
                state_vec=state_all[t],
                worker_temperature=buffer.worker_temperature,
                manager_temperature=buffer.manager_temperature,
            )
            worker_logp.append(out.worker_logp)
            worker_ent.append(out.worker_entropy)
            worker_v.append(out.worker_value)
            manager_logp.append(out.manager_logp)
            manager_ent.append(out.manager_entropy)
            manager_v_extr.append(out.manager_value_extr)
            manager_v_expl.append(out.manager_value_expl)
            goals.append(out.goal)
            state = out.next_state.reset_where(dones[t])

        final_state_vec = None
        boot = None
        if bootstrap:
            # Boundary values are targets, never part of the differentiated
            # replay.  They are computed exactly once and then held fixed for
            # both sides of the epiplexity measurement.
            with torch.no_grad():
                final_state_vec = self.agent.trunk(buffer.final_obs)
                boot = self.agent.step(
                    buffer.final_obs,
                    state,
                    buffer.final_feedback,
                    state_vec=final_state_vec,
                    worker_temperature=buffer.worker_temperature,
                    manager_temperature=buffer.manager_temperature,
                )

        return _Replay(
            state_all=state_all,
            final_state_vec=final_state_vec,
            end_state=state,
            boot=boot,
            worker_logp=torch.stack(worker_logp),
            worker_ent=torch.stack(worker_ent),
            worker_v=torch.stack(worker_v),
            manager_logp=torch.stack(manager_logp),
            manager_ent=torch.stack(manager_ent),
            manager_v_extr=torch.stack(manager_v_extr),
            manager_v_expl=torch.stack(manager_v_expl),
            goals=torch.stack(goals),
        )

    # -- update ----------------------------------------------------------
    def update(
        self, buffer: RolloutBuffer, *, measure_epiplexity: bool = False
    ) -> Dict[str, float]:
        cfg = self.cfg
        t_steps = len(buffer)
        n_envs = self.env.n_envs

        segment_start = self._segment_start_state
        replay = self._replay_segment(buffer, segment_start, bootstrap=True)
        state_all = replay.state_all
        final_state_vec = replay.final_state_vec
        boot = replay.boot
        assert final_state_vec is not None and boot is not None

        ops = buffer.tensor("ops")
        args = buffer.tensor("args")
        codes = buffer.tensor("manager_codes")
        feedback = buffer.tensor("feedback")
        active = buffer.tensor("manager_active").float()
        rewards = buffer.reward_tensor()
        dones = buffer.done_tensor()

        self._segment_start_state = replay.end_state.detach()

        worker_logp = replay.worker_logp
        worker_ent = replay.worker_ent
        worker_v = replay.worker_v
        manager_logp = replay.manager_logp
        manager_ent = replay.manager_ent
        manager_v_extr = replay.manager_v_extr
        manager_v_expl = replay.manager_v_expl
        goals = replay.goals

        # -- rewards ----------------------------------------------------
        with torch.no_grad():
            detached = state_all.detach()
            next_state_vec = torch.cat(
                [detached[1:], final_state_vec.unsqueeze(0)], dim=0
            )
            # At an episode boundary the "next" state belongs to a different
            # task, so the goal is scored against the state actually reached.
            keep = (1.0 - dones).unsqueeze(-1)
            next_state_vec = keep * next_state_vec + (1.0 - keep) * detached

            goal_reward = self.goal_reward(goals.detach(), detached, next_state_vec)
            worker_reward = (
                cfg.worker_goal_weight * goal_reward + cfg.worker_extrinsic_weight * rewards
            )

            novelty = self.agent.goal_ae.novelty(detached.reshape(-1, detached.shape[-1]))
            novelty = novelty.view(t_steps, n_envs)
            self.expl_norm.update(novelty)
            expl_reward = self.expl_norm.normalize(novelty)

        # -- worker advantage ------------------------------------------
        with torch.no_grad():
            worker_done = dones
            if not cfg.worker_bootstrap_across_goals:
                # A goal change ends the worker's horizon just as an episode end does.
                next_active = torch.cat(
                    [active[1:], boot.manager_active.float().unsqueeze(0)], dim=0
                )
                worker_done = torch.clamp(dones + next_active, max=1.0)
            w_adv, w_ret = self._gae(
                worker_reward, worker_v.detach(), worker_done, boot.worker_value
            )

        # -- manager advantage -----------------------------------------
        with torch.no_grad():
            extr_target = self._manager_targets(
                rewards, dones, manager_v_extr.detach(), active, boot.manager_value_extr
            )
            expl_target = self._manager_targets(
                expl_reward, dones, manager_v_expl.detach(), active, boot.manager_value_expl
            )
            m_adv = (
                cfg.manager_extrinsic_weight * (extr_target - manager_v_extr.detach())
                + cfg.manager_exploration_weight * (expl_target - manager_v_expl.detach())
            )

        # -- losses ------------------------------------------------------
        entropy_coef = self._entropy_coef()
        w_adv_n = _normalize(w_adv)
        worker_policy = -(worker_logp * w_adv_n).mean()
        # Huber, not MSE. The moment the agent starts earning the solve bonus,
        # returns jump by an order of magnitude and a squared error sends a
        # gradient through the *shared trunk* large enough to destroy the
        # representation the policy depends on. Measured on the warm-up ladder:
        # with MSE the solve rate climbed to 0.63 and then collapsed to the
        # random baseline within 10k steps, worker entropy falling from 2.25 to
        # 0.78 as the policy locked onto a single operator. Huber bounds the
        # per-element gradient and the collapse does not happen.
        worker_value_loss = F.smooth_l1_loss(worker_v, w_ret)
        worker_entropy = worker_ent.mean()

        mask = active
        denom = mask.sum().clamp(min=1.0)
        # Only the steps the manager actually acted on carry a meaningful
        # advantage, so the normalisation is over those alone.
        m_adv_n = _masked_normalize(m_adv, mask)
        manager_policy = -((manager_logp * m_adv_n) * mask).sum() / denom
        manager_value_loss = (
            (F.smooth_l1_loss(manager_v_extr, extr_target, reduction="none") * mask).sum()
            / denom
            + (F.smooth_l1_loss(manager_v_expl, expl_target, reduction="none") * mask).sum()
            / denom
        )
        manager_entropy = (manager_ent * mask).sum() / denom

        worker_loss = (
            worker_policy
            + cfg.worker_value_coef * worker_value_loss
            - entropy_coef * worker_entropy
        )
        manager_loss = (
            manager_policy
            + cfg.manager_value_coef * manager_value_loss
            - cfg.manager_entropy_coef * manager_entropy
        )
        loss = worker_loss + manager_loss

        self.worker_optimizer.zero_grad(set_to_none=True)
        self.manager_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        policy_params = self._worker_params + self._manager_params
        grad_norm = nn.utils.clip_grad_norm_(
            policy_params, cfg.max_grad_norm,
        )

        # The two optimizers have disjoint ownership.  Step and measure the
        # Director first while the worker-owned trunk is unchanged; otherwise
        # a trunk update would masquerade as Director critic learning.  Stored
        # goal codes make the worker replay independent of the changed manager.
        self.manager_optimizer.step()
        manager_epi_delta = 0.0
        if measure_epiplexity:
            with torch.no_grad():
                after_manager = self._replay_segment(
                    buffer, segment_start, bootstrap=False
                )
                manager_after_loss = (
                    (
                        F.smooth_l1_loss(
                            after_manager.manager_v_extr,
                            extr_target,
                            reduction="none",
                        )
                        * mask
                    ).sum()
                    / denom
                    + (
                        F.smooth_l1_loss(
                            after_manager.manager_v_expl,
                            expl_target,
                            reduction="none",
                        )
                        * mask
                    ).sum()
                    / denom
                )
            manager_epi_delta = float(manager_value_loss.detach() - manager_after_loss)

        self.worker_optimizer.step()
        worker_epi_delta = 0.0
        if measure_epiplexity:
            with torch.no_grad():
                after_worker = self._replay_segment(
                    buffer, segment_start, bootstrap=False
                )
                worker_after_loss = F.smooth_l1_loss(
                    after_worker.worker_v, w_ret
                )
            worker_epi_delta = float(worker_value_loss.detach() - worker_after_loss)

        # -- goal autoencoder, on detached states ----------------------
        ae_loss, ae_stats = self.agent.goal_ae.loss(state_all.detach().reshape(-1, state_all.shape[-1]))
        self.goal_optimizer.zero_grad(set_to_none=True)
        (cfg.goal_ae_coef * ae_loss).backward()
        nn.utils.clip_grad_norm_(self.agent.goal_ae.parameters(), cfg.max_grad_norm)
        self.goal_optimizer.step()

        self.updates += 1
        if buffer.behaviour_logp:
            # Cheap tripwire on the single most breakable invariant in a
            # recurrent on-policy trainer: the replay must reproduce the
            # distribution that was actually acted on.
            drift = float(
                (torch.stack(buffer.behaviour_logp) - worker_logp.detach()).abs().max()
            )
            self.last_replay_drift = drift
        stats = {
            "loss/total": float(loss.detach()),
            "loss/worker_policy": float(worker_policy.detach()),
            "loss/worker_value": float(worker_value_loss.detach()),
            "loss/manager_policy": float(manager_policy.detach()),
            "loss/manager_value": float(manager_value_loss.detach()),
            "policy/worker_entropy": float(worker_entropy.detach()),
            "policy/manager_entropy": float(manager_entropy.detach()),
            "policy/entropy_coef": entropy_coef,
            "policy/grad_norm": float(grad_norm),
            "reward/env_mean": float(rewards.mean()),
            "reward/goal_mean": float(goal_reward.mean()),
            "reward/worker_mean": float(worker_reward.mean()),
            "reward/novelty_mean": float(novelty.mean()),
            "value/worker": float(worker_v.mean().detach()),
            "value/manager_extr": float(manager_v_extr.mean().detach()),
            "hierarchy/director_every": int(self.agent.cfg.manager_every),
            "hierarchy/worker_task_weight": float(cfg.worker_extrinsic_weight),
            "hierarchy/director_proper": bool(cfg.director_proper),
            "epiplexity/worker_delta": worker_epi_delta,
            "epiplexity/director_delta": manager_epi_delta,
        }
        stats.update(ae_stats)
        stats.update(self._action_stats(ops))
        return stats

    # -- crossed epiplexity duel ----------------------------------------
    def _snapshot(self) -> _TrainerSnapshot:
        return _TrainerSnapshot(
            agent=_cpu_tree(self.agent.state_dict()),
            worker_optimizer=_cpu_tree(self.worker_optimizer.state_dict()),
            manager_optimizer=_cpu_tree(self.manager_optimizer.state_dict()),
            goal_optimizer=_cpu_tree(self.goal_optimizer.state_dict()),
            expl_norm=(self.expl_norm.mean, self.expl_norm.var, self.expl_norm.count),
            env=self.env.state_dict(),
            obs={key: value.detach().cpu().clone() for key, value in self._obs.items()},
            state=_copy_agent_state(self._state),
            feedback=self._feedback.detach().cpu().clone(),
            segment_start_state=_copy_agent_state(self._segment_start_state),
            updates=self.updates,
            env_steps=self.env_steps,
            torch_rng=torch.get_rng_state().clone(),
            cuda_rng=(
                [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            numpy_rng=copy.deepcopy(np.random.get_state()),
            python_rng=random.getstate(),
        )

    def _restore(self, snapshot: _TrainerSnapshot) -> None:
        self.agent.load_state_dict(snapshot.agent)
        self.worker_optimizer.load_state_dict(snapshot.worker_optimizer)
        self.manager_optimizer.load_state_dict(snapshot.manager_optimizer)
        self.goal_optimizer.load_state_dict(snapshot.goal_optimizer)
        self.expl_norm.mean, self.expl_norm.var, self.expl_norm.count = snapshot.expl_norm
        self.env.load_state_dict(snapshot.env)
        self._obs = {
            key: value.to(self.device) for key, value in snapshot.obs.items()
        }
        self._state = _copy_agent_state(snapshot.state, self.device)
        self._feedback = snapshot.feedback.to(self.device)
        self._segment_start_state = _copy_agent_state(
            snapshot.segment_start_state, self.device
        )
        self.updates = snapshot.updates
        self.env_steps = snapshot.env_steps
        torch.set_rng_state(snapshot.torch_rng)
        if snapshot.cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(snapshot.cuda_rng)
        np.random.set_state(snapshot.numpy_rng)
        random.setstate(snapshot.python_rng)

    def _reset_after_duel(self) -> None:
        """Begin a clean episode after independently recombining two winners."""
        obs = self.env.reset()
        self._obs = to_tensors(obs, self.device)
        self._state = self.agent.initial_state(self.env.n_envs, self.device)
        self._feedback = torch.zeros(self.env.n_envs, 2, device=self.device)
        self._segment_start_state = self.agent.initial_state(
            self.env.n_envs, self.device
        )

    def duel_update(self) -> Tuple[Dict[str, float], List[dict]]:
        """Run the crossed candidates and install independent component winners.

        Candidate A pairs an exploitative Director with an exploratory worker;
        candidate B reverses that pairing.  Every candidate starts with the
        same parameters, optimizer moments, live environments and RNG streams.
        Critic loss is measured before and after each update against the exact
        same fixed return targets.  Those deltas form a list and their sum is
        the component's duel-episode AUC.
        """
        cfg = self.cfg
        base = self._snapshot()
        treatments = {
            "worker_explore": (
                cfg.epiplexity_high_temperature,
                cfg.epiplexity_low_temperature,
            ),
            "director_explore": (
                cfg.epiplexity_low_temperature,
                cfg.epiplexity_high_temperature,
            ),
        }
        candidates: Dict[str, Dict[str, Any]] = {}

        try:
            for name, (worker_temperature, manager_temperature) in treatments.items():
                self._restore(base)
                worker_curve: List[float] = []
                director_curve: List[float] = []
                episodes: List[dict] = []
                stats: Dict[str, Any] = {}
                for _ in range(cfg.epiplexity_duel_updates):
                    buffer = self.collect(
                        worker_temperature=worker_temperature,
                        manager_temperature=manager_temperature,
                    )
                    stats = self.update(buffer, measure_epiplexity=True)
                    worker_curve.append(stats["epiplexity/worker_delta"])
                    director_curve.append(stats["epiplexity/director_delta"])
                    episodes.extend(buffer.finished_infos())
                candidates[name] = {
                    "snapshot": self._snapshot(),
                    "stats": stats,
                    "episodes": episodes,
                    "worker_auc": epiplexity_auc(worker_curve),
                    "director_auc": epiplexity_auc(director_curve),
                }
        except BaseException:
            # scripts/train.py saves in its finally block. Never let Ctrl-C or
            # a failed candidate turn that durable checkpoint into half a duel.
            self._restore(base)
            raise

        # Ties deliberately go to the lower-temperature treatment.  A duel
        # should demand positive evidence before paying for more randomness.
        worker_winner = (
            "worker_explore"
            if candidates["worker_explore"]["worker_auc"]
            > candidates["director_explore"]["worker_auc"]
            else "director_explore"
        )
        director_winner = (
            "director_explore"
            if candidates["director_explore"]["director_auc"]
            > candidates["worker_explore"]["director_auc"]
            else "worker_explore"
        )

        worker_snapshot: _TrainerSnapshot = candidates[worker_winner]["snapshot"]
        director_snapshot: _TrainerSnapshot = candidates[director_winner]["snapshot"]
        self._restore(worker_snapshot)
        named_params = dict(self.agent.named_parameters())
        with torch.no_grad():
            for name in self._manager_param_names:
                named_params[name].copy_(director_snapshot.agent[name].to(self.device))
        self.manager_optimizer.load_state_dict(director_snapshot.manager_optimizer)

        # Both forks performed real environment work even though only one live
        # environment history can continue.  Count both for budgets/SPS; the
        # worker winner owns the continuing curriculum and goal/trunk state.
        self.updates = base.updates + sum(
            candidate["snapshot"].updates - base.updates
            for candidate in candidates.values()
        )
        self.env_steps = base.env_steps + sum(
            candidate["snapshot"].env_steps - base.env_steps
            for candidate in candidates.values()
        )
        self.duel_rounds += 1
        self.worker_explore_wins += int(worker_winner == "worker_explore")
        self.director_explore_wins += int(director_winner == "director_explore")
        self._reset_after_duel()

        stats = dict(candidates[worker_winner]["stats"])
        stats.update(
            {
                "epiplexity/worker_explore_auc": candidates["worker_explore"][
                    "worker_auc"
                ],
                "epiplexity/worker_exploit_auc": candidates["director_explore"][
                    "worker_auc"
                ],
                "epiplexity/director_explore_auc": candidates["director_explore"][
                    "director_auc"
                ],
                "epiplexity/director_exploit_auc": candidates["worker_explore"][
                    "director_auc"
                ],
                "epiplexity/worker_winner": (
                    "explore" if worker_winner == "worker_explore" else "exploit"
                ),
                "epiplexity/director_winner": (
                    "explore" if director_winner == "director_explore" else "exploit"
                ),
                "epiplexity/worker_explore_win_rate": (
                    self.worker_explore_wins / self.duel_rounds
                ),
                "epiplexity/director_explore_win_rate": (
                    self.director_explore_wins / self.duel_rounds
                ),
                "epiplexity/duel_round": self.duel_rounds,
            }
        )
        return stats, list(candidates[worker_winner]["episodes"])

    def _action_stats(self, ops: torch.Tensor) -> Dict[str, float]:
        """How concentrated the operator choice is, and on what.

        A hierarchy that has quietly collapsed onto one operator looks fine in
        the loss but shows up here immediately, so it is logged every update.
        """
        counts = torch.bincount(ops.reshape(-1), minlength=self.spec.n_ops).float()
        probs = counts / counts.sum().clamp(min=1.0)
        nonzero = probs[probs > 0]
        top = int(probs.argmax())
        return {
            "actions/op_entropy": float(-(nonzero * nonzero.log()).sum()),
            "actions/distinct_ops": int((counts > 0).sum()),
            "actions/top_op": self.spec.op_names[top],
            "actions/top_op_share": round(float(probs[top]), 4),
        }

    def _entropy_coef(self) -> float:
        cfg = self.cfg
        if cfg.worker_entropy_anneal_steps <= 0:
            return cfg.worker_entropy_coef
        frac = min(1.0, self.env_steps / cfg.worker_entropy_anneal_steps)
        return cfg.worker_entropy_coef + frac * (
            cfg.worker_entropy_final - cfg.worker_entropy_coef
        )

    # -- loop ------------------------------------------------------------
    def train(self, total_steps: Optional[int] = None) -> None:
        total = total_steps or self.cfg.total_steps
        self._segment_start_state = self.agent.initial_state(self.env.n_envs, self.device)
        while self.env_steps < total:
            before_update = self.updates
            if self.cfg.epiplexity_duel:
                stats, episodes = self.duel_update()
            else:
                buffer = self.collect()
                stats = self.update(buffer)
                episodes = buffer.finished_infos()
            self.episode_stats.extend(episodes)
            if self._crossed_interval(before_update, self.updates, self.cfg.log_every):
                self.log(stats, episodes)
            if self._crossed_interval(
                before_update, self.updates, self.cfg.checkpoint_every
            ):
                self.save(self.run_dir / "checkpoint.pt")
            if (
                self._crossed_interval(before_update, self.updates, self.cfg.eval_every)
                and self.dev_tasks
            ):
                self.evaluate()

    @staticmethod
    def _crossed_interval(before: int, after: int, interval: int) -> bool:
        return bool(interval and after // interval > before // interval)

    def evaluate(self) -> Dict[str, float]:
        """Run the ARC protocol on the internal dev split and log the result.

        These tasks are excluded from every training pool, so this is the only
        number in the run that is not measured on data the agent trains on.
        """
        from .evaluate import evaluate_tasks

        tasks = self.dev_tasks[: self.cfg.eval_tasks]
        summary, _ = evaluate_tasks(
            self.agent,
            self.spec,
            tasks,
            self.env.cfg,
            self.device,
            n_attempts=self.cfg.eval_attempts,
            seed=self.updates,
        )
        row = {f"eval/{k}": v for k, v in summary.items()}
        row.update(update=self.updates, env_steps=self.env_steps, event="eval")
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  [eval] {len(tasks)} dev tasks: demo_fit={summary['demo_fit_rate']:.3f} "
            f"exact@2={summary['exact_at_2']:.3f} pass@{self.cfg.eval_attempts}="
            f"{summary['pass_at_n']:.3f}",
            flush=True,
        )
        return summary

    def log(self, stats: Dict[str, float], episodes: List[dict]) -> None:
        recent = list(self.episode_stats)[-200:]
        row = dict(stats)
        row.update(
            update=self.updates,
            env_steps=self.env_steps,
            sps=round(self.env_steps / max(1e-6, time.time() - self.start_time), 1),
            episodes=len(self.episode_stats),
        )
        if recent:
            row.update(
                solve_rate=round(float(np.mean([e.get("solved_demos", False) for e in recent])), 4),
                generalize_rate=round(float(np.mean([e.get("generalized", False) for e in recent])), 4),
                test_rate=round(float(np.mean([e.get("solved_test", False) for e in recent])), 4),
                mean_return=round(float(np.mean([e.get("episode_return", 0.0) for e in recent])), 3),
                mean_statements=round(float(np.mean([e.get("n_statements", 0) for e in recent])), 2),
                halt_rate=round(float(np.mean([e.get("halted", False) for e in recent])), 3),
            )
        row.update({f"env/{k}": v for k, v in self.env.stats().items()})
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"[{row['update']:6d}] steps={row['env_steps']:>9,} "
            f"stage={row.get('env/stage', '-')} "
            f"solve={row.get('solve_rate', 0):.3f} gen={row.get('generalize_rate', 0):.3f} "
            f"ret={row.get('mean_return', 0):+.2f} "
            f"goal_r={stats['reward/goal_mean']:+.3f} "
            f"H_w={stats['policy/worker_entropy']:.2f} "
            f"top={stats['actions/top_op']}:{stats['actions/top_op_share']:.2f} "
            f"sps={row['sps']:.0f}",
            flush=True,
        )

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        source = self.env.source
        torch.save(
            {
                "agent": self.agent.state_dict(),
                "worker_optimizer": self.worker_optimizer.state_dict(),
                "manager_optimizer": self.manager_optimizer.state_dict(),
                "goal_optimizer": self.goal_optimizer.state_dict(),
                "updates": self.updates,
                "env_steps": self.env_steps,
                "duel_rounds": self.duel_rounds,
                "worker_explore_wins": self.worker_explore_wins,
                "director_explore_wins": self.director_explore_wins,
                "expl_norm": {
                    "mean": self.expl_norm.mean,
                    "var": self.expl_norm.var,
                    "count": self.expl_norm.count,
                },
                "agent_config": asdict(self.agent.cfg),
                "train_config": asdict(self.cfg),
                "curriculum": source.state_dict() if hasattr(source, "state_dict") else None,
            },
            path,
        )

    def _load_legacy_optimizer(self, state: Dict[str, Any]) -> None:
        """Split a pre-duel single Adam checkpoint without losing its moments."""
        goal_ids = {id(parameter) for parameter in self.agent.goal_ae.parameters()}
        legacy_params = [
            parameter
            for parameter in self.agent.parameters()
            if id(parameter) not in goal_ids
        ]
        legacy = torch.optim.Adam(legacy_params, lr=self.cfg.lr, eps=1e-5)
        legacy.load_state_dict(state)

        for target, parameters in (
            (self.worker_optimizer, self._worker_params),
            (self.manager_optimizer, self._manager_params),
        ):
            target.state.clear()
            for parameter in parameters:
                if parameter in legacy.state:
                    target.state[parameter] = copy.deepcopy(legacy.state[parameter])
            source_group = legacy.param_groups[0]
            target_group = target.param_groups[0]
            for key, value in source_group.items():
                if key != "params":
                    target_group[key] = copy.deepcopy(value)

    def load(self, path: Path) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        saved = blob["agent"]
        current = self.agent.state_dict()
        mismatched = [
            f"  {k}: checkpoint {tuple(v.shape)} vs config {tuple(current[k].shape)}"
            for k, v in saved.items()
            if k in current and v.shape != current[k].shape
        ]
        if mismatched:
            raise RuntimeError(
                f"{path} was trained with a different network shape:\n"
                + "\n".join(mismatched[:8])
                + "\n\nOnly `grid_side` may differ between a warm-up config and the "
                "config resuming from it -- it changes no parameter shape. Every other "
                "size (grid_channels, d_model, state_dim, lstm_dim, mem_len, goal_*) "
                "must match."
            )
        self.agent.load_state_dict(saved)
        if "worker_optimizer" in blob and "manager_optimizer" in blob:
            self.worker_optimizer.load_state_dict(blob["worker_optimizer"])
            self.manager_optimizer.load_state_dict(blob["manager_optimizer"])
        elif "optimizer" in blob:
            self._load_legacy_optimizer(blob["optimizer"])
        else:
            raise KeyError(f"{path} has no policy optimizer state")
        self.goal_optimizer.load_state_dict(blob["goal_optimizer"])
        self.updates = int(blob.get("updates", 0))
        self.env_steps = int(blob.get("env_steps", 0))
        self.duel_rounds = int(blob.get("duel_rounds", 0))
        self.worker_explore_wins = int(blob.get("worker_explore_wins", 0))
        self.director_explore_wins = int(blob.get("director_explore_wins", 0))
        norm = blob.get("expl_norm") or {}
        self.expl_norm.mean = float(norm.get("mean", self.expl_norm.mean))
        self.expl_norm.var = float(norm.get("var", self.expl_norm.var))
        self.expl_norm.count = float(norm.get("count", self.expl_norm.count))
        source = self.env.source
        if blob.get("curriculum") and hasattr(source, "load_state_dict"):
            source.load_state_dict(blob["curriculum"])
