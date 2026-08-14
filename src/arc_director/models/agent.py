"""The Director agent: shared trunk, goal-setting manager, statement-writing worker.

Layout, and why
---------------
::

    obs ─┬─ GridEncoder ─ PairEncoder ─ SetAttention ─┐
         └─ register / scalar features ───────────────┴─ state s_t   (shared trunk)
                                                          │
                    ┌─────────────────────────────────────┴──────────────┐
                    │                                                    │
             Manager LSTM                                         Worker LSTM
        (every K steps, at the                          (every step, fed s_t, the goal,
         abstract time scale)                            the previous action, and the
                    │                                    EPN episodic memory)
                    │                                                    │
            8 x 8 goal code                                   op head → arg heads
                    │                                          (masked, autoregressive)
            GoalAutoencoder.decode
                    │
                 goal g ──────────────────────────────────────────────► worker input

The trunk is shared. In Director both policies read the same world-model state,
and the analogue here is that both read the same demonstration-set encoding --
there is no reason for the manager and the worker to learn "what does this task
look like" twice, and a shared trunk is what makes the manager's goals mean the
same thing the worker's rewards are measured in. What is *not* shared is
recurrence: the manager runs at the abstract time scale, so its LSTM advances
once per goal, while the worker's advances once per statement.

The worker gets the episodic-memory attention and the manager does not. The
worker is the one that has to avoid rewriting the statement it just wrote; the
manager sees a coarser picture by construction.

By default the manager's gradients do not reach the trunk (``manager_grads_to_
trunk``). Two objectives pulling on one representation is the classic way to
get a hierarchy that trains neither half well; the worker's is the denser
signal, so it gets the representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..dsl.machine import MachineSpec
from ..env.task_env import PAIR_FEATURE_DIM, SCALAR_DIM
from .encoders import GridEncoder, PairEncoder
from .epn import EpisodicMemory, SetAttention, write_memory
from .goal import GoalAutoencoder

__all__ = ["AgentConfig", "AgentState", "StepOutput", "DirectorAgent"]

_NEG = -1e9


@dataclass
class AgentConfig:
    """Sizes and switches. The defaults are the CPU-runnable "small" preset."""

    grid_side: int = 30
    grid_channels: Tuple[int, int, int] = (24, 48, 48)
    grid_dim: int = 64
    d_model: int = 128
    set_heads: int = 4
    set_layers: int = 2
    state_dim: int = 128
    lstm_dim: int = 256
    action_emb: int = 32

    use_memory: bool = True
    mem_len: int = 24
    mem_dim: int = 64
    mem_heads: int = 4

    goal_groups: int = 8
    goal_codes: int = 8
    goal_hidden: int = 256
    goal_kl_weight: float = 0.1

    manager_every: int = 4
    manager_grads_to_trunk: bool = False


@dataclass
class AgentState:
    """Everything carried between steps. All tensors are ``(B, ...)``.

    The previous *action* is not here: the environment already reports it in
    ``obs["last_action"]``, and having one source of truth for it removes a
    whole class of desynchronisation bug between acting and the update pass.
    The previous reward is passed in per step as ``feedback``.
    """

    worker_h: torch.Tensor
    worker_c: torch.Tensor
    manager_h: torch.Tensor
    manager_c: torch.Tensor
    memory: torch.Tensor
    memory_mask: torch.Tensor
    goal: torch.Tensor
    goal_codes: torch.Tensor
    since_goal: torch.Tensor
    step_index: torch.Tensor

    def detach(self) -> "AgentState":
        return AgentState(
            **{k: v.detach() for k, v in self.__dict__.items()}
        )

    def reset_where(self, done: torch.Tensor) -> "AgentState":
        """Zero the per-episode carry for finished environments."""
        keep = (1.0 - done).unsqueeze(-1)
        keep_i = (1 - done.long())
        return AgentState(
            worker_h=self.worker_h * keep,
            worker_c=self.worker_c * keep,
            manager_h=self.manager_h * keep,
            manager_c=self.manager_c * keep,
            memory=self.memory * keep.unsqueeze(-1),
            memory_mask=self.memory_mask * keep,
            goal=self.goal * keep,
            goal_codes=self.goal_codes * keep_i.unsqueeze(-1),
            since_goal=self.since_goal * keep_i,
            step_index=self.step_index * keep_i,
        )


@dataclass
class StepOutput:
    """One step's outputs, for acting and for the update pass alike."""

    op: torch.Tensor
    args: torch.Tensor
    worker_logp: torch.Tensor
    worker_entropy: torch.Tensor
    worker_value: torch.Tensor
    manager_active: torch.Tensor
    manager_codes: torch.Tensor
    manager_logp: torch.Tensor
    manager_entropy: torch.Tensor
    manager_value_extr: torch.Tensor
    manager_value_expl: torch.Tensor
    state_vec: torch.Tensor
    goal: torch.Tensor
    next_state: AgentState


# ---------------------------------------------------------------------------
# Trunk
# ---------------------------------------------------------------------------


class Trunk(nn.Module):
    """Observation -> one state vector, shared by both policies."""

    def __init__(self, cfg: AgentConfig, n_registers: int) -> None:
        super().__init__()
        self.grid_encoder = GridEncoder(
            side=cfg.grid_side, channels=cfg.grid_channels, out_dim=cfg.grid_dim
        )
        self.pair_encoder = PairEncoder(
            self.grid_encoder, feat_dim=PAIR_FEATURE_DIM, d_model=cfg.d_model
        )
        self.set_attention = SetAttention(cfg.d_model, cfg.set_heads, cfg.set_layers)
        self.to_state = nn.Sequential(
            nn.Linear(cfg.d_model + n_registers + SCALAR_DIM, cfg.state_dim),
            nn.ELU(),
            nn.Linear(cfg.state_dim, cfg.state_dim),
        )
        self.state_dim = cfg.state_dim

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = self.pair_encoder(obs["grids"], obs["grid_shapes"], obs["pair_feats"])
        pooled, _ = self.set_attention(tokens, obs["slot_mask"])
        x = torch.cat([pooled, obs["reg_occ"], obs["scalars"]], dim=-1)
        return self.to_state(x)


# ---------------------------------------------------------------------------
# Factored action head
# ---------------------------------------------------------------------------


class FactoredActor(nn.Module):
    """Operator head followed by autoregressive, masked argument heads.

    Sampling one head at a time conditioned on the previous choices is what
    keeps this tractable: the joint space is ``n_ops * n_args^5`` (about 1.5
    billion), while the factored form is six softmaxes over at most 98 options.
    Every softmax is masked to what the machine will actually accept, so an
    invalid statement is not merely discouraged -- it has probability zero.
    """

    def __init__(self, hidden_dim: int, spec: MachineSpec, emb_dim: int = 32) -> None:
        super().__init__()
        self.spec = spec
        self.n_ops = spec.n_ops
        self.n_args = spec.n_args
        self.max_arity = spec.max_arity

        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, self.n_ops)
        )
        self.op_emb = nn.Embedding(self.n_ops, emb_dim)
        self.arg_emb = nn.Embedding(self.n_args, emb_dim)
        self.pos_emb = nn.Embedding(self.max_arity, emb_dim)
        self.arg_head = nn.Sequential(
            nn.Linear(hidden_dim + 3 * emb_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, self.n_args),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        op_mask: torch.Tensor,
        arg_mask: torch.Tensor,
        actions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(op, args, logp, entropy)``.

        ``actions`` re-evaluates a stored action instead of sampling, which is
        what the update pass needs.
        """
        batch = hidden.shape[0]
        device = hidden.device

        op_logits = self.op_head(hidden).masked_fill(~op_mask, _NEG)
        op_dist = torch.distributions.Categorical(logits=op_logits / temperature)
        op = actions[0] if actions is not None else op_dist.sample()
        logp = op_dist.log_prob(op)
        entropy = op_dist.entropy()

        # arg_mask is (B, n_ops, arity, n_args); select the chosen operator's rows.
        chosen = arg_mask[torch.arange(batch, device=device), op]  # (B, arity, n_args)

        e_op = self.op_emb(op)
        running = torch.zeros(batch, e_op.shape[-1], device=device)
        args = torch.zeros(batch, self.max_arity, dtype=torch.long, device=device)

        for p in range(self.max_arity):
            mask_p = chosen[:, p]
            active = mask_p.any(dim=-1)
            e_pos = self.pos_emb(torch.full((batch,), p, dtype=torch.long, device=device))
            logits = self.arg_head(torch.cat([hidden, e_op, running, e_pos], dim=-1))
            logits = logits.masked_fill(~mask_p, _NEG)
            # Inactive rows (this operator has no parameter p) would be all
            # -inf; give them a flat distribution and discard their statistics.
            logits = torch.where(active.unsqueeze(-1), logits, torch.zeros_like(logits))
            dist = torch.distributions.Categorical(logits=logits / temperature)
            a = actions[1][:, p] if actions is not None else dist.sample()
            a = torch.where(active, a, torch.zeros_like(a))
            args[:, p] = a
            logp = logp + torch.where(active, dist.log_prob(a), torch.zeros_like(logp))
            entropy = entropy + torch.where(active, dist.entropy(), torch.zeros_like(entropy))
            running = running + self.arg_emb(a) * active.unsqueeze(-1)

        return op, args, logp, entropy


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class DirectorAgent(nn.Module):
    """Manager + worker over a shared trunk, with a discrete goal space."""

    def __init__(self, cfg: AgentConfig, spec: MachineSpec) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = spec

        self.trunk = Trunk(cfg, n_registers=spec.n_registers)
        self.goal_ae = GoalAutoencoder(
            cfg.state_dim,
            n_groups=cfg.goal_groups,
            n_codes=cfg.goal_codes,
            hidden=cfg.goal_hidden,
            kl_weight=cfg.goal_kl_weight,
        )

        # -- manager --------------------------------------------------
        self.manager_lstm = nn.LSTMCell(cfg.state_dim + cfg.state_dim + 2, cfg.lstm_dim)
        self.manager_head = nn.Linear(cfg.lstm_dim, cfg.goal_groups * cfg.goal_codes)
        self.manager_value_extr = nn.Linear(cfg.lstm_dim, 1)
        self.manager_value_expl = nn.Linear(cfg.lstm_dim, 1)

        # -- worker ---------------------------------------------------
        self.op_emb = nn.Embedding(spec.n_ops, cfg.action_emb)
        self.arg_emb = nn.Embedding(spec.n_args, cfg.action_emb)

        mem_in = cfg.state_dim + 2 * cfg.action_emb
        self.memory_token = nn.Sequential(
            nn.Linear(mem_in, cfg.mem_dim), nn.ELU(), nn.Linear(cfg.mem_dim, cfg.mem_dim)
        )
        if cfg.use_memory:
            self.memory_attention = EpisodicMemory(
                cfg.mem_dim, cfg.state_dim, d_model=cfg.d_model,
                n_heads=cfg.mem_heads, max_len=cfg.mem_len,
            )
            mem_out = cfg.d_model
        else:
            self.memory_attention = None
            mem_out = 0

        worker_in = (
            cfg.state_dim          # current state
            + cfg.state_dim        # goal
            + 2 * cfg.action_emb   # previous action
            + 3                    # previous reward, previous error, goal progress
            + mem_out
        )
        self.worker_lstm = nn.LSTMCell(worker_in, cfg.lstm_dim)
        self.actor = FactoredActor(cfg.lstm_dim, spec, emb_dim=cfg.action_emb)
        self.worker_value = nn.Linear(cfg.lstm_dim, 1)

    # -- state -----------------------------------------------------------
    def initial_state(self, batch: int, device: torch.device) -> AgentState:
        cfg = self.cfg

        def z(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device)

        return AgentState(
            worker_h=z(batch, cfg.lstm_dim),
            worker_c=z(batch, cfg.lstm_dim),
            manager_h=z(batch, cfg.lstm_dim),
            manager_c=z(batch, cfg.lstm_dim),
            memory=z(batch, cfg.mem_len, cfg.mem_dim),
            memory_mask=z(batch, cfg.mem_len),
            goal=z(batch, cfg.state_dim),
            goal_codes=torch.zeros(batch, cfg.goal_groups, dtype=torch.long, device=device),
            since_goal=torch.zeros(batch, dtype=torch.long, device=device),
            step_index=torch.zeros(batch, dtype=torch.long, device=device),
        )

    # -- one step --------------------------------------------------------
    def step(
        self,
        obs: Dict[str, torch.Tensor],
        state: AgentState,
        feedback: torch.Tensor,
        actions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        manager_action: Optional[torch.Tensor] = None,
        state_vec: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> StepOutput:
        """Run manager (when due) and worker for one environment step.

        Parameters
        ----------
        feedback:
            ``(B, 2)`` of previous reward and previous-step error flag.
        actions, manager_action:
            Supplied during the update pass to re-evaluate stored actions
            instead of sampling new ones.
        state_vec:
            A trunk output computed in one batched pass over a whole rollout,
            so the Python loop here carries only the recurrent parts.
        """
        s = self.trunk(obs) if state_vec is None else state_vec
        batch = s.shape[0]
        cfg = self.cfg

        # -- manager, at the abstract time scale ----------------------
        due = (state.since_goal % cfg.manager_every == 0)
        s_manager = s if cfg.manager_grads_to_trunk else s.detach()
        manager_in = torch.cat([s_manager, state.goal, feedback], dim=-1)
        mh, mc = self.manager_lstm(manager_in, (state.manager_h, state.manager_c))
        keep = due.float().unsqueeze(-1)
        manager_h = keep * mh + (1 - keep) * state.manager_h
        manager_c = keep * mc + (1 - keep) * state.manager_c

        code_logits = self.manager_head(manager_h).view(batch, cfg.goal_groups, cfg.goal_codes)
        code_dist = torch.distributions.Categorical(logits=code_logits / temperature)
        codes = manager_action if manager_action is not None else code_dist.sample()
        manager_logp = code_dist.log_prob(codes).sum(-1)
        manager_entropy = code_dist.entropy().sum(-1)
        value_extr = self.manager_value_extr(manager_h).squeeze(-1)
        value_expl = self.manager_value_expl(manager_h).squeeze(-1)

        # Detached: the goal decoder is trained by the autoencoder's own
        # reconstruction loss and by nothing else. Without this the manager's
        # policy gradient would reshape the goal space to make its own goals
        # easier to reach, which is the hierarchical version of moving the
        # goalposts -- and the goal would also keep a live graph alive across
        # every step of the window it is held for.
        new_goal = self.goal_ae.decode_indices(codes).detach()
        goal = torch.where(due.unsqueeze(-1), new_goal, state.goal)
        goal_codes = torch.where(due.unsqueeze(-1), codes, state.goal_codes)

        # -- worker ---------------------------------------------------
        last = obs["last_action"]
        prev_action = torch.cat(
            [self.op_emb(last[:, 0]), self.arg_emb(last[:, 1:]).sum(dim=1)], dim=-1
        )
        progress = (state.since_goal % cfg.manager_every).float() / cfg.manager_every
        parts = [
            s,
            goal.detach(),  # the goal is a target, not something the worker shapes
            prev_action,
            feedback,
            progress.unsqueeze(-1),
        ]
        if self.memory_attention is not None:
            parts.append(self.memory_attention(state.memory, state.memory_mask, s))
        wh, wc = self.worker_lstm(torch.cat(parts, dim=-1), (state.worker_h, state.worker_c))

        op, args, logp, entropy = self.actor(
            wh, obs["op_mask"], obs["arg_mask"], actions=actions, temperature=temperature
        )
        worker_value = self.worker_value(wh).squeeze(-1)

        # -- carry ----------------------------------------------------
        token = self.memory_token(
            torch.cat([s, self.op_emb(op), self.arg_emb(args).sum(dim=1)], dim=-1)
        )
        memory, memory_mask = write_memory(
            state.memory, state.memory_mask, token, state.step_index
        )

        next_state = AgentState(
            worker_h=wh,
            worker_c=wc,
            manager_h=manager_h,
            manager_c=manager_c,
            memory=memory,
            memory_mask=memory_mask,
            goal=goal,
            goal_codes=goal_codes,
            since_goal=state.since_goal + 1,
            step_index=state.step_index + 1,
        )

        return StepOutput(
            op=op,
            args=args,
            worker_logp=logp,
            worker_entropy=entropy,
            worker_value=worker_value,
            manager_active=due,
            manager_codes=codes,
            manager_logp=manager_logp,
            manager_entropy=manager_entropy,
            manager_value_extr=value_extr,
            manager_value_expl=value_expl,
            state_vec=s,
            goal=goal,
            next_state=next_state,
        )
