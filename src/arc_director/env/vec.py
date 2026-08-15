"""Synchronous vectorised environments.

Rollouts need a batch dimension for the network to be worth running, and the
DSL environment is microseconds per step, so a plain in-process loop beats any
subprocess machinery: there is nothing to hide behind IPC latency. Autoreset is
handled here, and the terminal observation is preserved in ``infos`` so
bootstrapping uses the right value.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dsl.machine import MachineSpec
from .task_env import EnvConfig, Observation, ProgramEnv

__all__ = ["VecProgramEnv", "stack_observations"]


def stack_observations(obs: Sequence[Observation]) -> Dict[str, np.ndarray]:
    """Stack a list of observations into a batched dict of arrays."""
    keys = obs[0].as_dict().keys()
    return {k: np.stack([o.as_dict()[k] for o in obs]) for k in keys}


class VecProgramEnv:
    """A fixed number of :class:`ProgramEnv` instances stepped in lockstep.

    Each environment gets its own RNG stream and its own task, so a batch spans
    several tasks at once -- which is what keeps the shared LSTM from latching
    onto one task's idiosyncrasies within a rollout.
    """

    def __init__(
        self,
        n_envs: int,
        spec: MachineSpec,
        source,
        config: Optional[EnvConfig] = None,
        seed: int = 0,
        share_source: bool = True,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.n_envs = n_envs
        self.spec = spec
        self.cfg = config or EnvConfig()
        # A shared source means one curriculum state for the whole batch, which
        # is what we want: promotion should be decided on all episodes, not on
        # whatever environment 0 happened to see.
        self.source = source
        self.envs: List[ProgramEnv] = [
            ProgramEnv(spec, source, self.cfg, seed=seed * 1000 + i) for i in range(n_envs)
        ]
        self.episode_count = 0

    # -- lifecycle -------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        return stack_observations([e.reset() for e in self.envs])

    def step(
        self, ops: np.ndarray, args: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step every environment, autoresetting the ones that finish.

        Returns ``(obs, rewards, dones, infos)`` where ``obs`` is already the
        *reset* observation for finished environments. The terminal info dict
        of a finished episode carries ``final_observation`` so a bootstrapped
        value can still be computed if a caller wants one; with a terminal
        episode there is nothing to bootstrap, so the trainer simply masks.
        """
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.float32)
        infos: List[Dict[str, Any]] = []
        observations: List[Observation] = []

        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(ops[i]), args[i])
            rewards[i] = reward
            dones[i] = float(done)
            if done:
                self.episode_count += 1
                info = dict(info)
                info["final_observation"] = obs
                obs = env.reset()
            infos.append(info)
            observations.append(obs)

        return stack_observations(observations), rewards, dones, infos

    # -- convenience -----------------------------------------------------
    @property
    def max_steps(self) -> int:
        return self.envs[0].max_steps

    def stats(self) -> Dict[str, Any]:
        out = dict(self.source.stats())
        out["episodes"] = self.episode_count
        return out

    # -- fork/restore ----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Capture mutable rollout state without copying the ARC task pool.

        Epiplexity candidates must begin at the exact same environment state.
        Deep-copying this whole object would also duplicate every immutable ARC
        task, so only each ProgramEnv's live episode is copied here.
        """
        skip = {"spec", "source", "cfg"}
        return {
            "episode_count": self.episode_count,
            "source": copy.deepcopy(self.source.state_dict()),
            "envs": [
                copy.deepcopy({k: v for k, v in vars(env).items() if k not in skip})
                for env in self.envs
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        if len(state["envs"]) != self.n_envs:
            raise ValueError(
                f"environment snapshot has {len(state['envs'])} envs, expected {self.n_envs}"
            )
        self.source.load_state_dict(copy.deepcopy(state["source"]))
        self.episode_count = int(state["episode_count"])
        for env, saved in zip(self.envs, state["envs"]):
            for key, value in copy.deepcopy(saved).items():
                setattr(env, key, value)
