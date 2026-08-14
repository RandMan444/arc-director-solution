"""Configuration loading and object construction.

One YAML file describes a run; :func:`build` turns it into the four objects
that matter (spec, environment, agent, trainer). Unknown keys are an error
rather than a shrug -- a silently ignored ``lr`` in the wrong block is the kind
of thing that costs a day of confused staring at flat learning curves.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .arc.grid import MAX_SIDE
from .curriculum.sources import CurriculumSource, load_arc_tasks
from .curriculum.stages import DEFAULT_LADDER, Stage
from .dsl.machine import INT_LITERALS, MachineSpec
from .env.task_env import EnvConfig
from .env.vec import VecProgramEnv
from .models.agent import AgentConfig, DirectorAgent
from .train.director import DirectorTrainer, TrainConfig

__all__ = ["load_config", "build", "build_spec", "DEFAULTS"]


DEFAULTS: Dict[str, Any] = {
    "seed": 0,
    "machine": {
        "grid_registers": 4,
        "object_registers": 2,
        "object_set_registers": 2,
        "integer_registers": 2,
        "bool_registers": 1,
    },
    "curriculum": {
        "ladder": "default",
        "start_stage": 0,
        "auto_promote": True,
        "arc_root": "data/arc2",
        "arc_split": "training",
        "reachable_file": None,
        "dev_holdout": 0,
    },
    "env": {},
    "agent": {},
    "train": {},
}


def _from_dict(cls, values: Dict[str, Any]):
    """Instantiate a dataclass, rejecting keys it does not declare."""
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(
            f"{cls.__name__} has no option(s) {sorted(unknown)}; valid: {sorted(known)}"
        )
    return cls(**values)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - set(DEFAULTS) - {"run_dir", "device", "notes"}
    if unknown:
        raise KeyError(f"unknown top-level config key(s): {sorted(unknown)}")
    return _deep_merge(DEFAULTS, raw)


def build_spec(cfg: Dict[str, Any]) -> MachineSpec:
    m = cfg.get("machine", {})
    from .dsl.types import DslType

    layout = (
        (DslType.GRID, int(m.get("grid_registers", 4))),
        (DslType.OBJECT, int(m.get("object_registers", 2))),
        (DslType.OBJECT_SET, int(m.get("object_set_registers", 2))),
        (DslType.INTEGER, int(m.get("integer_registers", 2))),
        (DslType.BOOL, int(m.get("bool_registers", 1))),
    )
    return MachineSpec.build(
        ops=m.get("ops"),
        layout=layout,
        int_literals=m.get("int_literals", INT_LITERALS),
    )


#: Ladder presets. ``warmup`` stops before ARC, which is what lets a warm-up
#: run use a small ``grid_side``; ``arc`` skips the generated rungs for
#: fine-tuning a checkpoint that already has them.
LADDER_PRESETS = {
    "default": DEFAULT_LADDER,
    "warmup": tuple(s for s in DEFAULT_LADDER if s.kind == "warmup"),
    "arc": tuple(s for s in DEFAULT_LADDER if s.kind == "arc"),
}


def _build_ladder(spec_cfg: Any) -> Tuple[Stage, ...]:
    if spec_cfg is None:
        return DEFAULT_LADDER
    if isinstance(spec_cfg, str):
        if spec_cfg in LADDER_PRESETS:
            return LADDER_PRESETS[spec_cfg]
        raise ValueError(
            f"unknown ladder preset {spec_cfg!r}; have {sorted(LADDER_PRESETS)}"
        )
    stages = []
    for entry in spec_cfg:
        entry = dict(entry)
        if "groups" in entry and entry["groups"] is not None:
            entry["groups"] = tuple(entry["groups"])
        if "n_ops" in entry:
            entry["n_ops"] = tuple(entry["n_ops"])
        stages.append(_from_dict(Stage, entry))
    return tuple(stages)


def _load_reachable(path: Optional[str]) -> List[str]:
    if not path:
        return []
    file = Path(path)
    if not file.exists():
        return []
    import json

    blob = json.loads(file.read_text(encoding="utf-8"))
    if isinstance(blob, dict):
        return list(blob.get("task_ids", []))
    return list(blob)


def build(
    cfg: Dict[str, Any], run_dir: Optional[str] = None
) -> Tuple[MachineSpec, VecProgramEnv, DirectorAgent, DirectorTrainer]:
    """Build every object a run needs, validated against each other."""
    spec = build_spec(cfg)

    cur = cfg["curriculum"]
    stages = _build_ladder(cur.get("ladder", "default"))
    needs_arc = any(s.kind == "arc" for s in stages)
    # ``arc_root`` may be a list: ARC-1 training is a legitimate extra pool
    # (ARC-2's training set already overlaps it heavily) and roughly doubles the
    # number of tasks the coverage search can certify. Evaluation splits must
    # never appear here.
    roots = cur["arc_root"]
    roots = [roots] if isinstance(roots, str) else list(roots)
    arc_tasks: List = []
    if needs_arc:
        seen = set()
        for root in roots:
            for task in load_arc_tasks(root, cur["arc_split"]):
                if task.task_id in seen:
                    continue
                seen.add(task.task_id)
                arc_tasks.append(task)

    # An internal dev split, chosen deterministically by task id so it is the
    # same set on every run and every machine. These tasks never enter a
    # training pool, which makes periodic evaluation the one number in the run
    # that is not measured on data the agent trains on. The public evaluation
    # split is a separate thing again and is only touched by scripts/evaluate.py.
    dev_tasks: List = []
    n_dev = int(cur.get("dev_holdout", 0) or 0)
    if needs_arc and n_dev:
        ordered = sorted(arc_tasks, key=lambda t: t.task_id)
        stride = max(1, len(ordered) // n_dev)
        held = {id(t) for t in ordered[::stride][:n_dev]}
        dev_tasks = [t for t in arc_tasks if id(t) in held]
        arc_tasks = [t for t in arc_tasks if id(t) not in held]
    source = CurriculumSource(
        spec,
        stages,
        arc_tasks=arc_tasks,
        reachable_ids=_load_reachable(cur.get("reachable_file")),
        start_stage=int(cur.get("start_stage", 0)),
        auto_promote=bool(cur.get("auto_promote", True)),
    )

    env_cfg = _from_dict(EnvConfig, cfg.get("env", {}))
    agent_cfg = _from_dict(AgentConfig, cfg.get("agent", {}))
    train_values = dict(cfg.get("train", {}))
    if run_dir or cfg.get("run_dir"):
        train_values["run_dir"] = run_dir or cfg["run_dir"]
    if cfg.get("device"):
        train_values["device"] = cfg["device"]
    train_values.setdefault("seed", int(cfg.get("seed", 0)))
    train_cfg = _from_dict(TrainConfig, train_values)

    _validate(stages, agent_cfg, env_cfg)

    env = VecProgramEnv(
        train_cfg.n_envs, spec, source, env_cfg, seed=int(cfg.get("seed", 0))
    )
    agent = DirectorAgent(agent_cfg, spec)
    trainer = DirectorTrainer(spec, env, agent, train_cfg, dev_tasks=dev_tasks)
    return spec, env, agent, trainer


def _validate(stages: Sequence[Stage], agent_cfg: AgentConfig, env_cfg: EnvConfig) -> None:
    """Catch the config mistakes that would otherwise show up as silence."""
    needed = 0
    for stage in stages:
        needed = max(needed, MAX_SIDE if stage.kind == "arc" else stage.max_side)
    if agent_cfg.grid_side < needed:
        raise ValueError(
            f"agent.grid_side={agent_cfg.grid_side} but the ladder shows grids up to "
            f"{needed}. The encoder would only ever see the top-left corner. "
            "(grid_side changes no parameter shapes, so a checkpoint trained at a "
            "smaller side can be loaded at a larger one.)"
        )
    if env_cfg.holdout_demo and env_cfg.min_visible_demos < 1:
        raise ValueError("env.min_visible_demos must be >= 1")
    longest = max(stage.max_steps for stage in stages)
    if agent_cfg.use_memory and agent_cfg.mem_len < longest:
        raise ValueError(
            f"agent.mem_len={agent_cfg.mem_len} is shorter than the longest episode "
            f"({longest} steps); later statements would fall out of the episodic memory"
        )
