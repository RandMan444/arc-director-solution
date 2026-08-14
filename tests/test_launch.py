"""The launch button safely sequences generated programs and ARC training."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location("arc_director_launch", ROOT / "scripts/run_launch.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configs(tmp_path):
    warm_dir = tmp_path / "warm"
    arc_dir = tmp_path / "arc"
    warm = tmp_path / "warm.yaml"
    full = tmp_path / "full.yaml"
    warm.write_text(f"run_dir: {warm_dir.as_posix()}\n", encoding="utf-8")
    full.write_text(f"run_dir: {arc_dir.as_posix()}\n", encoding="utf-8")
    return warm, full, warm_dir / "checkpoint.pt", arc_dir / "checkpoint.pt"


def test_fresh_launch_runs_generated_programs_then_arc(tmp_path, monkeypatch):
    launcher = load_launcher()
    warm, full, warm_checkpoint, _ = configs(tmp_path)
    calls = []

    def fake_run(script, args):
        calls.append((script, list(args)))
        if script == "train.py" and str(warm) in args:
            warm_checkpoint.parent.mkdir(parents=True)
            warm_checkpoint.write_bytes(b"checkpoint")
        return 0

    monkeypatch.setattr(launcher, "_run", fake_run)
    monkeypatch.setattr(launcher, "_prepare_arc_inputs", lambda _cfg: 0)
    assert launcher.main(
        ["--warmup-config", str(warm), "--arc-config", str(full), "--fresh"]
    ) == 0

    assert [script for script, _args in calls] == ["train.py", "train.py"]
    assert "Self-generated DSL programs" in calls[0][1]
    assert calls[1][1][calls[1][1].index("--resume") + 1] == str(warm_checkpoint)
    assert "ARC-1 + ARC-2 curriculum" in calls[1][1]


def test_resume_prefers_the_furthest_arc_checkpoint(tmp_path, monkeypatch):
    launcher = load_launcher()
    warm, full, _warm_checkpoint, arc_checkpoint = configs(tmp_path)
    arc_checkpoint.parent.mkdir(parents=True)
    arc_checkpoint.write_bytes(b"checkpoint")
    calls = []
    prepared = []
    monkeypatch.setattr(launcher, "_run", lambda script, args: calls.append((script, list(args))) or 0)
    monkeypatch.setattr(launcher, "_prepare_arc_inputs", lambda cfg: prepared.append(cfg) or 0)

    assert launcher.main(
        ["--warmup-config", str(warm), "--arc-config", str(full), "--resume"]
    ) == 0
    assert len(calls) == 1
    assert calls[0][1][calls[0][1].index("--resume") + 1] == str(arc_checkpoint)
    assert len(prepared) == 1


def test_fresh_launch_refuses_to_overwrite_a_checkpoint(tmp_path):
    launcher = load_launcher()
    warm, full, warm_checkpoint, _ = configs(tmp_path)
    warm_checkpoint.parent.mkdir(parents=True)
    warm_checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(SystemExit, match="RESUME"):
        launcher.main(
            ["--warmup-config", str(warm), "--arc-config", str(full), "--fresh"]
        )


def test_resume_without_a_checkpoint_explains_how_to_start(tmp_path):
    launcher = load_launcher()
    warm, full, _, _ = configs(tmp_path)
    with pytest.raises(SystemExit, match="FRESH"):
        launcher.main(
            ["--warmup-config", str(warm), "--arc-config", str(full), "--resume"]
        )

