"""Dataset preparation prefers an existing local checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_preparer():
    path = ROOT / "scripts/prepare_arc_data.py"
    spec = importlib.util.spec_from_file_location("arc_prepare_data", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_copies_local_splits_without_cloning(tmp_path, monkeypatch):
    preparer = load_preparer()
    source = tmp_path / "source"
    for split in preparer.SPLITS:
        folder = source / split
        folder.mkdir(parents=True)
        (folder / "task.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(preparer, "DATA", tmp_path / "target")
    monkeypatch.setattr(preparer, "_local_candidates", lambda _name: [source])

    record = preparer.prepare("arc2", "https://invalid.example/repo.git")

    assert record["source"] == str(source)
    assert (preparer.DATA / "arc2" / "training" / "task.json").is_file()
    assert (preparer.DATA / "arc2" / "evaluation" / "task.json").is_file()
