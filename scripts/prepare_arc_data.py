"""Prepare ARC-AGI-1 and ARC-AGI-2 data without needless downloads.

Local checkouts are preferred (including the predecessor project beside this
repository).  Missing datasets are shallow-cloned from their official GitHub
repositories, and only the JSON task folders are copied into ``data/``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DATASETS = {
    "arc1": "https://github.com/fchollet/ARC-AGI.git",
    "arc2": "https://github.com/arcprize/ARC-AGI-2.git",
}
SPLITS = ("training", "evaluation")


def _counts(root: Path) -> dict[str, int]:
    return {split: len(list((root / split).glob("*.json"))) for split in SPLITS}


def _ready(root: Path) -> bool:
    counts = _counts(root)
    return all(counts[split] > 0 for split in SPLITS)


def _local_candidates(name: str) -> list[Path]:
    candidates = [ROOT.parent / "arc-2-solution" / "data" / name]
    if name == "arc1":
        candidates.extend([ROOT.parent / "ARC-AGI" / "data", ROOT.parent / "ARC-AGI"])
    else:
        candidates.extend(
            [ROOT.parent / "ARC-AGI-2" / "data", ROOT.parent / "ARC-AGI-2"]
        )
    return candidates


def _copy_splits(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        folder = source / split
        if not folder.is_dir():
            raise FileNotFoundError(f"dataset source has no {folder}")
        shutil.copytree(folder, target / split, dirs_exist_ok=True)


def prepare(name: str, url: str) -> dict[str, object]:
    target = DATA / name
    if _ready(target):
        counts = _counts(target)
        print(f"{name}: already ready {counts}", flush=True)
        return {"source": "existing", "counts": counts}

    for candidate in _local_candidates(name):
        if _ready(candidate):
            print(f"{name}: copying local tasks from {candidate}", flush=True)
            _copy_splits(candidate, target)
            return {"source": str(candidate), "counts": _counts(target)}

    DATA.mkdir(parents=True, exist_ok=True)
    temp_root = DATA / ".download"
    temp_root.mkdir(parents=True, exist_ok=True)
    print(f"{name}: cloning official dataset {url}", flush=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as temporary:
        checkout = Path(temporary) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(checkout)],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        source = checkout / "data"
        _copy_splits(source, target)
    return {"source": url, "revision": revision, "counts": _counts(target)}


def main() -> int:
    records: dict[str, object] = {}
    try:
        for name, url in DATASETS.items():
            records[name] = prepare(name, url)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"ARC data preparation failed: {error}", file=sys.stderr)
        return 1
    manifest = DATA / "dataset_revisions.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"ARC data ready; provenance recorded in {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
