"""The dependency-free live dashboard and its durable state."""

from __future__ import annotations

import json
import urllib.request

from arc_director.dashboard import DashboardServer


def test_dashboard_names_the_director_metrics_and_escapes_title(tmp_path):
    with DashboardServer(
        tmp_path,
        "Director <test>",
        port=None,
        phase="Self-generated DSL programs",
        phase_index=1,
        phase_total=2,
    ) as dashboard:
        dashboard.update(status="training", updates=7)
        page = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert "Director &lt;test&gt;" in page
    assert "Held-out generalization" in page
    assert "Demonstration solve rate" in page
    assert "ARC evaluation" in page
    assert "ARC-AGI-1 exact @ 2" in page
    assert "ARC-AGI-2 exact @ 2" in page
    assert "Top operator share" in page
    assert "generated programs" in page.lower()
    state = json.loads((tmp_path / "dashboard.json").read_text(encoding="utf-8"))
    assert state["phase"] == "Self-generated DSL programs"
    assert state["phase_index"] == 1
    assert state["phase_total"] == 2
    assert state["updates"] == 7


def test_dashboard_serves_the_run_directory_on_an_ephemeral_port(tmp_path):
    with DashboardServer(tmp_path, port=0) as dashboard:
        assert dashboard.url is not None
        with urllib.request.urlopen(dashboard.url, timeout=2) as response:
            page = response.read().decode("utf-8")
    assert "ARC Director" in page
