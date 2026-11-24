"""Streamlit AppTest checks: pages render, filters and scenarios take effect, uploads stay in one session."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from funnelforge.core import prepare
from scripts.make_messy_sample import build

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
PAGES = ["", "segments", "daily", "scenarios", "data", "method"]


def run(page="", **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if page:
        at._page_hash = next(h for h, i in at._registered_pages.items() if i["url_pathname"] == page)
        at.run()
    return at


def text(at):
    return " ".join(m.value for m in at.markdown)


def upload_state():
    clean, rep = prepare(build())
    return dict(upload_df=clean, upload_report=rep, upload_name="ga_export.csv", upload_key="k1")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error
    assert "Synthetic demonstration data" in text(at)


def test_funnel_page_numbers_and_truth():
    t = text(run(""))
    assert "4,000" in t and "444" in t and "Synthetic check" in t and "Add to cart" in t


@pytest.mark.parametrize("page", ["", "segments", "daily", "scenarios", "data"])
def test_upload_used_everywhere(page):
    at = run(page, **upload_state())
    assert not at.exception, [e.value for e in at.exception]
    assert "Your uploaded file" in text(at) and "Synthetic check" not in text(at)


def test_upload_is_isolated_between_sessions():
    a = run("", **upload_state())
    b = run("")
    assert "Your uploaded file" in text(a)
    assert "Synthetic demonstration data" in text(b) and "upload_df" not in b.session_state


def test_upload_is_never_written_to_disk(tmp_path):
    before = {p for p in ROOT.rglob("*") if ".git" not in p.parts and "__pycache__" not in p.parts}
    run("data", **upload_state())
    after = {p for p in ROOT.rglob("*") if ".git" not in p.parts and "__pycache__" not in p.parts}
    assert after - before == set()


def test_filters_change_results():
    at = run("")
    at.sidebar.multiselect[1].set_value(["Desktop"]).run()
    t = text(at)
    assert not at.exception and "Filtered:" in t and "1,440" in t


def test_empty_filter_shows_warning():
    at = run("segments")
    at.sidebar.multiselect[0].set_value([]).run()
    assert not at.exception and at.warning


def test_segment_dimension_switch():
    at = run("segments")
    assert "same conversion" in text(at)
    at.button_group[0].set_value("device").run()
    assert not at.exception and "Tablet" in text(at) and "Mobile" in text(at)


def test_scenario_sliders():
    at = run("scenarios")
    at.slider[0].set_value(10).run()
    assert not at.exception and "Scenario purchases" in text(at)
    at.slider[3].set_value(30).run()
    assert not at.exception and "capped" in text(at)


def test_switch_back_to_demo_keeps_upload():
    at = run("", **upload_state())
    at.sidebar.radio[0].set_value("Synthetic demo (4,000 sessions)").run()
    assert not at.exception and "Synthetic demonstration data" in text(at) and "upload_df" in at.session_state
