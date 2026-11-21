"""End-to-end test of the Streamlit page with Streamlit's own headless test runner (no browser, no model).
It uses the stub backend, so it checks the page logic, not the language model."""
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("TRIAGE_BACKEND", "stub")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    return at


def test_page_loads_and_warns_about_demo_mode(app):
    assert not app.exception
    assert any("DEMO MODE" in e.value for e in app.error)


def test_empty_message_is_rejected(app):
    next(b for b in app.button if b.label == "Triage").click().run()
    assert any("enter a message" in e.value for e in app.error)


def test_triage_a_message(app):
    app.text_area(key="msg").set_value("my card keeps getting declined at checkout!!").run()
    next(b for b in app.button if b.label == "Triage").click().run()
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Category"] == "PAYMENT" and metrics["Intent"] == "payment_issue"
    assert any("HIGH" in m.value for m in app.markdown)
    assert app.session_state["n_gen"] == 1 and app.session_state["n_valid"] == 1


def test_example_button_fills_the_box(app):
    next(b for b in app.button if b.label == "Use example").click().run()
    assert app.text_area(key="msg").value
