"""Tests for model_backend that do not need PyTorch or a model."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_backend as mb  # noqa: E402
import triage_core as core  # noqa: E402


def test_missing_model_gives_a_helpful_error(tmp_path):
    with pytest.raises(mb.ModelNotFound) as e:
        mb.resolve_model_source(str(tmp_path / "nope"))
    assert "final_model" in str(e.value) and "TRIAGE_MODEL" in str(e.value)


def test_folder_without_config_is_rejected(tmp_path):
    with pytest.raises(mb.ModelNotFound):
        mb.resolve_model_source(str(tmp_path))


def test_folder_with_config_is_accepted(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert mb.resolve_model_source(str(tmp_path)) == str(tmp_path)


def test_hub_repo_id_is_accepted():
    assert mb.resolve_model_source("someone/support-triage-qwen") == "someone/support-triage-qwen"


def test_env_variable_is_used(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setenv("TRIAGE_MODEL", str(tmp_path))
    assert mb.resolve_model_source() == str(tmp_path)


def test_fingerprint_changes_with_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"a" * 10)
    f1 = mb.folder_fingerprint(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"a" * 11)
    assert f1 != mb.folder_fingerprint(str(tmp_path))


def test_stub_backend_runs_through_the_whole_pipeline():
    be = mb.load_backend(backend="stub")
    assert be.is_stub
    gen = be.generate("my card keeps getting declined!!")
    res = core.analyze("my card keeps getting declined!!", gen.text, gen.token_texts, gen.token_probs)
    assert res.parsed_ok and res.intent == "payment_issue" and res.final_priority == "high"
    assert res.intent_confidence == pytest.approx(0.95 ** len("payment_issue"))


def test_app_never_selects_the_stub_by_itself(monkeypatch):
    monkeypatch.delenv("TRIAGE_BACKEND", raising=False)
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    monkeypatch.chdir(ROOT)                       # no ./final_model in the repo
    with pytest.raises(mb.ModelNotFound):
        mb.load_backend()
