"""Unit tests for the task logic. They need only pytest, no GPU, no model and no Streamlit."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import triage_core as core  # noqa: E402


def good(**over):
    obj = {"category": "ORDER", "intent": "cancel_order", "priority": "medium",
           "reply": "I can help with order {{Order Number}}. We will cancel it for you."}
    obj.update(over)
    return json.dumps(obj)


# ---- the task definition must match what the model was trained on -------------------------------------------
def test_system_prompt_is_identical_to_the_one_used_in_training():
    trained = (ROOT / "samples" / "system_prompt_from_training.txt").read_text()
    assert core.SYSTEM_PROMPT == trained


def test_label_space():
    assert len(core.CATEGORIES) == 11 and len(core.INTENTS) == 27
    assert set(core.INTENT_TO_CATEGORY.values()) == set(core.CATEGORIES)


# ---- priority rule ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("intent,text,expected", [
    ("track_order", "where is my order", "low"),
    ("track_order", "where is my order ASAP", "medium"),
    ("cancel_order", "please cancel my order", "medium"),
    ("cancel_order", "cancel it right now", "high"),
    ("payment_issue", "my card is declined", "high"),
    ("complaint", "this is unacceptable!!", "high"),       # already high stays high
])
def test_derive_priority(intent, text, expected):
    assert core.derive_priority(intent, text) == expected


# ---- parsing ------------------------------------------------------------------------------------------------
def test_parse_strict():
    obj, strict = core.parse_output(good())
    assert strict and obj["intent"] == "cancel_order"


def test_parse_with_code_fence_and_chatter():
    obj, strict = core.parse_output("Sure!\n```json\n" + good() + "\n```")
    assert obj is not None and not strict


@pytest.mark.parametrize("text", ["", "not json at all", "{broken", "[1, 2]", None])
def test_parse_rejects_garbage(text):
    assert core.parse_output(text)[0] is None


def test_schema_problems_are_specific():
    bad = json.loads(good(intent="accept_payment_methods", priority="urgent"))
    msgs = " | ".join(core.schema_problems(bad))
    assert "accept_payment_methods" in msgs and "priority" in msgs
    assert core.schema_problems(json.loads(good())) == []
    assert core.schema_problems({"category": "ORDER"})            # missing keys


# ---- invented numbers ---------------------------------------------------------------------------------------
def test_invented_numbers():
    assert core.invented_numbers("Your order 12345 ships in 3 days.", "where is order 12345") == ["3"]
    assert core.invented_numbers("Order {{Order Number}} is on its way.", "where is my order") == []
    assert core.invented_numbers("Call us 24/7.\n1. Log in\n2. Click reset", "reset password") == []


# ---- placeholders -------------------------------------------------------------------------------------------
def test_placeholders_and_fill():
    reply = "Hi {{Person Name}}, order {{Order Number}} (again {{Order Number}}) is fine."
    assert core.placeholders_in(reply) == ["Person Name", "Order Number"]
    out = core.fill_placeholders(reply, {"Order Number": "77231", "Person Name": "  "})
    assert out == "Hi {{Person Name}}, order 77231 (again 77231) is fine."   # blank value stays visible


def test_suggest_values():
    assert core.suggest_values("cancel order 77231 please") == {"Order Number": "77231"}
    assert core.suggest_values("order #40518") == {"Order Number": "40518"}
    assert core.suggest_values("no numbers here") == {}


# ---- confidence from token probabilities --------------------------------------------------------------------
def test_value_confidence_multiplies_only_the_value_tokens():
    toks = ['{"', 'intent', '":"', 'cancel', '_order', '","', 'priority']
    probs = [0.99, 0.99, 0.99, 0.9, 0.5, 0.99, 0.99]
    assert core.value_confidence(toks, probs, "intent") == pytest.approx(0.45)
    assert core.value_confidence(toks, probs, "category") is None
    assert core.value_confidence([], [], "intent") is None


# ---- analyze(): the safety checks ---------------------------------------------------------------------------
def names(res, status):
    return {c.name for c in res.checks if c.status == status}


def test_clean_output_is_ready():
    msg = "please cancel order 77231"
    toks = list(good()); probs = [0.99] * len(toks)
    res = core.analyze(msg, good(), toks, probs)
    assert res.parsed_ok and res.verdict == "ready" and res.final_priority == "medium"
    assert res.placeholders == ["Order Number"]


def test_garbage_goes_to_manual():
    res = core.analyze("hello", "I am sorry, I cannot do that.")
    assert res.verdict == "manual" and "valid JSON" in names(res, "fail")


def test_invented_label_goes_to_manual():
    res = core.analyze("do you take paypal", good(intent="accept_payment_methods"))
    assert res.verdict == "manual" and "schema" in names(res, "fail")


def test_category_intent_mismatch_is_flagged():
    res = core.analyze("cancel order 1234", good(category="PAYMENT"))
    assert res.verdict == "check" and "category fits intent" in names(res, "fail")


def test_invented_number_is_flagged():
    res = core.analyze("cancel my order", good(reply="Cancelled. Refund arrives in 5 days."))
    assert res.verdict == "check" and "no invented numbers" in names(res, "fail")


def test_priority_disagreement_escalates_to_the_higher_level():
    res = core.analyze("cancel my order right now", good(priority="low"))
    assert res.model_priority == "low" and res.rule_priority == "high" and res.final_priority == "high"
    assert "priority agrees with rule" in names(res, "warn")


def test_low_confidence_warns():
    text = good(); toks = list(text); probs = [0.99] * len(toks)
    i = text.index("cancel_order")
    for k in range(i, i + len("cancel_order")):
        probs[k] = 0.9 ** 3        # 12 tokens at ~0.73 each -> far below the floor
    res = core.analyze("cancel it", text, toks, probs, confidence_floor=0.8)
    assert res.intent_confidence < 0.8 and "model certainty" in names(res, "warn")


def test_extracted_json_is_a_warning_not_a_failure():
    res = core.analyze("cancel order 1", "```json\n" + good() + "\n```")
    assert res.parsed_ok and "valid JSON" in names(res, "warn")


# ---- input warnings -----------------------------------------------------------------------------------------
def test_input_warnings():
    assert core.input_warnings("") == ["The message is empty."]
    assert core.input_warnings("Where is my package?") == []
    assert any("several separate requests" in w for w in core.input_warnings(
        "Change my address. Also where is my refund. And I need an invoice. Thanks."))
    assert any("much longer" in w for w in core.input_warnings("word " * 100))


# ---- batch helpers ------------------------------------------------------------------------------------------
def test_read_messages_csv():
    data = b"id,Message\n1,where is my order\n2,\n3,  cancel   my order \n"
    assert core.read_messages_csv(data) == ["where is my order", "cancel my order"]
    assert core.read_messages_csv("just text\nline two\n") == ["line two"]    # falls back to column 0, skips header


def test_sort_queue_puts_manual_first_then_by_priority():
    a = core.analyze("where is my order", good(intent="track_order", category="ORDER", priority="low"))
    b = core.analyze("my card is declined", good(intent="payment_issue", category="PAYMENT", priority="high"))
    c = core.analyze("???", "garbage")
    assert [r.user_text for r in core.sort_queue([a, b, c])] == ["???", "my card is declined", "where is my order"]


def test_records_to_csv_roundtrip():
    res = core.analyze("cancel order 77231", good())
    text = core.records_to_csv([res.to_record()])
    assert text.splitlines()[0].startswith("message,category,intent,priority")
    assert "77231" in text
