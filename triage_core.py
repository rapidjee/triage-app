"""triage_core.py - the task logic of the support-ticket triage app (no Streamlit, no PyTorch).

Everything here is plain Python so it can be unit-tested without a GPU or a web server.

The definitions in the "Task definition" block are copied from the Milestone 3 training notebook on purpose:
the fine-tuned model only works well if the app builds the prompt EXACTLY as it was built during training,
and if the app judges the model's output with the same rules that were used in the evaluation.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --------------------------------------------------------------------------------------------------------------
# Task definition (mirrors Milestone 3)
# --------------------------------------------------------------------------------------------------------------
# intent -> category, as found in the Bitext customer-support dataset (11 categories, 27 intents).
INTENT_TO_CATEGORY = {
    "create_account": "ACCOUNT", "delete_account": "ACCOUNT", "edit_account": "ACCOUNT",
    "recover_password": "ACCOUNT", "registration_problems": "ACCOUNT", "switch_account": "ACCOUNT",
    "check_cancellation_fee": "CANCEL",
    "contact_customer_service": "CONTACT", "contact_human_agent": "CONTACT",
    "delivery_options": "DELIVERY", "delivery_period": "DELIVERY",
    "complaint": "FEEDBACK", "review": "FEEDBACK",
    "check_invoice": "INVOICE", "get_invoice": "INVOICE",
    "cancel_order": "ORDER", "change_order": "ORDER", "place_order": "ORDER", "track_order": "ORDER",
    "check_payment_methods": "PAYMENT", "payment_issue": "PAYMENT",
    "check_refund_policy": "REFUND", "get_refund": "REFUND", "track_refund": "REFUND",
    "change_shipping_address": "SHIPPING", "set_up_shipping_address": "SHIPPING",
    "newsletter_subscription": "SUBSCRIPTION",
}
CATEGORIES = sorted(set(INTENT_TO_CATEGORY.values()))
INTENTS = sorted(INTENT_TO_CATEGORY)
LEVELS = ["low", "medium", "high"]
SCHEMA_KEYS = {"category", "intent", "priority", "reply"}

PH = re.compile(r"\{\{([^}]+)\}\}")  # placeholders such as {{Order Number}}

SYSTEM_PROMPT = (
    "You are the triage assistant for an online retailer's customer support team. Read the customer's "
    "message and respond with ONLY a JSON object with exactly these keys, in this order: category, intent, "
    "priority, reply.\n"
    "- category: one of <<CATS>>.\n"
    "- intent: one of <<INTENTS>>.\n"
    "- priority: low, medium, or high. Use high for payment problems, complaints, or angry or urgent "
    "customers; medium for cancellations, changes, refunds, and access problems; low for information requests.\n"
    "- reply: a short, polite draft reply for a human agent to review. Never invent order numbers, amounts, "
    "dates, or timeframes; use placeholders such as {{Order Number}} for anything the customer did not state.\n"
    "Output JSON only, with no other text."
).replace("<<CATS>>", ", ".join(CATEGORIES)).replace("<<INTENTS>>", ", ".join(INTENTS))

# Rule-based priority used to create the training labels. The app re-computes it as an independent cross-check.
HIGH = {"payment_issue", "complaint"}
MEDIUM = {"cancel_order", "change_order", "change_shipping_address", "contact_human_agent", "delete_account",
          "get_refund", "recover_password", "registration_problems", "track_refund"}
URGENT = re.compile(
    r"\b(asap|urgent(?:ly)?|immediately|right now|right away|emergency|ridiculous|unacceptable|furious|angry|"
    r"outrageous|terrible|worst|fed up)\b|!{2,}", re.I)


def base_priority(intent: str) -> str:
    return "high" if intent in HIGH else "medium" if intent in MEDIUM else "low"


def derive_priority(intent: str, text: str, flags: str = "") -> str:
    """Intent decides the base level; urgent or angry wording raises it one level."""
    lvl = LEVELS.index(base_priority(intent))
    if URGENT.search(text or "") or "W" in (flags or ""):
        lvl = min(lvl + 1, 2)
    return LEVELS[lvl]


# --------------------------------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------------------------------
TRAIN_TYPICAL_CHARS = 100      # training messages averaged 47 characters and never exceeded 92
MAX_INPUT_CHARS = 1500         # hard limit for the app (keeps generation time bounded)


def clean_message(text: str) -> str:
    """Collapse whitespace; the model was trained on single-line customer messages."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def input_warnings(text: str) -> list[str]:
    """Plain-language notes about messages the model was not trained for. Never blocks, only informs."""
    notes = []
    n = len(text)
    if n == 0:
        return ["The message is empty."]
    if n > MAX_INPUT_CHARS:
        notes.append(f"The message is {n} characters; only the first {MAX_INPUT_CHARS} will be used.")
    elif n > TRAIN_TYPICAL_CHARS * 3:
        notes.append("This message is much longer than the short messages the model was trained on "
                     "(they averaged about 47 characters), so its labels are less reliable.")
    if len(re.findall(r"[.?!]\s+\S", text)) >= 3:
        notes.append("The message seems to contain several separate requests. The model picks only one intent, "
                     "so the agent should check whether the others were missed.")
    if len(re.findall(r"[A-Za-z]{2,}", text)) < 2:
        notes.append("The message has very little text to classify.")
    return notes


# --------------------------------------------------------------------------------------------------------------
# Output parsing and validation (mirrors the Milestone 3 evaluation code)
# --------------------------------------------------------------------------------------------------------------
def parse_output(text: str):
    """Return (object or None, strict). strict=True means the whole output was exactly one JSON object."""
    t = (text or "").strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass
    t2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    s, e = t2.find("{"), t2.rfind("}")
    if s != -1 and e > s:
        try:
            obj = json.loads(t2[s:e + 1])
            if isinstance(obj, dict):
                return obj, False
        except Exception:
            pass
    return None, False


def schema_problems(obj) -> list[str]:
    """Empty list = the object has exactly the expected keys and allowed values."""
    if not isinstance(obj, dict):
        return ["the output is not a JSON object"]
    problems = []
    missing, extra = SCHEMA_KEYS - set(obj), set(obj) - SCHEMA_KEYS
    if missing:
        problems.append("missing keys: " + ", ".join(sorted(missing)))
    if extra:
        problems.append("unexpected keys: " + ", ".join(sorted(extra)))
    if "category" in obj and obj["category"] not in CATEGORIES:
        problems.append(f"category '{obj['category']}' is not one of the {len(CATEGORIES)} known categories")
    if "intent" in obj and obj["intent"] not in INTENTS:
        problems.append(f"intent '{obj['intent']}' is not one of the {len(INTENTS)} known intents")
    if "priority" in obj and obj["priority"] not in LEVELS:
        problems.append(f"priority '{obj['priority']}' is not low, medium or high")
    if "reply" in obj and not (isinstance(obj["reply"], str) and obj["reply"].strip()):
        problems.append("the reply is empty")
    return problems


def schema_ok(obj) -> bool:
    return not schema_problems(obj)


def _digits(text: str) -> set:
    return set(re.findall(r"\d+", text))


def invented_numbers(reply: str, user_text: str) -> list[str]:
    """Digits in the reply that are not in the customer's message (ignoring placeholders and list numbering)."""
    body = PH.sub(" ", reply or "")
    body = re.sub(r"(?m)^\s*\d+[.)]\s*", " ", body)       # "1. Log in" style numbering
    body = re.sub(r"\b24\s*/\s*7\b", " ", body)
    return sorted(_digits(body) - _digits(user_text or ""))


def placeholders_in(reply: str) -> list[str]:
    """Unique placeholder names in order of first appearance."""
    seen, out = set(), []
    for name in PH.findall(reply or ""):
        name = name.strip()
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def suggest_values(user_text: str) -> dict:
    """Pre-fill placeholders with values the customer actually wrote (the agent can still change them)."""
    out = {}
    m = re.search(r"(?:order|#)\s*(?:number|no\.?|num)?\s*#?\s*(\d{4,})", user_text or "", re.I)
    if m:
        out["Order Number"] = m.group(1)
    m = re.search(r"\b(INV-\d{3,})\b", user_text or "", re.I)
    if m:
        out["Invoice Number"] = m.group(1)
    return out


def fill_placeholders(reply: str, values: dict) -> str:
    """Replace {{Name}} with the agent's value; placeholders left blank stay visible so none is sent by mistake."""
    def sub(m):
        v = (values or {}).get(m.group(1).strip(), "")
        return v.strip() if v and v.strip() else m.group(0)
    return PH.sub(sub, reply or "")


# --------------------------------------------------------------------------------------------------------------
# Confidence from token probabilities
# --------------------------------------------------------------------------------------------------------------
def value_confidence(token_texts: list[str], token_probs: list[float], key: str) -> Optional[float]:
    """Probability the model gave to the string it wrote for `key` (e.g. "intent"), as the product of the
    probabilities of the tokens that make up that value. This is the model's own certainty. It is NOT calibrated:
    it tells how sure the model was, not how often it is right."""
    if not token_texts or len(token_texts) != len(token_probs):
        return None
    full = "".join(token_texts)
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"([^"]*)"', full)
    if not m:
        return None
    start, end = m.start(1), m.end(1)
    pos, logp, hit = 0, 0.0, False
    for tok, p in zip(token_texts, token_probs):
        a, b = pos, pos + len(tok)
        pos = b
        if b > start and a < end:           # this token overlaps the value's characters
            hit = True
            logp += math.log(max(p, 1e-12))
    return math.exp(logp) if hit else None


# --------------------------------------------------------------------------------------------------------------
# From raw model text to a reviewed triage result
# --------------------------------------------------------------------------------------------------------------
CONFIDENCE_FLOOR = 0.80   # below this the app asks the agent to double-check the label


@dataclass
class Check:
    name: str
    status: str            # "ok" | "warn" | "fail"
    detail: str


@dataclass
class TriageResult:
    user_text: str
    raw_output: str
    parsed: Optional[dict] = None
    strict_json: bool = False
    category: str = ""
    intent: str = ""
    model_priority: str = ""
    rule_priority: str = ""
    final_priority: str = ""
    reply: str = ""
    placeholders: list = field(default_factory=list)
    intent_confidence: Optional[float] = None
    category_confidence: Optional[float] = None
    checks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    seconds: float = 0.0
    n_tokens: int = 0

    @property
    def parsed_ok(self) -> bool:
        return self.parsed is not None and not schema_problems(self.parsed)

    @property
    def verdict(self) -> str:
        if not self.parsed_ok:
            return "manual"          # could not use the output: the agent handles this ticket without the draft
        if any(c.status == "fail" for c in self.checks) or any(c.status == "warn" for c in self.checks):
            return "check"
        return "ready"

    @property
    def verdict_text(self) -> str:
        return {"ready": "Ready for agent review",
                "check": "Needs a closer look - see the flagged checks",
                "manual": "Could not use the model output - handle this ticket manually"}[self.verdict]

    def to_record(self) -> dict:
        return {"message": self.user_text, "category": self.category, "intent": self.intent,
                "priority": self.final_priority, "model_priority": self.model_priority,
                "rule_priority": self.rule_priority,
                "intent_confidence": None if self.intent_confidence is None else round(self.intent_confidence, 3),
                "verdict": self.verdict, "flags": "; ".join(c.name for c in self.checks if c.status != "ok"),
                "draft_reply": self.reply}


def analyze(user_text: str, raw_output: str, token_texts: Optional[list] = None,
            token_probs: Optional[list] = None, seconds: float = 0.0,
            confidence_floor: float = CONFIDENCE_FLOOR) -> TriageResult:
    """Turn the model's raw text into a result plus the list of safety checks the agent should see."""
    res = TriageResult(user_text=user_text, raw_output=raw_output, seconds=seconds,
                       n_tokens=len(token_texts or []), warnings=input_warnings(user_text))
    obj, strict = parse_output(raw_output)
    res.parsed, res.strict_json = obj, strict

    # 1. is the output JSON at all?
    if obj is None:
        res.checks.append(Check("valid JSON", "fail", "No JSON object could be read from the model output."))
        return res
    res.checks.append(Check("valid JSON", "ok" if strict else "warn",
                            "The output is exactly one JSON object." if strict else
                            "The JSON had to be extracted from surrounding text."))

    # 2. does it follow the schema?
    problems = schema_problems(obj)
    if problems:
        res.checks.append(Check("schema", "fail", "; ".join(problems)))
        res.reply = obj.get("reply", "") if isinstance(obj.get("reply"), str) else ""
        return res
    res.checks.append(Check("schema", "ok", "All four fields are present with allowed values."))

    res.category, res.intent = obj["category"], obj["intent"]
    res.model_priority, res.reply = obj["priority"], obj["reply"].strip()
    res.placeholders = placeholders_in(res.reply)

    # 3. category must belong to the intent (the model writes them independently)
    expected = INTENT_TO_CATEGORY[res.intent]
    if expected == res.category:
        res.checks.append(Check("category fits intent", "ok", f"{res.intent} belongs to {expected}."))
    else:
        res.checks.append(Check("category fits intent", "fail",
                                f"The intent {res.intent} belongs to {expected}, but the model wrote {res.category}."))

    # 4. independent rule-based priority vs the model's priority
    res.rule_priority = derive_priority(res.intent, res.user_text)
    if res.rule_priority == res.model_priority:
        res.checks.append(Check("priority agrees with rule", "ok", f"Both say {res.model_priority}."))
    else:
        res.checks.append(Check("priority agrees with rule", "warn",
                                f"The model says {res.model_priority}; the keyword rule says {res.rule_priority}. "
                                "The higher of the two is shown."))
    res.final_priority = max(res.model_priority, res.rule_priority, key=LEVELS.index)

    # 5. numbers in the reply that the customer never wrote
    inv = invented_numbers(res.reply, res.user_text)
    if inv:
        res.checks.append(Check("no invented numbers", "fail",
                                "The draft contains numbers the customer did not write: " + ", ".join(inv)
                                + ". Remove or verify them before sending."))
    else:
        res.checks.append(Check("no invented numbers", "ok", "Every number in the draft comes from the customer."))

    # 6. how sure was the model about the intent?
    if token_texts and token_probs:
        res.intent_confidence = value_confidence(token_texts, token_probs, "intent")
        res.category_confidence = value_confidence(token_texts, token_probs, "category")
    if res.intent_confidence is None:
        res.checks.append(Check("model certainty", "ok", "Token probabilities were not available."))
    elif res.intent_confidence < confidence_floor:
        res.checks.append(Check("model certainty", "warn",
                                f"The model gave its intent only {res.intent_confidence:.0%} probability "
                                f"(below {confidence_floor:.0%}); confirm the label."))
    else:
        res.checks.append(Check("model certainty", "ok", f"Intent probability {res.intent_confidence:.0%}."))

    # 7. placeholders are information for the agent, not a problem
    if res.placeholders:
        res.checks.append(Check("placeholders to fill", "ok",
                                "Fill these before sending: " + ", ".join(res.placeholders) + "."))
    return res


# --------------------------------------------------------------------------------------------------------------
# Batch helpers
# --------------------------------------------------------------------------------------------------------------
MAX_BATCH = 50


def read_messages_csv(data: bytes | str, column: str = "message") -> list[str]:
    """Read customer messages from a CSV that has a header row. Uses `column`, or the first column if absent."""
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    idx = header.index(column.lower()) if column.lower() in header else 0
    out = []
    for r in rows[1:]:
        if len(r) > idx and clean_message(r[idx]):
            out.append(clean_message(r[idx]))
    return out


def records_to_csv(records: Iterable[dict]) -> str:
    records = list(records)
    if not records:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
    return buf.getvalue()


def sort_queue(results: Iterable[TriageResult]) -> list[TriageResult]:
    """Most urgent first; tickets the model could not handle go to the top because a person must read them."""
    def key(r: TriageResult):
        if not r.parsed_ok:
            return (-3, 0)                   # below every priority level (high = -2)
        return (-LEVELS.index(r.final_priority), 0 if r.verdict == "check" else 1)
    return sorted(results, key=key)


# --------------------------------------------------------------------------------------------------------------
# Example messages for the demo buttons
# --------------------------------------------------------------------------------------------------------------
EXAMPLES = {
    "Cancel an order": "hey so i accidentally ordered the wrong size of those running shoes yesterday, order 77231, can you cancel it before it ships??",
    "Card declined": "my card keeps getting declined at checkout even though i know theres money on it, whats going on!!",
    "Where is my package?": "Where is my package? Order 52089 was supposed to arrive three days ago.",
    "Formal complaint": "I'm really unhappy with how my last order was handled. The package was damaged and nobody replied to my emails. I want to file a formal complaint.",
    "Password reset": "I forgot my password and the reset email never arrives. Help me get back into my account.",
    "Refund question": "What's your policy on refunds if the item doesn't fit?",
    "Two requests in one": "Please change my delivery address to 14 Elm Street. Also I want to know where my refund is. And can I get an invoice for July?",
    "Off-topic (not a support ticket)": "What is the capital of Australia and who won the football game last night?",
}
