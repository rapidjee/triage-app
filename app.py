"""Support Ticket Triage - Streamlit app (Milestone 4).

A customer writes a message; the fully fine-tuned Qwen2.5-0.5B-Instruct returns a category, an intent, a priority
and a draft reply as JSON. This page checks that output, shows it to a human support agent, lets the agent fill the
{{placeholders}}, edit the draft and decide. Nothing is sent to the customer automatically.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import model_backend as mb
import triage_core as core

st.set_page_config(page_title="Support Ticket Triage", layout="wide")

ROOT = Path(__file__).parent
PRIORITY_COLOR = {"high": "red", "medium": "orange", "low": "green"}
STATUS_LABEL = {"ok": "OK", "warn": "CHECK", "fail": "FAIL"}


# --------------------------------------------------------------------------------------------------------------
# Loading (cached so the model is read from disk once per server process, not on every click)
# --------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the fine-tuned model (first start only)...")
def get_backend(source: str, kind: str):
    return mb.load_backend(source or None, kind or None)


@st.cache_data
def get_results() -> dict:
    p = ROOT / "assets" / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def init_state():
    st.session_state.setdefault("history", [])       # list of dicts: {"id": int, "result": TriageResult}
    st.session_state.setdefault("next_id", 1)
    st.session_state.setdefault("current_id", None)
    st.session_state.setdefault("n_gen", 0)
    st.session_state.setdefault("n_valid", 0)
    st.session_state.setdefault("msg", "")


def set_example(name: str):
    st.session_state["msg"] = core.EXAMPLES[name]


def run_one(backend, text: str, floor: float) -> tuple[int, core.TriageResult]:
    """Generate, analyse and record one ticket in this browser session's history."""
    text = core.clean_message(text)[: core.MAX_INPUT_CHARS]
    gen = backend.generate(text, mb.MAX_NEW_TOKENS)
    res = core.analyze(text, gen.text, gen.token_texts, gen.token_probs, gen.seconds, confidence_floor=floor)
    rid = st.session_state["next_id"]
    st.session_state["next_id"] += 1
    st.session_state["history"].append({"id": rid, "result": res})
    st.session_state["n_gen"] += 1
    st.session_state["n_valid"] += int(res.parsed_ok)
    return rid, res


# --------------------------------------------------------------------------------------------------------------
# Page parts
# --------------------------------------------------------------------------------------------------------------
def sidebar(backend):
    info = backend.describe()
    st.sidebar.header("Model")
    st.sidebar.write(f"**{info['backend']}**")
    if backend.is_stub:
        st.sidebar.error("Demo mode: this is NOT the fine-tuned model.")
    else:
        st.sidebar.write(f"Loaded from: `{info['source']}`")
        st.sidebar.write(f"Parameters: {info['parameters'] / 1e6:.0f} M (all trained, no LoRA)")
        st.sidebar.write(f"Running on: {info['device']} ({info['dtype']})")
        st.sidebar.write(f"Weights fingerprint: `{info['fingerprint']}`")
    st.sidebar.caption("The base model, Qwen2.5-0.5B-Instruct, was fully fine-tuned on 1,620 labelled support "
                       "messages from the Bitext dataset (Milestone 3).")

    st.sidebar.header("Settings")
    floor = st.sidebar.slider("Ask for a double-check below this intent probability", 0.50, 0.99,
                              core.CONFIDENCE_FLOOR, 0.01,
                              help="The model's own probability for the intent it wrote. It is not calibrated: "
                                   "a high value does not guarantee the label is right.")

    st.sidebar.header("This session")
    n, v = st.session_state["n_gen"], st.session_state["n_valid"]
    st.sidebar.write(f"Tickets triaged: **{n}**")
    st.sidebar.write(f"Outputs that passed the format check: **{v}/{n}**" if n else
                     "Outputs that passed the format check: -")
    st.sidebar.caption("For reference, the base model without fine-tuning produced a valid format in only 46% "
                       "of test messages; the fine-tuned model in 100%.")

    st.sidebar.header("Privacy")
    st.sidebar.caption("Messages are processed on the machine running this app and kept only in this browser "
                       "session's memory. Nothing is written to disk or sent to another service.")
    return floor


def render_result(rid: int, res: core.TriageResult):
    """Everything the agent sees for one ticket."""
    banner = {"ready": st.success, "check": st.warning, "manual": st.error}[res.verdict]
    banner(res.verdict_text)
    for w in res.warnings:
        st.info(w)

    if not res.parsed_ok:
        st.write("The model's raw output could not be turned into a ticket:")
        st.code(res.raw_output or "(empty)", language="json")
        for c in res.checks:
            st.write(f"**{STATUS_LABEL[c.status]}** - {c.name}: {c.detail}")
        return

    left, mid, right, last = st.columns(4)
    left.metric("Category", res.category)
    mid.metric("Intent", res.intent)
    if res.intent_confidence is not None:
        mid.caption(f"model probability {res.intent_confidence:.0%}")
    right.markdown("**Priority**")
    right.markdown(f"## :{PRIORITY_COLOR[res.final_priority]}[{res.final_priority.upper()}]")
    right.caption(f"model: {res.model_priority} | keyword rule: {res.rule_priority}")
    last.metric("Generation time", f"{res.seconds:.1f} s")
    last.caption(f"{res.n_tokens} tokens")

    st.subheader("Draft reply")
    values = core.suggest_values(res.user_text)
    if res.placeholders:
        st.caption("The model wrote placeholders where the customer gave no value. Fill them in, then review.")
        cols = st.columns(min(3, len(res.placeholders)))
        entered = {}
        for i, name in enumerate(res.placeholders):
            entered[name] = cols[i % len(cols)].text_input(name, value=values.get(name, ""), key=f"ph_{rid}_{name}")
    else:
        entered = {}
    filled = core.fill_placeholders(res.reply, entered)
    still_open = core.placeholders_in(filled)
    edited = st.text_area("Draft (editable)", value=filled, height=220, key=f"reply_{rid}_{abs(hash(filled))}")
    if still_open:
        st.warning("Still to fill before sending: " + ", ".join(still_open))

    st.subheader("Checks")
    st.table(pd.DataFrame([{"Result": STATUS_LABEL[c.status], "Check": c.name, "Detail": c.detail}
                           for c in res.checks]))

    st.radio("Agent decision", ["Not decided", "Send as drafted", "Send after editing", "Rewrite from scratch"],
             key=f"dec_{rid}", horizontal=True)
    c1, c2 = st.columns(2)
    c1.download_button("Download reply (.txt)", edited, file_name=f"reply_{rid}.txt", key=f"dl_{rid}")
    c2.download_button("Download ticket (.json)",
                       json.dumps({**res.to_record(), "final_reply": edited}, indent=2, ensure_ascii=False),
                       file_name=f"ticket_{rid}.json", key=f"dj_{rid}")
    with st.expander("Raw model output"):
        st.code(res.raw_output, language="json")


def tab_single(backend, floor):
    st.write("Paste a customer message (or choose an example), then press **Triage**.")
    ex_col, btn_col = st.columns([3, 1])
    choice = ex_col.selectbox("Examples", list(core.EXAMPLES), label_visibility="collapsed")
    btn_col.button("Use example", on_click=set_example, args=(choice,))
    st.text_area("Customer message", key="msg", height=110,
                 placeholder="e.g. Where is my package? Order 52089 was supposed to arrive three days ago.")
    if st.button("Triage", type="primary"):
        text = core.clean_message(st.session_state["msg"])
        if not text:
            st.error("Please enter a message first.")
        else:
            with st.spinner("The model is reading the message and writing a draft..."):
                rid, _ = run_one(backend, text, floor)
            st.session_state["current_id"] = rid
    cur = st.session_state["current_id"]
    if cur is not None:
        item = next((h for h in st.session_state["history"] if h["id"] == cur), None)
        if item:
            st.divider()
            st.caption(f"Ticket #{cur}")
            render_result(cur, item["result"])


def tab_batch(backend, floor):
    st.write(f"Triage up to {core.MAX_BATCH} messages at once, then work through them most urgent first.")
    up = st.file_uploader("CSV with a header row and a 'message' column", type=["csv"])
    pasted = st.text_area("...or paste one message per line", height=120, key="batch_text")
    if st.button("Triage all", type="primary"):
        msgs = core.read_messages_csv(up.getvalue()) if up is not None else \
            [core.clean_message(x) for x in pasted.splitlines() if core.clean_message(x)]
        if not msgs:
            st.error("No messages found.")
            return
        if len(msgs) > core.MAX_BATCH:
            st.warning(f"Only the first {core.MAX_BATCH} of {len(msgs)} messages will be processed.")
            msgs = msgs[: core.MAX_BATCH]
        bar, results = st.progress(0.0), []
        for i, m in enumerate(msgs, 1):
            _, res = run_one(backend, m, floor)
            results.append(res)
            bar.progress(i / len(msgs))
        ordered = core.sort_queue(results)
        df = pd.DataFrame([r.to_record() for r in ordered])
        st.success(f"Triaged {len(results)} messages. "
                   f"{sum(r.verdict == 'ready' for r in results)} ready, "
                   f"{sum(r.verdict == 'check' for r in results)} need a closer look, "
                   f"{sum(r.verdict == 'manual' for r in results)} need manual handling.")
        st.dataframe(df)
        st.download_button("Download results (.csv)", core.records_to_csv(df.to_dict("records")),
                           file_name="triage_results.csv")


def tab_queue():
    hist = st.session_state["history"]
    if not hist:
        st.info("No tickets yet in this session.")
        return
    by_res = {id(h["result"]): h for h in hist}
    ordered = core.sort_queue([h["result"] for h in hist])
    rows = []
    for r in ordered:
        h = by_res[id(r)]
        rec = r.to_record()
        rec = {"ticket": h["id"], **{k: rec[k] for k in ("priority", "intent", "verdict", "message")},
               "agent_decision": st.session_state.get(f"dec_{h['id']}", "Not decided")}
        rows.append(rec)
    st.write("Most urgent first. Tickets the model could not handle are listed at the top.")
    st.dataframe(pd.DataFrame(rows))
    if st.button("Clear session history"):
        st.session_state["history"], st.session_state["current_id"] = [], None
        st.session_state["n_gen"] = st.session_state["n_valid"] = 0
        st.rerun()


def tab_about():
    res = get_results()
    st.subheader("What this app does")
    st.write(
        "A customer message goes to a fully fine-tuned Qwen2.5-0.5B-Instruct (494 million parameters, every weight "
        "updated; no LoRA). The model answers with a JSON object holding a category, an intent, a priority and a "
        "draft reply. The app does not trust that output blindly: it checks the format, checks that the category "
        "belongs to the intent, compares the priority with an independent keyword rule, looks for numbers the "
        "customer never wrote, and shows the model's own probability for the intent. A human agent always makes the "
        "final decision.")
    if not res:
        st.info("assets/results.json was not found, so the evaluation numbers are not shown.")
        return

    st.subheader("Training (Milestone 3, Colab T4)")
    t = res["training"]
    a, b, c, d = st.columns(4)
    a.metric("Validation loss", f"{t['val_loss_after']:.3f}", f"{t['val_loss_after'] - t['val_loss_before']:+.3f} vs start",
             delta_color="inverse")
    b.metric("Perplexity", f"{t['perplexity_after']:.2f}", f"from {t['perplexity_before']:.2f}", delta_color="off")
    c.metric("Token accuracy", f"{t['token_acc_after']:.1%}", f"from {t['token_acc_before']:.1%}", delta_color="off")
    d.metric("Training time", f"{t['minutes']} min", f"peak {t['peak_gpu_gb']} GB GPU", delta_color="off")
    tr = pd.DataFrame(res["train_log"]).set_index("step")["loss"].rename("training loss")
    va = pd.DataFrame(res["val_log"]).set_index("step")["val_loss"].rename("validation loss")
    st.line_chart(pd.concat([tr, va], axis=1))
    st.caption(f"Validation loss was lowest at step {t['val_loss_min_step']} ({t['val_loss_min']:.3f}) and rose "
               "slightly afterwards, a sign of mild overfitting in the third epoch.")

    st.subheader("Base model vs fine-tuned model")
    labels = {"schema_valid_rate": "valid output format", "category_acc": "category correct",
              "intent_acc": "intent correct", "priority_acc": "priority correct",
              "invented_number_rate": "replies with invented numbers (lower is better)",
              "placeholder_recall": "placeholders reproduced", "rougeL_mean": "reply overlap (ROUGE-L)"}
    which = st.radio("Evaluation set", ["test (270 messages from the dataset)", "stress (27 hand-written messages)"],
                     horizontal=True)
    key = "test" if which.startswith("test") else "stress"
    rows = {labels[k]: v for k, v in res[key].items() if k in labels and None not in v}
    df = pd.DataFrame(rows, index=res["columns"]).T
    st.dataframe(df.round(3))
    st.bar_chart(df.loc[[labels["schema_valid_rate"], labels["category_acc"], labels["intent_acc"],
                         labels["priority_acc"]]])

    st.subheader("Limitations")
    for note in res["notes"]:
        st.write("- " + note)
    st.write("- The training data is synthetic and templated, and its customer messages average 47 characters. "
             "Real emails are longer, messier and often contain several requests.")
    st.write("- The model picks one of 27 fixed intents. A message about something else is still forced into one of them.")
    st.write("- The model also invented the label `accept_payment_methods` in one stress case (not a real intent); "
             "the schema check catches this and sends the ticket to manual handling.")
    st.write("- Model probabilities show how sure the model is, not how often it is right.")


# --------------------------------------------------------------------------------------------------------------
def main():
    init_state()
    st.title("Support Ticket Triage")
    st.caption("Fine-tuned Qwen2.5-0.5B-Instruct | category, intent, priority and a draft reply for a human agent")

    try:
        backend = get_backend("", "")      # model location comes from TRIAGE_MODEL or ./final_model
    except mb.ModelNotFound as e:
        st.error("The fine-tuned model was not found.")
        st.code(str(e))
        st.stop()
    if backend.is_stub:
        st.error("DEMO MODE: the keyword stand-in is running, not the fine-tuned model. "
                 "Unset TRIAGE_BACKEND to use the real model.")

    floor = sidebar(backend)
    t1, t2, t3, t4 = st.tabs(["Triage a ticket", "Batch (CSV)", "Session queue", "About the model"])
    with t1:
        tab_single(backend, floor)
    with t2:
        tab_batch(backend, floor)
    with t3:
        tab_queue()
    with t4:
        tab_about()


main()
