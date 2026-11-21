"""check_model.py - confirm that the model the app will load really is the fine-tuned model.

Run it once after copying `final_model/` next to app.py (or after setting TRIAGE_MODEL):

    python check_model.py                 # ten probe messages
    python check_model.py --model my-user/support-triage-qwen0.5b

Why this works: the base Qwen2.5-0.5B-Instruct, prompted the same way, produced a valid JSON ticket for only 46% of the
Milestone 3 test messages. A fully fine-tuned checkpoint produces one for essentially every message. So if at least
9 of 10 probe messages come back as a valid ticket, you are looking at the fine-tuned weights, not the base model.
"""
import argparse
import json
import sys
from pathlib import Path

import model_backend as mb
import triage_core as core


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="folder or Hugging Face repo id (default: TRIAGE_MODEL or ./final_model)")
    ap.add_argument("--min-valid", type=int, default=9, help="how many of the probes must be valid tickets")
    args = ap.parse_args()

    probes = json.loads((Path(__file__).parent / "samples" / "probe.json").read_text())
    backend = mb.load_backend(args.model)
    print("loaded:", backend.describe())

    n_valid = n_intent = 0
    for p in probes:
        gen = backend.generate(p["message"])
        res = core.analyze(p["message"], gen.text, gen.token_texts, gen.token_probs, gen.seconds)
        n_valid += res.parsed_ok
        n_intent += res.parsed_ok and res.intent == p["expected_intent"]
        print(f"{'ok  ' if res.parsed_ok else 'FAIL'} {res.intent or '-':26s} expected {p['expected_intent']:26s} "
              f"({gen.seconds:.1f}s)  {p['message'][:50]}")
    print(f"\nvalid tickets: {n_valid}/{len(probes)}   correct intent: {n_intent}/{len(probes)}")

    if getattr(backend, "is_stub", False):
        print("This is the stub backend, not a model. Nothing was verified.")
        return 2
    if n_valid >= args.min_valid:
        print("PASS - the loaded weights behave like the fine-tuned model.")
        return 0
    print("FAIL - too many invalid outputs. This looks like the BASE model or a damaged checkpoint.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
