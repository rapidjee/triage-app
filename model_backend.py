"""model_backend.py - loads the fine-tuned model and turns one customer message into raw model text.

Two backends share one interface (`generate(user_text, max_new_tokens) -> GenResult`):

* TransformersBackend - the real thing: loads the fully fine-tuned Qwen2.5-0.5B-Instruct saved by the
  Milestone 3 notebook (the `final_model/` folder) and runs greedy decoding on it.
* StubBackend - a keyword-matching stand-in used ONLY for unit tests and for checking the page layout
  without a model. It is never selected automatically; the app shows a red banner whenever it is active.

PyTorch and transformers are imported lazily, so importing this module (and running the unit tests) does not
require them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import triage_core as core

DEFAULT_MODEL_DIR = "final_model"        # the folder the Milestone 3 notebook saves
MAX_NEW_TOKENS = 640                     # same limit as the Milestone 3 evaluation


class ModelNotFound(RuntimeError):
    """Raised with a human-readable explanation of where the model should be and how to get it."""


@dataclass
class GenResult:
    text: str
    token_texts: list = field(default_factory=list)
    token_probs: list = field(default_factory=list)
    seconds: float = 0.0


# --------------------------------------------------------------------------------------------------------------
# Finding the model
# --------------------------------------------------------------------------------------------------------------
def resolve_model_source(source: str | None = None) -> str:
    """Return a local folder that holds the model, or a Hugging Face repo id such as 'username/triage-qwen'.

    Order: explicit argument, then the TRIAGE_MODEL environment variable, then ./final_model.
    """
    source = source or os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL_DIR
    p = Path(source).expanduser()
    if p.is_dir():
        if not (p / "config.json").exists():
            raise ModelNotFound(f"The folder '{p}' exists but has no config.json, so it is not a saved model.")
        return str(p)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", source) and not source.startswith((".", "/")) and not p.exists():
        return source                                    # looks like a Hub repo id; transformers will download it
    raise ModelNotFound(
        f"Could not find the fine-tuned model at '{source}'.\n"
        "Put the `final_model` folder produced by the Milestone 3 notebook next to app.py, or set the "
        "TRIAGE_MODEL environment variable to its location (or to a Hugging Face repo id such as "
        "'your-username/support-triage-qwen0.5b').")


def folder_fingerprint(path: str) -> str:
    """Short hash over config.json plus the names and sizes of the weight files.
    It identifies WHICH saved model is loaded (useful in screenshots and bug reports); it is not a security check."""
    p = Path(path)
    if not p.is_dir():
        return "hub:" + path
    h = hashlib.sha256()
    cfg = p / "config.json"
    if cfg.exists():
        h.update(cfg.read_bytes())
    for f in sorted(p.glob("*.safetensors")) + sorted(p.glob("pytorch_model*.bin")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------------------------------------------
# Real backend
# --------------------------------------------------------------------------------------------------------------
class TransformersBackend:
    name = "fine-tuned model"
    is_stub = False

    def __init__(self, source: str | None = None, device: str | None = None):
        self.source = resolve_model_source(source)
        import torch                                     # lazy: only needed when a real model is loaded
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 is fast on a GPU; on a CPU float32 is the safe, supported choice (0.5B parameters = about 2 GB).
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.source, dtype=dtype)
        except TypeError:                                # older transformers call the argument torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.source, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.dtype = str(dtype).replace("torch.", "")
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.fingerprint = folder_fingerprint(self.source)
        self._stop_ids = {i for i in (self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                      self.tokenizer.eos_token_id) if i is not None}
        self._lock = threading.Lock()                    # one generation at a time; Streamlit may serve several tabs

    def describe(self) -> dict:
        return {"backend": self.name, "source": self.source, "fingerprint": self.fingerprint,
                "parameters": self.n_params, "device": self.device, "dtype": self.dtype}

    def build_prompt(self, user_text: str) -> str:
        """Exactly the chat format used for training: system prompt + customer message, then the assistant turn."""
        messages = [{"role": "system", "content": core.SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, user_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> GenResult:
        torch = self.torch
        prompt = self.build_prompt(user_text)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        t0 = time.time()
        with self._lock, torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                temperature=None, top_p=None, top_k=None, pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True, return_dict_in_generate=True)
        seconds = time.time() - t0

        new_ids = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
        token_ids, token_texts, token_probs = [], [], []
        for step, tok_id in enumerate(new_ids):
            if tok_id in self._stop_ids:                 # the end-of-turn token is not part of the answer
                break
            # probability the model assigned to the token it actually chose (greedy = the most likely one)
            logp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[tok_id]
            token_ids.append(tok_id)
            token_texts.append(self.tokenizer.decode([tok_id], clean_up_tokenization_spaces=False))
            token_probs.append(float(logp.exp()))
        # the displayed text is decoded in one piece (safe for multi-byte characters); the per-token strings
        # are only used to locate the intent/category values when computing confidence
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return GenResult(text=text, token_texts=token_texts, token_probs=token_probs, seconds=seconds)


# --------------------------------------------------------------------------------------------------------------
# Stand-in backend for tests and layout checks (NOT a model)
# --------------------------------------------------------------------------------------------------------------
_STUB_RULES = [  # (regex, intent) - first match wins
    (r"cancel", "cancel_order"), (r"declin|card|payment", "payment_issue"), (r"complain|unhappy", "complaint"),
    (r"password", "recover_password"), (r"where is my (package|order)|track", "track_order"),
    (r"refund", "get_refund"), (r"invoice", "get_invoice"), (r"address", "change_shipping_address"),
]


class StubBackend:
    name = "STUB (keyword rules, not a model)"
    is_stub = True

    def __init__(self, *_, **__):
        self.source = "none"
        self.fingerprint = "stub"
        self.n_params = 0
        self.device, self.dtype = "none", "none"

    def describe(self) -> dict:
        return {"backend": self.name, "source": self.source, "fingerprint": self.fingerprint,
                "parameters": 0, "device": "none", "dtype": "none"}

    def generate(self, user_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> GenResult:
        intent = next((i for rx, i in _STUB_RULES if re.search(rx, user_text, re.I)), "contact_customer_service")
        obj = {"category": core.INTENT_TO_CATEGORY[intent], "intent": intent,
               "priority": core.derive_priority(intent, user_text),
               "reply": "Thank you for contacting us. Regarding order {{Order Number}}, a support agent "
                        "will follow up shortly. [stub reply]"}
        text = json.dumps(obj, ensure_ascii=False)
        # fake per-character "tokens" with a fixed probability so the confidence code path is exercised
        return GenResult(text=text, token_texts=list(text), token_probs=[0.95] * len(text), seconds=0.0)


def load_backend(source: str | None = None, backend: str | None = None):
    """backend='stub' (or TRIAGE_BACKEND=stub) selects the stand-in; anything else loads the real model."""
    backend = (backend or os.environ.get("TRIAGE_BACKEND") or "model").lower()
    if backend == "stub":
        return StubBackend()
    return TransformersBackend(source)
