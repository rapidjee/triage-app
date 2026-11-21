# Support Ticket Triage

A Streamlit app that reads a customer-support message and gives a human agent a **category, an intent, a priority and a draft reply**. The model behind it is **Qwen2.5-0.5B-Instruct, fully fine-tuned** (all 494 M parameters, no LoRA) on the Bitext customer-support dataset. The app never sends anything to a customer: a person reviews every ticket.

Course project for *Advanced Uses of Generative AI* (Milestone 4).

<!-- Add your screenshots to docs/screenshots/ and uncomment:
![Triage a ticket](docs/screenshots/01_triage.png)
![Checks and queue](docs/screenshots/02_checks_queue.png)
-->

## What the app does

1. The agent pastes a message (or uploads a CSV for batch mode).
2. The fine-tuned model answers with one JSON object: `category`, `intent`, `priority`, `reply`.
3. `triage_core.py` checks the answer before the agent sees it:
   - valid JSON with exactly the expected keys and allowed values;
   - the category belongs to the intent (the model writes them separately);
   - the priority agrees with an independent keyword rule (if not, the higher one is shown);
   - no digits in the draft that the customer did not write;
   - the model's own probability for its intent (below 80% asks for a double-check);
   - `{{Placeholders}}` in the reply are listed and become input boxes, so none is sent by mistake.
4. The agent fills the placeholders, edits the draft, records a decision and downloads the reply.

Tabs: **Triage a ticket**, **Batch (CSV)**, **Session queue** (most urgent first), **About the model** (training curves and evaluation).

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/support-ticket-triage.git
cd support-ticket-triage
pip install -r requirements.txt
```

**Get the model.** The weights (about 1 GB) are not in git. Use one of:

* put the `final_model/` folder produced by the Milestone 3 notebook next to `app.py`; or
* host it on the Hugging Face Hub and run `export TRIAGE_MODEL=your-username/support-triage-qwen0.5b`; or
* `export TRIAGE_MODEL=/path/to/final_model`.

```bash
python check_model.py     # proves the loaded weights are the fine-tuned ones (10 probe messages)
streamlit run app.py
```

No GPU? Open `colab/run_app_in_colab.ipynb` in Google Colab: it loads the model, runs `check_model.py` and gives you a public link to the app. On a CPU the app works but each ticket takes roughly 15 to 40 seconds.

## How we know the app uses the fine-tuned model

* The app loads weights only from `TRIAGE_MODEL` or `./final_model`, shows the path, parameter count and a weights fingerprint in the sidebar, and refuses to start if the model is missing. There is no silent fallback to the base model.
* `check_model.py` sends 10 messages and requires at least 9 valid tickets. The base model with the same prompt produced a valid ticket for only 46% of test messages; the fine-tuned model for 100%.
* The sidebar counts how many outputs passed the format check during the session.
* A keyword-rule stand-in (`TRIAGE_BACKEND=stub`) exists for tests and layout work only. The app shows a red banner whenever it is active.

## Results (from the Milestone 3 notebook)

| | base zero-shot | base + 3 examples | **fine-tuned** |
|---|---|---|---|
| Test set (270 messages): valid format | 45.9% | 54.8% | **100%** |
| Test set: intent accuracy | 17.8% | 33.3% | **99.6%** |
| Stress set (27 hand-written): intent accuracy | 25.9% | 37.0% | **81.5%** |
| Stress set: priority accuracy | 25.9% | 25.9% | **92.6%** |
| Replies with invented numbers (stress) | 7.4% | 3.7% | **0%** |

Validation loss went from 1.395 to 0.646 (perplexity 4.03 to 1.91) in 20.7 minutes on a Colab T4. It was lowest at step 204 (0.616), so epoch 3 overfits slightly.

## Limitations

* The Bitext data is synthetic and templated (customer messages average 47 characters). The 99.6% test score shows the model learned this dataset, not that it will do as well on real email. The 27 hand-written stress messages are a more honest estimate.
* The model chooses one of 27 fixed intents, so multi-request or off-topic messages are forced into one. The app warns about long or multi-request messages.
* Probabilities are the model's own certainty, not calibrated hit rates.
* In one stress case the model invented an intent label (`accept_payment_methods`). The schema check catches this and sends the ticket to manual handling.
* Messages are kept only in the browser session's memory; nothing is stored or sent elsewhere.

## Repository layout

```
app.py              Streamlit interface
triage_core.py      prompt, validation, priority rule, confidence, batch helpers (no torch, no streamlit)
model_backend.py    loads the fine-tuned model and generates (greedy decoding, token probabilities)
check_model.py      verifies the loaded weights are the fine-tuned ones
assets/results.json real training log and evaluation tables used by the About tab
samples/            probe messages; the exact system prompt used in training
tests/              unit tests (pytest) and a Streamlit AppTest
colab/              run the app on a free Colab GPU
notebooks/          Milestone 3 training notebook (with outputs)
presentation/       Milestone 4 slides
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The unit tests cover the prompt (identical to the training prompt), parsing, every safety check, confidence maths, placeholder handling, batch/CSV helpers and model discovery. They need no GPU. `tests/test_app.py` drives the real page with Streamlit's headless `AppTest` using the stub backend, so it tests page logic, not the language model.

## Data and licences

Training data: [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Base model: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Check both licences before redistributing the weights.
