# Real-Time Hinglish Moderation Platform

Streaming moderation system for Hinglish (Hindi-English code-mixed) chat: classifies
messages, tracks per-user trust/velocity, and detects coordinated campaigns via
SimHash + Union-Find clustering.

This repo runs **end-to-end today** using synthetic sample data (`data/sample/`) so you
can see the whole pipeline work before plugging in real datasets and a fine-tuned model.

---

## 0. What's already done vs. what YOU need to do

| Piece | Status |
|---|---|
| Docker Compose (Redpanda + Redis) | ✅ Done |
| Kafka producer (simulator) | ✅ Done — uses `data/sample/sample_messages.jsonl` for now |
| Kafka consumer pipeline (normalize → classify → simhash → trust) | ✅ Done — classifier stage uses a placeholder rule-based scorer until you plug in a real model |
| SimHash + Union-Find campaign clustering | ✅ Done, tested, threshold empirically calibrated |
| Redis trust score + rate limiting | ✅ Done, tested, velocity limit empirically calibrated |
| Streamlit dashboard | ✅ Done |
| Data prep/cleaning + quality report | ✅ Done, tested (`scripts/prepare_data.py`) |
| Model bake-off (HingBERT/HingRoBERTa/MuRIL) | ✅ Script written — **you run it**, needs HF access + GPU for finetune mode |
| Fine-tuning script (HingBERT/HingRoBERTa on Bohra) | ✅ Script written — **you run it**, needs GPU + real data |
| Tier 3 adversarial test set + runner | ✅ Done, tested (`data/tricky_test_messages.md`, `scripts/run_tricky_tests.py`) |
| Active learning / moderator feedback loop | ✅ Done, tested — dashboard review panel + `scripts/export_feedback_queue.py` |
| Latency benchmark harness (stretch) | ✅ Done, tested — runs today against placeholder, ready for real model |
| ONNX export + verification (stretch) | ✅ Script written — **you run it**, needs a real fine-tuned checkpoint |
| Real datasets (Bohra 2018, PRISM, HateCheckHIn) | ❌ **You download these** (see §2) |
| Real model weights (HingBERT/HingRoBERTa) | ❌ **You download these** (see §2) |
| Actually running training / bake-off | ❌ **You run this on Colab/Kaggle** (see §2b/§3) |

Why: my sandbox can only reach pypi/npm/github — not Hugging Face, Kaggle, or Twitter's
API, which is where the real data and model weights live. Everything code-related is
built and runnable; the data/model fetching has to happen on your machine or in Colab.

---

## 1. Quick start (runs right now, with synthetic data)

```bash
# 1. Start infra
docker compose up -d
# (or `make up`)

# 2. Install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# (or `make install`)

# 3. Sanity-check the pipeline logic without Kafka (needs only Redis up)
python scripts/smoke_test.py
# (or `make smoke-test`)

# 4. Terminal A: start the consumer (pipeline)
python consumer/consumer.py

# 5. Terminal B: start the producer (replays data/sample/sample_messages.jsonl)
python producer/producer.py

# 6. Terminal C: dashboard
streamlit run dashboard/app.py
# (or `make dashboard`)
```

You should see messages flow through Kafka → consumer → Redis, with verdicts and
campaign clusters showing up live in the dashboard. The classifier is currently a
**placeholder rule-based scorer** (keyword + heuristic) — good enough to prove the
pipeline works, not good enough to claim "AI moderation." Swap it for the real model
per §3.

If Kafka/Redpanda gives you setup friction (a known risk per the plan, §7), run
`scripts/smoke_test.py` first — it exercises the full classify → cluster → trust
pipeline directly against the sample data with only Redis running, so you can
confirm the pipeline logic itself is correct independent of the broker.

---

## 1b. Tools for the "empirical tuning" parts of the plan

The plan (§6/§7) is explicit that the SimHash threshold and classifier quality need
real tuning, not values copied from a tutorial. Two scripts help with that:

**`python scripts/tune_simhash_threshold.py`** (or `make tune-threshold`) — sweeps
Hamming-distance thresholds against labeled same-campaign / different-campaign
message pairs and reports recall vs. false-merge count at each threshold, so you
pick a number with actual precision/recall behind it. Pre-loaded with pairs from
the sample spam block; swap in your own labeled pairs once you have real injected
spam (Week 6).

**`python scripts/run_tricky_tests.py`** (or `make tricky-tests`) — runs the 24
hand-written adversarial test messages in `data/tricky_test_messages.md` (sarcasm,
negation, reclaimed slang, obfuscation, script-mixing) through the classifier and
reports where it disagrees with the expected human judgment, broken down by
category. This is the Tier 3 testing and failure-mode documentation the plan (§8)
asks for. Run this again once you've swapped in the real fine-tuned model — the
placeholder scorer currently fails most negation/sarcasm cases by design (it's
pure keyword matching), which is worth showing explicitly in your report as the
reason a real model is needed.

**`python scripts/tune_rate_limits.py`** (or `make tune-rate-limits`) — same idea
applied to `DEFAULT_VELOCITY_LIMIT`/`TIGHTENED_VELOCITY_LIMIT` in
`consumer/trust.py` (plan §6, Week 5). Simulates a few user behavior patterns
(casual chatter, active legit user, burst spammer, slow-drip spammer) and reports
which candidate limits cause false positives on legitimate users vs. missing real
spam bursts. This is how the current default (20 msgs/min) was picked — the
original naive guess of 12 turned out to false-positive on a fast-but-legitimate
active user in the simulation. Re-run with real behavior once you have it.

---

## 2. Getting the real datasets and model (you do this locally, not in this chat)

**HingBERT / HingRoBERTa (model weights)** — CC-BY-4.0, free, no registration:
```bash
pip install transformers torch
python -c "from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('l3cube-pune/hing-roberta'); \
AutoModel.from_pretrained('l3cube-pune/hing-roberta')"
```
This downloads from Hugging Face automatically the first time you run it, as long as
you have internet access (your machine will, even though my sandbox doesn't).

**Bohra et al. 2018 dataset** — public tweet-ID release. Because it's ID-only, you need
to "hydrate" the IDs into full tweets yourself (Twitter/X API access, or a hydration
tool like `twarc`). Search: "Bohra 2018 hindi-english code-mixed hate speech dataset
github" — the repo has the ID list and hydration instructions.

**PRISM dataset** — on Hugging Face, MIT license:
```python
from datasets import load_dataset
ds = load_dataset("<PRISM dataset repo — search Hugging Face for 'PRISM code-mixed'>")
```

**HateCheckHIn** — public GitHub repo with the diagnostic test suite. Search
"HateCheckHIn github" and clone it into `data/hatecheckhin/`.

Once you have these, drop them into:
```
data/bohra/          # raw + processed Bohra data
data/prism/          # PRISM augmentation set
data/hatecheckhin/   # diagnostic suite
```

---

## 2b. Cleaning your data and picking a model (Week 1, plan §5)

Once you've hydrated Bohra (or pulled PRISM), don't feed it straight into
`scripts/train.py` — run it through the prep/inspection step first:

```bash
python scripts/prepare_data.py \
    --raw_csv data/bohra/raw_hydrated.csv \
    --text_col tweet_text --label_col hate_label \
    --label_map "none:0,offensive:1,hateful:2" \
    --output_dir data/bohra
```

This drops missing/empty/duplicate rows, prints class balance and text-length
stats, warns if one class dominates (>85%) — which macro-F1 is sensitive to —
and does a stratified train/test split so rare classes show up in both splits.
This is the "look at the data yourself" step the plan (§6) says not to skip;
the printed report is what to actually look at, not just the row count.

Then run the Week 1 model bake-off to lock in HingBERT vs HingRoBERTa vs MuRIL
with a reason instead of an assumption:

```bash
# fast sanity check, no training
python scripts/model_bakeoff.py --data_dir data/bohra --mode zero_shot

# slower but meaningful: small fine-tune + macro-F1 comparison per model
python scripts/model_bakeoff.py --data_dir data/bohra --mode finetune --epochs 2 --sample_size 2000
```

`--mode finetune` is the one that actually answers the question — budget
15-30 min per model on a free Colab GPU. Whichever wins on macro-F1 is what you
pass as `--model_name` to the real `scripts/train.py` run.

---

## 3. Fine-tuning (you run this, I can't — needs a GPU)

`scripts/train.py` is a full Hugging Face `Trainer`-based fine-tuning script, ready to
run once you have Bohra data in `data/bohra/train.csv` and `data/bohra/test.csv`
(columns: `text`, `label`). Upload the repo to Colab/Kaggle, attach a free GPU runtime,
and run:

```bash
python scripts/train.py --data_dir data/bohra --model_name l3cube-pune/hing-roberta --output_dir models/hingroberta-finetuned
```

Watch the loss curve, decide when it's converged (this is the "your judgment" part —
see plan §6). When done, point `consumer/classifier.py` at the checkpoint in
`models/hingroberta-finetuned` instead of the placeholder scorer (see the `TODO` in
that file — it's a single function swap).

---

## 4. Project structure

```
hinglish-mod-platform/
├── docker-compose.yml       # Redpanda + Redis
├── requirements.txt
├── Makefile                 # convenience commands (make up/install/test/...)
├── producer/producer.py     # simulated chat stream → Kafka
├── consumer/
│   ├── consumer.py          # main pipeline loop
│   ├── classifier.py        # placeholder scorer (swap for real model)
│   ├── simhash_cluster.py   # SimHash + Union-Find campaign detection
│   ├── trust.py             # Redis trust score + rate limiting
│   └── feedback.py          # moderator confirm/overturn store (active learning loop)
├── dashboard/app.py         # Streamlit live dashboard + moderator review panel
├── scripts/
│   ├── prepare_data.py              # raw data cleaning + quality report + train/test split
│   ├── model_bakeoff.py             # Week 1: HingBERT vs HingRoBERTa vs MuRIL comparison
│   ├── train.py                     # HF Trainer fine-tuning script (run on Colab/Kaggle)
│   ├── evaluate.py                  # macro-F1 + HateCheckHIn evaluation
│   ├── tune_simhash_threshold.py    # empirical SimHash threshold calibration
│   ├── tune_rate_limits.py          # empirical velocity limit calibration
│   ├── run_tricky_tests.py          # Tier 3 adversarial test runner
│   ├── smoke_test.py                # Kafka-free end-to-end pipeline check
│   ├── export_feedback_queue.py     # active learning loop: feedback -> training CSV
│   ├── latency_benchmark.py         # p50/p95/p99 latency + throughput (stretch)
│   └── export_onnx.py               # ONNX export + verification (stretch)
├── data/
│   ├── sample/sample_messages.jsonl   # synthetic Hinglish messages for local testing
│   └── tricky_test_messages.md        # 24 hand-written adversarial test cases
└── tests/                   # unit tests for simhash/union-find/trust logic
```

---

## 9. Stretch goals — what's ready to run

**Active learning / moderator feedback loop** (plan §9) — the dashboard now has a
"Moderator review" panel: for any flagged/held-for-review message, click Confirm
or pick a corrected label and click Overturn. Decisions are stored in Redis
(`consumer/feedback.py`). Export them into training-ready CSV rows with:

```bash
python scripts/export_feedback_queue.py --output data/bohra/feedback_batch_1.csv
```

Tested end-to-end with synthetic moderator decisions — confirms/overturns are
recorded correctly, already-reviewed messages don't reappear in the panel, and
the export correctly maps corrected labels to the training label scheme. Appends
onto `data/bohra/train.csv` for your next fine-tuning cycle; the script also
warns if the overturn rate is high enough (>30%) that you should investigate the
classifier's specific failure pattern (e.g. via `run_tricky_tests.py`-style
category breakdown) before assuming more data alone fixes it.

**Latency benchmarking** (`scripts/latency_benchmark.py`) — works right now
against the placeholder classifier for a baseline, and against a real model with
no code changes once you have one:

```bash
python scripts/latency_benchmark.py --mode placeholder --iterations 200   # runs today
python scripts/latency_benchmark.py --mode pytorch --model_path models/hingroberta-finetuned
python scripts/latency_benchmark.py --mode onnx --model_path models/hingroberta-onnx
```

Reports p50/p95/p99 latency and throughput. This is the "measured systems
contribution" the plan (§9) calls out as turning "I fine-tuned a model" into an
actual number.

**ONNX export** (`scripts/export_onnx.py`) — exports a fine-tuned checkpoint to
ONNX and verifies its outputs match the original PyTorch model within tolerance
before you trust any latency numbers from it:

```bash
pip install optimum[onnxruntime] onnx onnxruntime
python scripts/export_onnx.py --model_path models/hingroberta-finetuned --output_path models/hingroberta-onnx
```

**Not yet started** (per plan §9, lower priority — only pursue if core finishes
early): PRISM as OOD eval, Marathi extension via MeBERT/L3Cube-MeCorpus, real
Discord/Telegram bot integration, token-level explainability (why a message was
flagged). I've held off scaffolding these because they either need real API
credentials I can't test against, or a real fine-tuned model to be meaningful —
happy to build them once you're further along, so I'm not guessing at
integration details I can't verify.

---

## 5. Next steps (map to your Week 1-2 deliverables)

- [x] Kafka/Redpanda + Redis local setup
- [x] Chat-stream simulator (currently on synthetic data — swap source in
      `producer/producer.py` once you have Bohra/PRISM downloaded)
- [x] Data prep/cleaning tooling ready (`scripts/prepare_data.py`) — run it once you
      have hydrated Bohra data
- [x] Model bake-off tooling ready (`scripts/model_bakeoff.py`) — run it on Colab/Kaggle
      once you have data prepped, to lock in HingBERT vs HingRoBERTa vs MuRIL with a reason
- [x] End-to-end message flow working, classified with placeholder scorer
- [x] SimHash threshold empirically calibrated against synthetic spam (re-run
      `scripts/tune_simhash_threshold.py` once you have real injected spam)
- [x] Tier 3 adversarial test set ready (`data/tricky_test_messages.md` — re-run
      `scripts/run_tricky_tests.py` once you've swapped in the real fine-tuned model)

Everything from Week 3 onward (real fine-tuning, HateCheckHIn pass) depends on you
having pulled the real data/model per §2, and running §2b/§3 on a GPU.
