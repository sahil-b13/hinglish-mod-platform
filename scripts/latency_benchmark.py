"""
Latency benchmark: measures per-message inference latency (p50/p95/p99) for a
classifier. Works with any classifier implementing .predict(text), so it runs
right now against the placeholder RuleBasedClassifier as a baseline, and later
against HingBertClassifier / an ONNX-exported model with no changes needed —
just point --mode at what you want to measure.

This is the "measured systems contribution" the plan (§9 stretch goal) wants
alongside "I fine-tuned a model" — a concrete latency number, not just a claim.

Run now (placeholder baseline):
    python scripts/latency_benchmark.py --mode placeholder

Later, once you have a real model:
    python scripts/latency_benchmark.py --mode pytorch --model_path models/hingroberta-finetuned
    python scripts/latency_benchmark.py --mode onnx --model_path models/hingroberta-onnx
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))

# A mix of short/medium/long realistic Hinglish messages, since latency can vary
# with input length — benchmarking on only one length gives a misleading number.
BENCH_TEXTS = [
    "ok",
    "bhai theek hai",
    "yaar aaj match kitna accha tha, India ne jeet liya",
    "kal weather kaisa rahega, koi update hai kya, mujhe bahar jaana hai shaam ko",
    "tu ek number ka chutiya hai, kabhi kuch sahi nahi karta, teri wajah se sabka nuksan hota hai",
    "yeh bilkul wahi scam hai jo pichhle mahine bhi chal raha tha, sabko savdhan rehna chahiye "
    "kyunki yeh log alag alag naam se yehi scheme baar baar chala rahe hain",
]


def get_classifier(mode: str, model_path: str = None):
    if mode == "placeholder":
        from classifier import RuleBasedClassifier
        return RuleBasedClassifier()

    if mode == "pytorch":
        if not model_path:
            raise SystemExit("--model_path required for --mode pytorch")
        from classifier import HingBertClassifier
        return HingBertClassifier(model_path=model_path)

    if mode == "onnx":
        if not model_path:
            raise SystemExit("--model_path required for --mode onnx")
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer
        except ImportError:
            raise SystemExit("Run: pip install optimum[onnxruntime] onnx onnxruntime")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ORTModelForSequenceClassification.from_pretrained(model_path)

        class OnnxWrapper:
            def predict(self, text):
                import numpy as np
                inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
                import torch
                outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
                logits = outputs.logits.detach().numpy() if hasattr(outputs.logits, "detach") else outputs.logits
                pred = int(np.argmax(logits))
                return {"label": str(pred), "confidence": float(np.max(logits))}

        return OnnxWrapper()

    raise SystemExit(f"Unknown mode: {mode}")


def percentile(values: list, p: float) -> float:
    values = sorted(values)
    idx = int(len(values) * p)
    idx = min(idx, len(values) - 1)
    return values[idx]


def main(args):
    classifier = get_classifier(args.mode, args.model_path)

    print(f"Benchmarking mode='{args.mode}' over {args.iterations} iterations "
          f"across {len(BENCH_TEXTS)} sample texts of varying length...")

    # warmup — first call often includes lazy init overhead, don't count it
    for text in BENCH_TEXTS:
        classifier.predict(text)

    latencies_ms = []
    start_total = time.perf_counter()
    for _ in range(args.iterations):
        for text in BENCH_TEXTS:
            t0 = time.perf_counter()
            classifier.predict(text)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
    total_elapsed = time.perf_counter() - start_total

    n = len(latencies_ms)
    print(f"\nTotal calls: {n} in {total_elapsed:.2f}s "
          f"({n / total_elapsed:.0f} msgs/sec throughput, single-threaded)")
    print(f"Latency (ms): p50={percentile(latencies_ms, 0.50):.3f}  "
          f"p95={percentile(latencies_ms, 0.95):.3f}  "
          f"p99={percentile(latencies_ms, 0.99):.3f}  "
          f"mean={statistics.mean(latencies_ms):.3f}  "
          f"max={max(latencies_ms):.3f}")

    print("\nNote: this is single-message, single-threaded, CPU/no-batching latency —")
    print("representative of the per-message consumer stage in this pipeline's current")
    print("design, not a claim about throughput at real platform scale (plan §7 is explicit")
    print("that this project hasn't been load-tested at thousands of msg/sec).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier latency benchmark")
    parser.add_argument("--mode", choices=["placeholder", "pytorch", "onnx"], default="placeholder")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--iterations", type=int, default=50,
                         help="how many passes over the BENCH_TEXTS set")
    args = parser.parse_args()
    main(args)
