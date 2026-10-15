"""
Stretch goal: export a fine-tuned HingBERT/HingRoBERTa checkpoint to ONNX and
verify the exported model's outputs match the original PyTorch model within
tolerance. This turns "I fine-tuned a model" into a measured systems
contribution (plan §9), and pairs with scripts/latency_benchmark.py to report
actual latency numbers.

**You run this** — needs a real fine-tuned checkpoint (from scripts/train.py)
and `pip install onnx onnxruntime optimum`.

Run:
    python scripts/export_onnx.py --model_path models/hingroberta-finetuned --output_path models/hingroberta-onnx
"""

import argparse
from pathlib import Path


def main(args):
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
    except ImportError:
        raise SystemExit(
            "Missing dependency. Run: pip install optimum[onnxruntime] onnx onnxruntime"
        )

    from transformers import AutoTokenizer
    import torch
    import numpy as np

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Exporting to ONNX (this downloads/uses the optimum ONNX export path)...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path, export=True
    )
    ort_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved ONNX model to {output_path}")

    # Sanity check: compare ONNX output to the original PyTorch model on a few
    # sample texts, since a silent export bug (wrong opset, mismatched labels)
    # is the kind of thing you want caught before you report latency numbers.
    from transformers import AutoModelForSequenceClassification

    print("\nVerifying ONNX output matches original PyTorch model...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pt_model.eval()

    test_texts = [
        "yeh product bilkul sahi hai",
        "tu bilkul bewakoof hai",
        "tumhe barbaad kar denge dekh lena",
    ]

    max_diff = 0.0
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            pt_logits = pt_model(**inputs).logits.numpy()

        ort_inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
        ort_output = ort_model(**{k: torch.tensor(v) for k, v in ort_inputs.items()})
        ort_logits = ort_output.logits.detach().numpy() if hasattr(ort_output.logits, "detach") else ort_output.logits

        diff = float(np.max(np.abs(pt_logits - ort_logits)))
        max_diff = max(max_diff, diff)
        print(f"  '{text[:40]}...' max logit diff: {diff:.6f}")

    if max_diff < 1e-3:
        print(f"\nONNX export verified: max diff {max_diff:.6f} across test texts (well within tolerance).")
    else:
        print(f"\n[!] WARNING: max diff {max_diff:.6f} is larger than expected — "
              f"inspect before trusting the ONNX model's outputs.")

    print(f"\nNext: python scripts/latency_benchmark.py --pytorch_path {model_path} "
          f"--onnx_path {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned model to ONNX")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
