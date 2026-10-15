"""
Evaluate a fine-tuned checkpoint: macro-F1 on held-out Bohra test data, plus (once
you've cloned HateCheckHIn per README §2) a pass through its diagnostic test suite.

Run:
    python scripts/evaluate.py --model_path models/hingroberta-finetuned --test_csv data/bohra/test.csv

For the HateCheckHIn pass, once you've cloned that repo into data/hatecheckhin/,
point --hatecheck_csv at its test-case CSV (check that repo's README for the exact
filename/columns — format varies by release) and this script will run the same
model against it and break down accuracy by functional test category, which is
the "here's exactly where it breaks" analysis the plan (§4) calls for.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score


def evaluate_split(model_path: str, csv_path: Path, label_col: str = "label"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    preds = []
    with torch.no_grad():
        for text in df["text"]:
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128)
            logits = model(**inputs).logits
            preds.append(int(torch.argmax(logits, dim=-1).item()))

    df["pred"] = preds
    return df


def main(args):
    test_path = Path(args.test_csv)
    df = evaluate_split(args.model_path, test_path)

    macro_f1 = f1_score(df["label"], df["pred"], average="macro")
    print(f"\nHeld-out macro-F1: {macro_f1:.4f}")
    print("\nFull classification report:")
    print(classification_report(df["label"], df["pred"]))

    if args.hatecheck_csv:
        hc_path = Path(args.hatecheck_csv)
        if not hc_path.exists():
            print(f"\n[!] --hatecheck_csv given but {hc_path} not found. "
                  f"See README §2 for how to get HateCheckHIn.")
        else:
            hc_df = evaluate_split(args.model_path, hc_path)
            if "category" in hc_df.columns:
                print("\nHateCheckHIn accuracy by functional category:")
                hc_df["correct"] = hc_df["pred"] == hc_df["label"]
                print(hc_df.groupby("category")["correct"].mean().sort_values())
            else:
                acc = (hc_df["pred"] == hc_df["label"]).mean()
                print(f"\nHateCheckHIn overall accuracy: {acc:.4f} "
                      f"(no 'category' column found for the per-category breakdown — "
                      f"check the HateCheckHIn CSV schema and adjust this script)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned classifier")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--hatecheck_csv", default=None,
                         help="Optional path to HateCheckHIn diagnostic CSV")
    args = parser.parse_args()
    main(args)
