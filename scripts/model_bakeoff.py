"""
Week 1 model bake-off: compares HingBERT vs HingRoBERTa vs MuRIL on a slice of your
labeled data, so the model choice going into fine-tuning (plan §5, Week 1) is
locked in with an actual reason instead of an assumption.

**You run this** once you have data/bohra/train.csv (or any labeled slice) and
internet access to Hugging Face — my sandbox can't reach huggingface.co.

Two comparison modes, pick based on time budget:

1. --mode zero_shot (default, fast): loads each base model's masked-LM head and
   measures perplexity/fill-mask behavior on a sample of your text as a rough
   "does this model already understand Hinglish" sanity check. Good for a quick
   first pass, not a real proxy for classification quality.

2. --mode finetune (slower, more meaningful): fine-tunes each candidate model for
   a small number of epochs on a subset of your training data and compares
   macro-F1 on a held-out slice. This is the number you actually want for the
   bake-off decision the plan asks for — budget ~15-30 min per model on a free
   Colab GPU for a few thousand rows.

Run:
    python scripts/model_bakeoff.py --data_dir data/bohra --mode finetune --epochs 2 --sample_size 2000
"""

import argparse
import time
from pathlib import Path

import pandas as pd

CANDIDATES = {
    "hingbert": "l3cube-pune/hing-bert",
    "hingroberta": "l3cube-pune/hing-roberta",
    "muril": "google/muril-base-cased",
}


def run_zero_shot_check(model_name: str, texts: list[str]):
    """Rough sanity check: average fill-mask confidence on masked tokens from your
    actual data. Higher isn't strictly 'better' but very low confidence across the
    board suggests the model's vocabulary/tokenizer doesn't fit Hinglish well."""
    from transformers import pipeline
    import numpy as np

    try:
        fill_mask = pipeline("fill-mask", model=model_name)
    except Exception as e:
        return {"error": str(e)}

    mask_token = fill_mask.tokenizer.mask_token
    scores = []
    for text in texts[:30]:
        words = text.split()
        if len(words) < 3:
            continue
        mid = len(words) // 2
        masked = " ".join(words[:mid] + [mask_token] + words[mid + 1:])
        try:
            result = fill_mask(masked, top_k=1)
            scores.append(result[0]["score"])
        except Exception:
            continue

    return {
        "mean_confidence": float(np.mean(scores)) if scores else None,
        "n_samples": len(scores),
    }


def run_finetune_check(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                        epochs: int, sample_size: int):
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, DataCollatorWithPadding,
    )

    train_sub = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_sub[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)

    num_labels = train_df["label"].nunique()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=f"/tmp/bakeoff_{model_name.replace('/', '_')}",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    elapsed = time.time() - start

    return {"macro_f1": metrics.get("eval_macro_f1"), "seconds": round(elapsed, 1)}


def main(args):
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"Bake-off across: {list(CANDIDATES.keys())}")
    print(f"Mode: {args.mode}\n")

    results = {}
    for short_name, hf_name in CANDIDATES.items():
        print(f"--- {short_name} ({hf_name}) ---")
        try:
            if args.mode == "zero_shot":
                result = run_zero_shot_check(hf_name, train_df["text"].tolist())
            else:
                result = run_finetune_check(hf_name, train_df, test_df, args.epochs, args.sample_size)
        except Exception as e:
            result = {"error": str(e)}
        results[short_name] = result
        print(result, "\n")

    print("=== Summary ===")
    for name, result in results.items():
        print(f"{name:15s}: {result}")

    if args.mode == "finetune":
        scored = {k: v["macro_f1"] for k, v in results.items() if v.get("macro_f1") is not None}
        if scored:
            best = max(scored, key=scored.get)
            print(f"\nBest by macro-F1: {best} ({scored[best]:.4f})")
            print("Use this in scripts/train.py's --model_name for the full fine-tune.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 1 model bake-off: HingBERT vs HingRoBERTa vs MuRIL")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", choices=["zero_shot", "finetune"], default="zero_shot")
    parser.add_argument("--epochs", type=int, default=2, help="only used in finetune mode")
    parser.add_argument("--sample_size", type=int, default=2000,
                         help="rows of training data to use per candidate in finetune mode "
                              "(keep small — this is a quick comparison, not the real fine-tune)")
    args = parser.parse_args()
    main(args)
